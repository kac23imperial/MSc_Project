import logging
import traceback
from abc import ABC, abstractmethod
from datetime import datetime, timedelta

import numpy as np
from pydantic import BaseModel

from ck_fridge_sim.control.control_base import Disturbances
from ck_fridge_sim.process_model import ProcessModel
from ck_fridge_sim.process_states import FullProcessState


class StepResult(BaseModel):
    step_length__s: float
    next_step_length__s: float
    new_process_state: FullProcessState
    error: float
    limiting_equipment_name: str | None = None
    limiting_state_name: str | None = None
    error_message: str | None = None


class SolverBase(BaseModel, ABC):
    initial_time_step__s: float = 1
    max_error_steps: int = 10

    @abstractmethod
    def _step(
        self,
        time_step__s: float,
        simulation_time: datetime,
        process_model: ProcessModel,
        process_states: FullProcessState,
        disturbances: Disturbances,
    ) -> StepResult:
        pass

    def step(
        self,
        last_step_length__s: float,
        simulation_time: datetime,
        process_model: ProcessModel,
        process_states: FullProcessState,
        disturbances: Disturbances,
        verbose: bool = False,
    ) -> StepResult:
        time_step__s = last_step_length__s
        n_error_steps = 0
        while n_error_steps < self.max_error_steps:
            try:
                step_result = self._step(
                    time_step__s,
                    simulation_time,
                    process_model,
                    process_states,
                    disturbances,
                )
                return step_result
            except Exception as ex:
                if verbose:
                    tb = traceback.format_exc()
                    logging.warning(f"Error {ex}\n{tb}")

                time_step__s *= 0.5
                n_error_steps += 1

        return StepResult(
            step_length__s=last_step_length__s,
            next_step_length__s=last_step_length__s,
            new_process_state=process_states,
            error=1,
            limiting_equipment_name=None,
            limiting_state_name=None,
            error_message=f"Error in solver step:\n{tb}",
        )


class RK12(SolverBase):
    max_step__s: float = 60
    relative_tolerance: float = 1e-2
    reject_buffer: float = 3
    max_error_steps: int = 10
    error_norm: str = "L2"  # L2 or Linf

    def _rk12(
        self,
        time_step__s: float,
        simulation_time: datetime,
        process_model: ProcessModel,
        process_states: FullProcessState,
        disturbances: Disturbances,
    ) -> tuple[FullProcessState, FullProcessState]:
        half_step_target_time = simulation_time + timedelta(seconds=time_step__s / 2)
        full_step_target_time = simulation_time + timedelta(seconds=time_step__s)

        k1_derivatives = process_model.get_state_derivatives(
            process_states, disturbances
        )
        k1_states = process_states.advance_states(k1_derivatives, half_step_target_time)
        k2_derivatives = process_model.get_state_derivatives(k1_states, disturbances)
        k2_states = k1_states.advance_states(k2_derivatives, full_step_target_time)
        average_derivatives = (k1_derivatives + k2_derivatives) * 0.5
        return process_states.advance_states(
            average_derivatives, full_step_target_time
        ), k2_states

    def _step(
        self,
        time_step__s: float,
        simulation_time: datetime,
        process_model: ProcessModel,
        process_states: FullProcessState,
        disturbances: Disturbances,
    ) -> tuple[float, float, tuple[str, str]]:
        n_steps = 0
        while n_steps < self.max_error_steps:
            # Perform RK2 and Euler steps
            states_rk2_step, states_dual_euler_step = self._rk12(
                time_step__s,
                simulation_time,
                process_model,
                process_states,
                disturbances,
            )
            error_vector = np.abs(
                np.array(states_rk2_step.state_vector)
                - np.array(states_dual_euler_step.state_vector)
            ) / (np.abs(np.array(process_states.state_vector) + 1e-4))

            if self.error_norm == "L2":
                # Calculate error 2 norm
                error = np.linalg.norm(error_vector)
            else:
                # calculate the infinity norm
                error = np.max(error_vector)

            limiting_arg = process_states.state_args_vector[np.argmax(error_vector)]
            # Adjust time step
            error = max(1e-6, error)
            correction_factor = min(
                1.1, max(0.6, (self.relative_tolerance / error) ** (1 / 2))
            )
            if error < self.relative_tolerance:
                next_step_length__s = min(
                    self.max_step__s, time_step__s * correction_factor
                )
                break
            else:
                next_step_length__s = time_step__s * correction_factor
                if error < self.relative_tolerance * self.reject_buffer:
                    break

                n_steps += 1

        return StepResult(
            step_length__s=time_step__s,
            next_step_length__s=next_step_length__s,
            new_process_state=states_dual_euler_step,
            error=error,
            limiting_equipment_name=limiting_arg[0],
            limiting_state_name=limiting_arg[1],
        )


class FixedStepEuler(SolverBase):
    step_length__s: float = 1

    def _step(
        self,
        time_step__s: float,
        simulation_time: datetime,
        process_model: ProcessModel,
        process_states: FullProcessState,
        disturbances: Disturbances,
    ) -> StepResult:
        target_time = simulation_time + timedelta(seconds=time_step__s)
        derivative = process_model.get_state_derivatives(process_states, disturbances)
        new_states = process_states.advance_states(derivative, target_time)
        return StepResult(
            step_length__s=time_step__s,
            next_step_length__s=self.step_length__s,
            new_process_state=new_states,
            error=0,
            limiting_equipment_name=None,
            limiting_state_name=None,
        )
