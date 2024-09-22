from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pydantic import BaseModel
from scipy.optimize import OptimizeResult
from tqdm import tqdm

from ck_fridge_sim.control.control_base import ControllerSetting, Disturbances
from ck_fridge_sim.control.control_system import ControlSystem
from ck_fridge_sim.control.controllers import ControlSystemState
from ck_fridge_sim.equipment_models import Measurement, TimePoint
from ck_fridge_sim.process_model import ProcessModel
from ck_fridge_sim.process_states import FullProcessState
from ck_fridge_sim.solver.solvers import RK12, FixedStepEuler


class OdeResult(OptimizeResult):
    pass


class DynamicSimulation(BaseModel):
    start_datetime: datetime
    end_datetime: datetime
    process_model: ProcessModel
    control_system: ControlSystem
    process_states: FullProcessState
    control_states: ControlSystemState
    disturbances: Disturbances
    settings: list[ControllerSetting]
    solver: FixedStepEuler | RK12
    result_measurements: list[list[Measurement]] = []
    result_actuators: list[list[TimePoint]] = []

    def load_state_from_json(
        self, process_state_path: Path, control_state_path: Path
    ) -> None:
        self.process_states = FullProcessState.load_from_json(process_state_path)
        self.control_states = ControlSystemState.load_from_json(control_state_path)
        self.start_datetime = self.process_states.equipment_state_list[0].date_time
        for state in self.process_states.equipment_state_list:
            if state.date_time != self.start_datetime:
                raise ValueError(
                    f"All states must have the same date_time. Error in {state.name}"
                )

        for state in self.control_states.control_states:
            state.date_time = self.start_datetime

    def dump_state_to_json(
        self, process_state_path: Path, control_state_path: Path
    ) -> None:
        self.process_states.dump_to_json(process_state_path)
        self.control_states.dump_to_json(control_state_path)

    def simulate(
        self,
        action: np.ndarray, # ADDED
        verbose: bool = False,
        callback: Callable[[float], None] | None = None,
    ) -> tuple[
        OdeResult,
        list[list[Measurement]],
        list[list[TimePoint]],
    ]:
        time_delta = self.end_datetime - self.start_datetime
        # print(f"----------------------------------------")
        # print(f"Debug start time; {self.start_datetime}")
        # print(f"Debug end time; {self.end_datetime}")
        # print(f"Debug delta time; {time_delta}")
        unit, scale = self.get_unit_scale(time_delta)
        end_time__s = time_delta.total_seconds()

        time = 0
        time_list = []
        measurement_list = []
        actuator_list = []
        next_time_step__s = self.solver.initial_time_step__s
        state_vector = []

        if not callback:
            pbar = tqdm(
                total=end_time__s / scale,
                desc="",
                unit=unit,
                unit_scale=True,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            )
        while time <= end_time__s:
            simulation_time = self.start_datetime + timedelta(seconds=time)

            self.control_system.update_settings_by_action(self.settings, action, simulation_time)

            # print(f"Update settings with action: {action} at {time}")
            self.control_system.action(
                self.control_states,
                self.process_model.get_measurements(),
                self.process_model.get_actuators(),
                simulation_time,
            )

            step_result = self.solver.step(
                next_time_step__s,
                simulation_time,
                self.process_model,
                self.process_states,
                self.disturbances,
                verbose=verbose,
            )

            if step_result.error_message:
                print("------------Current Process States with Error-----------")
                for equipment_state in self.process_states.equipment_state_list:
                    print(f"Equipment: {equipment_state.name}")
                    # Assuming ModelState has attributes to represent its state, adjust accordingly
                    for attribute_name, attribute_value in equipment_state.__dict__.items():
                        print(f"  {attribute_name}: {attribute_value}")
                # print(f"Solver error at time {time}: {step_result.error_message}")  # Just print the error
                error_message = f"Solver error at time {time}: {step_result.error_message}"  # Assign the message to the variable
                raise RuntimeError(error_message)  # Raise the exception with the message

            self.process_states = step_result.new_process_state
            next_time_step__s = step_result.next_step_length__s

            time_list.append(time)
            time += step_result.step_length__s

            measurement_list.append(self.process_model.get_measurements())
            actuator_list.append(
                [
                    item.model_copy(deep=True)
                    for item in self.process_model.get_actuators().actuator_list
                ]
            )
            state_vector.append(self.process_states.state_vector)
            self.control_states.update(step_result.step_length__s)

            if callback:
                callback(time / end_time__s)
            else:
                pbar.update(round(step_result.step_length__s / scale, 6))
                pbar.set_postfix(
                    {
                        "Error": step_result.error,
                        "Step length (s)": step_result.step_length__s,
                        "Limiting Arg": step_result.limiting_equipment_name
                        + " | "
                        + step_result.limiting_state_name,
                    }
                )

            # final time step
            if time + next_time_step__s > end_time__s:
                next_time_step__s = end_time__s - time
                if next_time_step__s == 0:  # Changed from `== 0`
                    # print(f"Final time step error at time {time}: {next_time_step__s}")
                    break

        if not callback:
            pbar.close()

        if time < end_time__s:
            print(f"Simulation failed to reach end time: {end_time__s}")

        # for equipment_state in self.process_states.equipment_state_list:
        #     print(f"Equipment: {equipment_state.name}")
        #     # Assuming ModelState has attributes to represent its state, adjust accordingly
        #     for attribute_name, attribute_value in equipment_state.__dict__.items():
        #         print(f"  {attribute_name}: {attribute_value}")
                
        return (
            OdeResult(
                t=np.array(time_list),
                y=np.array(state_vector).T,
                success=True,
                message="Success" if time >= end_time__s else "Failed",
            ),
            measurement_list,
            actuator_list,
        )
    
    def is_hpr_controller(self):
        for setting in self.settings:
            if setting.name == "HPR-1|level_setpoint" and setting.controller_name == "hpr_level_control":
                return True
        return False

    def plot_results(
        self,
        df_results: pd.DataFrame,
        device_names: list[str],
        point_names: list[str],
    ) -> go.Figure:
        fig = make_subplots(
            rows=len(point_names),
            cols=1,
            subplot_titles=point_names,
            shared_xaxes=True,
            vertical_spacing=0.03,
        )

        for i, point_name in enumerate(point_names):
            for device_name in device_names:
                for field in df_results.filter(like=device_name).filter(
                    like=point_name
                ):
                    fig.add_trace(
                        go.Scatter(
                            x=df_results.index,
                            y=df_results[field],
                            mode="lines",
                            name=field,
                        ),
                        row=i + 1,
                        col=1,
                    )

        fig.update_layout(
            title_text="Dynamic Simulation Results",
            height=100 + len(point_names) * 300,
            width=1000,
            showlegend=True,
            template="plotly_white",
        )

        return fig

    def process_results(
        self,
        solution: OdeResult,
        measurement_list: list[list[Measurement]],
        actuators_list: list[list[TimePoint]],
    ) -> pd.DataFrame:
        # # Debugging: Print original solution.t and their types
        # print("Original solution.t values and types:")
        # for t in solution.t:
        #     print(f"value: {t}, type: {type(t)}")

        state_list = [
            self.process_states.set_from_vector(
                state.tolist(), self.start_datetime + timedelta(seconds=time)
            )
            for time, state in zip(solution.t, solution.y.T)
        ]

        points = [
            {
                **{
                    f"{s.name} | {field}": getattr(s, field)
                    for s in state
                    for field in s.state_model_fields
                },
                "time": self.start_datetime
                + timedelta(seconds=round(solution.t[i], 3)),
            }
            for i, state in enumerate(state_list)
        ]

        df_states = pd.DataFrame(points).set_index("time")
        df_states = df_states.loc[~df_states.index.duplicated(keep="first")]

        points = [
            {
                **{
                    f"{m.name} | {field}": getattr(m, field)
                    for m in measurement
                    for field in m.measurement_model_fields
                },
                "time": self.start_datetime
                + timedelta(seconds=round(solution.t[i], 3)),
            }
            for i, measurement in enumerate(measurement_list)
        ]

        df_measurements = pd.DataFrame(points).set_index("time")
        df_measurements = df_measurements.loc[
            ~df_measurements.index.duplicated(keep="first")
        ]

        points = [
            {
                **{
                    f"{a.name} | {field}": getattr(a, field)
                    for a in actuator
                    for field in a.time_point_model_fields
                },
                "time": self.start_datetime
                + timedelta(seconds=round(solution.t[i], 3)),
            }
            for i, actuator in enumerate(actuators_list)
        ]
        df_actuators = pd.DataFrame(points).set_index("time")
        df_actuators = df_actuators.loc[~df_actuators.index.duplicated(keep="first")]

        return pd.concat([df_states, df_measurements, df_actuators], axis=1)

    @staticmethod
    def get_unit_scale(time: timedelta) -> tuple[str, int]:
        time__s = time.total_seconds()
        if time__s > 3600:
            unit = "h"
            scale = 3600
        elif time__s > 60:
            unit = "m"
            scale = 60
        else:
            unit = "s"
            scale = 1
        return unit, scale