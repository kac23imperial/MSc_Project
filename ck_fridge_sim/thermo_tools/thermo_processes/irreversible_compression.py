from typing import Optional

import numpy as np

from ck_fridge_sim.thermo_tools.single_fluid_thermo import SingleFluidThermo
from ck_fridge_sim.thermo_tools.thermo_constants import ThermoConstants
from ck_fridge_sim.thermo_tools.thermo_processes.isentropic_pressure_change import (
    IsentropicPressureChange,
)
from ck_fridge_sim.thermo_tools.thermo_processes.thermo_process_base import (
    ThermoProcessBase,
)


class IrreversibleCompression(ThermoProcessBase):
    """
    This is a model of an irreversible compression process represented by an isentropic
    efficiency. The efficiency is defined as:
    efficiency = (h2s - h1) / (h2 - h1)
    where h1 is the initial enthalpy, h2 is the final enthalpy, and h2s is the final
    enthalpy if the process were isentropic.
    """

    def __init__(
        self,
        final_pressure__pa: float,
        isentropic_efficiency: float,
        initial_state: Optional[SingleFluidThermo] = None,
    ):
        """
        Parameters
        ----------
        final_pressure__pa : float
            The final pressure of the fluid in Pa.
        isentropic_efficiency : float
            The isentropic efficiency of compression (between 0 and 1).
        initial_state : SingleFluidThermo
            A SingleFluidThermo object (or subclass) that represents the initial
            thermodynamic state of the fluid.
        """
        super().__init__(initial_state=initial_state)
        self.final_pressure__pa = final_pressure__pa
        self.isentropic_efficiency = isentropic_efficiency
        assert 0 < self.isentropic_efficiency <= 1

    def _compute_actual_enthalpy(self, isentropic_enthalpy__kj_kg: float) -> float:
        """
        Compute the actual enthalpy of the fluid given the isentropic enthalpy.
        Uses the equation:
        h2 = h1 + (h2s - h1) / eta

        Parameters
        ----------
        isentropic_enthalpy__kj_kg : float
            The isentropic enthalpy of the fluid in kJ/kg.

        Returns
        -------
        float
            The actual enthalpy of the fluid in kJ/kg.
        """
        assert (
            self.initial_state is not None
        ), "Initial state must be set before calling _compute_actual_enthalpy."
        initial_enthalpy__kj_kg = self.initial_state.thermodynamic_state[
            ThermoConstants.ENTHALPY
        ][ThermoConstants.VALUE]
        actual_enthalpy__kj_kg = (
            initial_enthalpy__kj_kg
            + (isentropic_enthalpy__kj_kg - initial_enthalpy__kj_kg)
            / self.isentropic_efficiency
        )
        return actual_enthalpy__kj_kg

    def _get_new_state(self) -> SingleFluidThermo:
        assert (
            self.initial_state is not None
        ), "Initial state must be set before calling _get_new_state."
        isentropic_state = IsentropicPressureChange(
            self.final_pressure__pa, self.initial_state
        ).run()
        isentropic_enthalpy__kj_kg = isentropic_state.thermodynamic_state[
            ThermoConstants.ENTHALPY
        ][ThermoConstants.VALUE]
        actual_enthalpy__kj_kg = self._compute_actual_enthalpy(
            isentropic_enthalpy__kj_kg
        )
        return self.fluid_property_class(
            pressure__pa=self.final_pressure__pa, enthalpy__kj_kg=actual_enthalpy__kj_kg
        )

    def _compute_transition(
        self,
        initial_pressure__pa: float,
        final_pressure__pa: float,
        initial_enthalpy__kj_kg: float,
        final_enthalpy__kj_kg: float,
        n_points: int = 100,
    ) -> list:
        """
        Computes the thermodynamic state along the transition curve undergone by the
        fluid during the process. The transition curve is stored in the
        transition_states.

        Parameters
        ----------
        initial_pressure__pa : float
            The initial pressure of the fluid in Pa.
        final_pressure__pa : float
            The final pressure of the fluid in Pa.
        initial_enthalpy__kj_kg : float
            The initial enthalpy of the fluid in kJ/kg.
        final_enthalpy__kj_kg : float
            The final enthalpy of the fluid in kJ/kg.

        Returns
        -------
        list[SingleFluidThermo]
            The thermodynamic states along the transition curve.
        """
        pressure_points__pa = np.linspace(
            initial_pressure__pa, final_pressure__pa, n_points
        )
        isentropic_enthalpy_points__kj_kg = [
            IsentropicPressureChange(pressure_point__pa, self.initial_state)
            .run()
            .thermodynamic_state[ThermoConstants.ENTHALPY][ThermoConstants.VALUE]
            for pressure_point__pa in pressure_points__pa
        ]
        actual_enthalpy_points__kj_kg = [
            self._compute_actual_enthalpy(isentropic_enthalpy_point__kj_kg)
            for isentropic_enthalpy_point__kj_kg in isentropic_enthalpy_points__kj_kg
        ]
        return [
            self.fluid_property_class(
                pressure__pa=pressure_point__pa, enthalpy__kj_kg=enthalpy_point__kj_kg
            )
            for pressure_point__pa, enthalpy_point__kj_kg in zip(
                pressure_points__pa, actual_enthalpy_points__kj_kg
            )
        ]
