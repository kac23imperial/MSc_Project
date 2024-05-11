from typing import Optional

import numpy as np

from ck_fridge_sim.thermo_tools.single_fluid_thermo import SingleFluidThermo
from ck_fridge_sim.thermo_tools.thermo_constants import ThermoConstants
from ck_fridge_sim.thermo_tools.thermo_processes.thermo_process_base import (
    ThermoProcessBase,
)


class IsentropicPressureChange(ThermoProcessBase):
    """
    This is a model of an isentropic (reversible) pressure change process.
    """

    def __init__(
        self,
        final_pressure__pa: float,
        initial_state: Optional[SingleFluidThermo] = None,
    ):
        """
        Parameters
        ----------
        final_pressure__pa : float
            The final pressure of the fluid in Pa.
        initial_state : SingleFluidThermo
            A SingleFluidThermo object (or subclass) that represents the initial
            thermodynamic state of the fluid.
        """
        super().__init__(initial_state=initial_state)
        self.final_pressure__pa = final_pressure__pa

    def _get_new_state(self) -> SingleFluidThermo:
        assert (
            self.initial_state is not None
        ), "Initial state must be set before running the process."
        entropy__kj_kgk = self.initial_state.thermodynamic_state[
            ThermoConstants.ENTROPY
        ][ThermoConstants.VALUE]
        return self.fluid_property_class(
            pressure__pa=self.final_pressure__pa, entropy__kj_kgk=entropy__kj_kgk
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
        List[SingleFluidThermo]
            The thermodynamic states along the transition curve.
        """
        assert (
            self.initial_state is not None
        ), "Initial state must be set before running the process."
        pressure_points__pa = np.linspace(
            initial_pressure__pa, final_pressure__pa, n_points
        )
        entropy__kj_kgk = self.initial_state.thermodynamic_state[
            ThermoConstants.ENTROPY
        ][ThermoConstants.VALUE]
        return [
            self.fluid_property_class(
                pressure__pa=pressure_point__pa, entropy__kj_kgk=entropy__kj_kgk
            )
            for pressure_point__pa in pressure_points__pa
        ]
