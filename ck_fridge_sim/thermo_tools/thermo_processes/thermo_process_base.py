from abc import ABC, abstractmethod
from typing import Optional

from ck_fridge_sim.thermo_tools.single_fluid_thermo import SingleFluidThermo
from ck_fridge_sim.thermo_tools.thermo_constants import ThermoConstants


class ThermoProcessBase(ABC):
    """
    Base class for all thermodynamic processes. This class is abstract, so it
    must be subclassed to be used. It defines the interface for all subclasses.
    Typical usage is like so:
    initial_state = SingleFluidThermo(pressure__pa=100000, temp__c=25)
    >>> compressor = IrreversibleCompression(
            initial_state=initial_state, final_pressure__pa=200000, efficiency=0.8)
    >>> new_state = compressor.run()
    >>> new_state.thermodyanmic_state
    >>> {
            'pressure': {
                'value': 200000.0,
                'units': 'Pa'
            },
            'temperature': {
                'value': 25.0,
                'units': 'C'
            },
            'quality': {
                'value': 0.0,
                'units': 'fraction'
            },
            'enthalpy': {
                'value': 0.0,
                'units': 'kJ/kg'
            },
            'entropy': {
                'value': 0.0,
                'units': 'kJ/kg-K'
            },
            'density': {
                'value': 0.0,
                'units': 'kg/m3'
            }
        }
    """

    def __init__(self, initial_state: Optional[SingleFluidThermo] = None):
        """
        Instantiates a new ThermoProcessBase object.

        Parameters
        ----------
        initial_state : SingleFluidThermo
            A SingleFluidThermo object (or subclass) that represents the initial
            thermodynamic state of the fluid.
        """
        self.initial_state = initial_state
        self._new_state = None
        self.transition_states = None

    def set_initial_state(self, initial_state: SingleFluidThermo):
        """
        Sets the initial state of the fluid.

        Parameters
        ----------
        initial_state : SingleFluidThermo
            A SingleFluidThermo object (or subclass) that represents the initial
            thermodynamic state of the fluid.
        """
        self.initial_state = initial_state

    @abstractmethod
    def _get_new_state(self) -> SingleFluidThermo:
        """
        This method should be overridden by subclasses to return a new thermodynamic
        state.

        Returns
        -------
        SingleFluidThermo
            A SingleFluidThermo object (or subclass) that represents the final
            thermodynamic state of the fluid.
        """
        return

    def run(self) -> SingleFluidThermo:
        """
        Runs the process and returns the final thermodynamic state of the fluid.

        Returns
        -------
        SingleFluidThermo
            A SingleFluidThermo object (or subclass) that represents the final
            thermodynamic state of the fluid.
        """
        self._new_state = self._get_new_state()
        return self._new_state

    @abstractmethod
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
        return

    def get_transition_curve(self, n_points: int = 100):
        """
        Computes the thermodynamic state along the transition curve undergone by the
        fluid during the process. The transition curve is stored in the
        transition_states.

        Parameters
        ----------
        n_points : int, optional
            The number of volume points used to estimate the integral, by default 100

        Returns
        -------
        List[SingleFluidThermo]
            The thermodynamic states along the transition curve.

        Raises
        ------
        Exception
            If the process has not been run yet.
        """
        assert (
            self.initial_state is not None
        ), "Initial state must be set before running process."
        if not self._new_state:
            raise Exception("Process has not been run yet.")

        initial_enthalpy__kj_kg = self.initial_state.thermodynamic_state[
            ThermoConstants.ENTHALPY
        ][ThermoConstants.VALUE]
        final_enthalpy__kj_kg = self._new_state.thermodynamic_state[
            ThermoConstants.ENTHALPY
        ][ThermoConstants.VALUE]
        initial_pressure__pa = self.initial_state.thermodynamic_state[
            ThermoConstants.PRESSURE
        ][ThermoConstants.VALUE]
        final_pressure__pa = self._new_state.thermodynamic_state[
            ThermoConstants.PRESSURE
        ][ThermoConstants.VALUE]

        transition_states = self._compute_transition(
            initial_pressure__pa=initial_pressure__pa,
            final_pressure__pa=final_pressure__pa,
            initial_enthalpy__kj_kg=initial_enthalpy__kj_kg,
            final_enthalpy__kj_kg=final_enthalpy__kj_kg,
            n_points=n_points,
        )
        self.transition_states = transition_states
        return self.transition_states

    @staticmethod
    def get_property_along_transition_curve(
        property_name: str, transition_states: list[SingleFluidThermo]
    ) -> list:
        """
        Returns the values of a property along the transition curve.

        Parameters
        ----------
        property_name : str
            The name of the property to return.
        transition_states : List[SingleFluidThermo]
            The thermodynamic states along the transition curve.

        Returns
        -------
        list[float]
            The values of the property along the transition curve.
        """
        return [
            state.thermodynamic_state[property_name][ThermoConstants.VALUE]
            for state in transition_states
        ]

    @property
    def fluid_property_class(self):
        """
        Returns
        -------
        SingleFluidThermo
            The class of the fluid property object used to compute the thermodynamic
            state.
        """
        assert (
            self.initial_state is not None
        ), "Initial state must be set before running the process."
        return self.initial_state.__class__

    @property
    def new_state(self):
        """
        Returns
        -------
        SingleFluidThermo
            The class of the fluid property object used to compute the thermodynamic
            state.

        Raises
        ------
        Exception
            If the process has not been run yet.
        """
        if self._new_state:
            return self._new_state
        else:
            raise AttributeError("Process has not been run yet.")
