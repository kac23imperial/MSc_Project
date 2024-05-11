import numpy as np

from ck_fridge_sim.thermo_tools.equations_of_state.cp_corrected_ammonia import (
    CPCorrectedAmmonia,
)
from ck_fridge_sim.thermo_tools.single_fluid_thermo import SingleFluidThermo


class AmmoniaThermo(SingleFluidThermo):
    FLUID = "Ammonia"
    EOS = CPCorrectedAmmonia

    CRIT_TEMPERATURE__K = 405.56
    CRIT_PRESSURE__PA = 11.3634e6
    TP_PARAMS = {
        "A": 1.35319274e-03,
        "B": -7.00055999e00,
        "C": 2.00160625e00,
        "D": -3.89816104e00,
    }
    PT_PARAMS = {
        "A": 9.94166099e-01,
        "B": 1.37260964e-01,
        "C": 1.43034141e-02,
        "D": 6.98867877e-04,
    }

    @staticmethod
    def _theta(temp__k: float, crit_temp__k: float) -> float:
        return 1 - temp__k / crit_temp__k

    @staticmethod
    def _exp_sum(scaled_value: float, A: float, B: float, C: float, D: float) -> float:
        return A + B * scaled_value + C * scaled_value**2 + D * scaled_value**3

    @staticmethod
    def saturated_pressure__pa(temp__k: float) -> float:
        """
        Convert Kelvin to Pascal on ammonia phase line. Fit from NIST data.

        Parameters
        ----------
        temp__k : float, Iterable
            Temperature in Kelvin

        Returns
        -------
        float
            Pressure in Pascal
        """
        scaled_temp = AmmoniaThermo._theta(temp__k, AmmoniaThermo.CRIT_TEMPERATURE__K)
        return AmmoniaThermo.CRIT_PRESSURE__PA * np.exp(
            AmmoniaThermo._exp_sum(scaled_temp, **AmmoniaThermo.TP_PARAMS)
            * AmmoniaThermo.CRIT_TEMPERATURE__K
            / temp__k
        )

    @staticmethod
    def _log_pressure_frac(pressure__pa: float, crit_pressure__pa: float) -> float:
        return np.log(pressure__pa / crit_pressure__pa)

    @staticmethod
    def saturated_temperature__k(pressure__pa: float) -> float:
        """
        Convert Pascal to Kelvin on ammonia phase line. Fit from NIST data.

        Parameters
        ----------
        pressure__pa : float, Iterable
            Pressure in Pascal

        Returns
        -------
        float
            Temperature in Kelvin
        """
        scaled_pressure = AmmoniaThermo._log_pressure_frac(
            pressure__pa, AmmoniaThermo.CRIT_PRESSURE__PA
        )
        return AmmoniaThermo.CRIT_TEMPERATURE__K * AmmoniaThermo._exp_sum(
            scaled_pressure, **AmmoniaThermo.PT_PARAMS
        )
