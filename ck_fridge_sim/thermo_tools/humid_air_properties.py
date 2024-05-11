from collections.abc import Iterable
from typing import Optional, Union, overload

import numpy as np
from CoolProp import CoolProp as CP

from ck_fridge_sim.thermo_tools.unit_conversions import UnitConversions


class HumidAirProperties:
    """
    This class calculates properties of humid air using the CoolProp library.
    See documentation here: http://www.coolprop.org/fluid_properties/HumidAir.html
    """

    PRESSURE = "P"
    WET_BULB_TEMPERATURE = "B"
    DRY_BULB_TEMPERATURE = TEMPERATURE = "T"
    DEW_POINT_TEMPERATURE = "D"
    RELATIVE_HUMIDITY = "R"
    HUMIDITY_MASS_RATIO = "W"
    SPECIFIC_HEAT = "C"
    ENTHALPY = "H"
    VOLUME = "Vha"
    VOLUME_DRY_AIR = "Vda"

    def __init__(self, ambient_pressure__pa: Optional[float] = 101325):
        """
        Parameters
        ----------
        ambient_pressure__pa : float, optional
            Ambient pressure in Pa, by default 101325
        """
        self._ambient_pressure__pa = ambient_pressure__pa

    @staticmethod
    def _convert_percent_list_to_fraction(
        relative_humidity__perc: Union[float, Iterable],
    ):
        """
        Convert relative humidity percentage to fraction

        Parameters
        ----------
        relative_humidity__perc : Union[float, Iterable]
            Relative humidity in %

        Returns
        -------
        Iterable
            Relative humidity in fraction
        """
        if isinstance(relative_humidity__perc, Iterable):
            relative_humidity__fraction = np.array(relative_humidity__perc) / 100
        else:
            relative_humidity__fraction = relative_humidity__perc / 100

        return relative_humidity__fraction

    def _validate_input(
        self,
        temp__c: Union[float, Iterable],
        relative_humidity__perc: Union[float, Iterable],
    ) -> None:
        """
        Validate inputs are the same length, and get the input type

        Parameters
        ----------
        temp__c : Union[float, Iterable]
            Temperature in Celsius
        relative_humidity__perc : Union[float, Iterable]
            Relative humidity in %
        """
        if isinstance(temp__c, (list, np.ndarray)) and isinstance(
            relative_humidity__perc, (list, np.ndarray)
        ):
            assert (
                len(temp__c) == len(relative_humidity__perc)
            ), f"Temperature and relative humidity must be the same length, got {len(temp__c)} and {len(relative_humidity__perc)}"

    @overload
    def get_wet_bulb_temp(
        self, temp__c: float, relative_humidity__perc: float
    ) -> float: ...

    @overload
    def get_wet_bulb_temp(
        self, temp__c: Iterable, relative_humidity__perc: Iterable
    ) -> list[float]: ...

    @overload
    def get_wet_bulb_temp(
        self, temp__c: float, relative_humidity__perc: Iterable
    ) -> list[float]: ...

    @overload
    def get_wet_bulb_temp(
        self, temp__c: Iterable, relative_humidity__perc: float
    ) -> list[float]: ...

    def get_wet_bulb_temp(
        self,
        temp__c: Union[float, Iterable],
        relative_humidity__perc: Union[float, Iterable],
    ) -> Union[float, list[float]]:
        """
        Calculate wet bulb temperature in Celsius

        Parameters
        ----------
        temp__c : Union[float, Iterable]
            Temperature in Celsius
        relative_humidity__perc : Union[float, Iterable]
            Relative humidity in %

        Returns
        -------
        Union[float, List[float]]
            Wet bulb temperature in Celsius
        """
        self._validate_input(temp__c, relative_humidity__perc)
        temp__k = UnitConversions.celsius_to_kelvin(temp__c)
        relative_humidity__fraction = self._convert_percent_list_to_fraction(
            relative_humidity__perc
        )
        temp_wb__k = CP.HAPropsSI(
            self.WET_BULB_TEMPERATURE,
            self.TEMPERATURE,
            temp__k,
            self.PRESSURE,
            self._ambient_pressure__pa,
            self.RELATIVE_HUMIDITY,
            relative_humidity__fraction,
        )
        return UnitConversions.kelvin_to_celsius(temp_wb__k)

    @overload
    def get_dew_point_temp(
        self, temp__c: float, relative_humidity__perc: float
    ) -> float: ...

    @overload
    def get_dew_point_temp(
        self, temp__c: Iterable, relative_humidity__perc: Iterable
    ) -> list[float]: ...

    @overload
    def get_dew_point_temp(
        self, temp__c: float, relative_humidity__perc: Iterable
    ) -> list[float]: ...

    @overload
    def get_dew_point_temp(
        self, temp__c: Iterable, relative_humidity__perc: float
    ) -> list[float]: ...

    def get_dew_point_temp(
        self,
        temp__c: Union[float, Iterable],
        relative_humidity__perc: Union[float, Iterable],
    ) -> Union[float, list[float]]:
        """
        Calculate dew point temperature in Celsius

        Parameters
        ----------
        temp__c : Union[float, Iterable]
            Temperature in Celsius
        relative_humidity__perc : Union[float, Iterable]
            Relative humidity in %

        Returns
        -------
        Union[float, List[float]]
            Dew point temperature in Celsius
        """
        self._validate_input(temp__c, relative_humidity__perc)
        temp__k = UnitConversions.celsius_to_kelvin(temp__c)
        relative_humidity__fraction = self._convert_percent_list_to_fraction(
            relative_humidity__perc
        )
        temp_dp__k = CP.HAPropsSI(
            self.DEW_POINT_TEMPERATURE,
            self.TEMPERATURE,
            temp__k,
            self.PRESSURE,
            self._ambient_pressure__pa,
            self.RELATIVE_HUMIDITY,
            relative_humidity__fraction,
        )
        return UnitConversions.kelvin_to_celsius(temp_dp__k)

    @overload
    def get_humidity_mass_ratio(
        self, temp__c: float, relative_humidity__perc: float
    ) -> float: ...

    @overload
    def get_humidity_mass_ratio(
        self, temp__c: Iterable, relative_humidity__perc: Iterable
    ) -> list[float]: ...

    @overload
    def get_humidity_mass_ratio(
        self, temp__c: float, relative_humidity__perc: Iterable
    ) -> list[float]: ...

    @overload
    def get_humidity_mass_ratio(
        self, temp__c: Iterable, relative_humidity__perc: float
    ) -> list[float]: ...

    def get_humidity_mass_ratio(
        self,
        temp__c: Union[float, Iterable],
        relative_humidity__perc: Union[float, Iterable],
    ) -> Union[float, list[float]]:
        """
        Calculate humidity mass ratio

        Parameters
        ----------
        temp__c : Union[float, Iterable]
            Temperature in Celsius
        relative_humidity__perc : Union[float, Iterable]
            Relative humidity in %

        Returns
        -------
        Union[float, List[float]]
            Humidity mass ratio in kg water/kg dry air
        """
        self._validate_input(temp__c, relative_humidity__perc)
        temp__k = UnitConversions.celsius_to_kelvin(temp__c)
        relative_humidity__fraction = self._convert_percent_list_to_fraction(
            relative_humidity__perc
        )
        output_value = CP.HAPropsSI(
            self.HUMIDITY_MASS_RATIO,
            self.TEMPERATURE,
            temp__k,
            self.PRESSURE,
            self._ambient_pressure__pa,
            self.RELATIVE_HUMIDITY,
            relative_humidity__fraction,
        )
        if isinstance(output_value, Iterable):
            return np.array(output_value, dtype=float).tolist()

        return float(output_value)

    @overload
    def get_heat_capacity(
        self, temp__c: float, relative_humidity__perc: float
    ) -> float: ...

    @overload
    def get_heat_capacity(
        self, temp__c: Iterable, relative_humidity__perc: Iterable
    ) -> list[float]: ...

    @overload
    def get_heat_capacity(
        self, temp__c: float, relative_humidity__perc: Iterable
    ) -> list[float]: ...

    @overload
    def get_heat_capacity(
        self, temp__c: Iterable, relative_humidity__perc: float
    ) -> list[float]: ...

    def get_heat_capacity(
        self,
        temp__c: Union[float, Iterable],
        relative_humidity__perc: Union[float, Iterable],
    ) -> Union[float, list[float]]:
        """
        Calculate heat capacity of air

        Parameters
        ----------
        temp__c : Union[float, Iterable]
            Temperature in Celsius
        relative_humidity__perc : Union[float, Iterable]
            Relative humidity in %

        Returns
        -------
        Union[float, List[float]]
            Heat capacity [kJ/kg K]
        """
        self._validate_input(temp__c, relative_humidity__perc)
        temp__k = UnitConversions.celsius_to_kelvin(temp__c)
        relative_humidity__fraction = self._convert_percent_list_to_fraction(
            relative_humidity__perc
        )
        output_value = (
            CP.HAPropsSI(
                self.SPECIFIC_HEAT,
                self.TEMPERATURE,
                temp__k,
                self.PRESSURE,
                self._ambient_pressure__pa,
                self.RELATIVE_HUMIDITY,
                relative_humidity__fraction,
            )
            / 1000
        )
        if isinstance(output_value, Iterable):
            return np.array(output_value, dtype=float).tolist()

        return float(output_value)

    @overload
    def get_enthalpy(self, temp__c: float, relative_humidity__perc: float) -> float: ...

    @overload
    def get_enthalpy(
        self, temp__c: Iterable, relative_humidity__perc: Iterable
    ) -> list[float]: ...

    @overload
    def get_enthalpy(
        self, temp__c: float, relative_humidity__perc: Iterable
    ) -> list[float]: ...

    @overload
    def get_enthalpy(
        self, temp__c: Iterable, relative_humidity__perc: float
    ) -> list[float]: ...

    def get_enthalpy(
        self,
        temp__c: Union[float, Iterable],
        relative_humidity__perc: Union[float, Iterable],
    ) -> Union[float, list[float]]:
        """
        Calculate enthalpy (total heat content) of humid air

        Parameters
        ----------
        temp__c : Union[float, Iterable]
            Temperature in Celsius
        relative_humidity__perc : Union[float, Iterable]
            Relative humidity in %

        Returns
        -------
        Union[float, List[float]]
            Enthalpy [kJ/kg]
        """
        self._validate_input(temp__c, relative_humidity__perc)
        temp__k = UnitConversions.celsius_to_kelvin(temp__c)
        relative_humidity__fraction = self._convert_percent_list_to_fraction(
            relative_humidity__perc
        )
        output_value = (
            CP.HAPropsSI(
                self.ENTHALPY,
                self.TEMPERATURE,
                temp__k,
                self.PRESSURE,
                self._ambient_pressure__pa,
                self.RELATIVE_HUMIDITY,
                relative_humidity__fraction,
            )
            / 1000
        )
        if isinstance(output_value, Iterable):
            return np.array(output_value, dtype=float).tolist()

        return float(output_value)

    @overload
    def get_density(self, temp__c: float, relative_humidity__perc: float) -> float: ...

    @overload
    def get_density(
        self, temp__c: Iterable, relative_humidity__perc: Iterable
    ) -> list[float]: ...

    @overload
    def get_density(
        self, temp__c: float, relative_humidity__perc: Iterable
    ) -> list[float]: ...

    @overload
    def get_density(
        self, temp__c: Iterable, relative_humidity__perc: float
    ) -> list[float]: ...

    def get_density(
        self,
        temp__c: Union[float, Iterable],
        relative_humidity__perc: Union[float, Iterable],
    ) -> Union[float, list[float]]:
        """
        Calculate density of humid air

        Parameters
        ----------
        temp__c : Union[float, Iterable]
            Temperature in Celsius
        relative_humidity__perc : Union[float, Iterable]
            Relative humidity in %

        Returns
        -------
        Union[float, List[float]]
            Density [kg humid air / m3 mixture]
        """
        self._validate_input(temp__c, relative_humidity__perc)
        temp__k = UnitConversions.celsius_to_kelvin(temp__c)
        relative_humidity__fraction = self._convert_percent_list_to_fraction(
            relative_humidity__perc
        )
        output_value = 1 / CP.HAPropsSI(
            self.VOLUME,
            self.TEMPERATURE,
            temp__k,
            self.PRESSURE,
            self._ambient_pressure__pa,
            self.RELATIVE_HUMIDITY,
            relative_humidity__fraction,
        )
        if isinstance(output_value, Iterable):
            return np.array(output_value, dtype=float).tolist()

        return float(output_value)

    @overload
    def get_density_dry_air(
        self, temp__c: float, relative_humidity__perc: float
    ) -> float: ...

    @overload
    def get_density_dry_air(
        self, temp__c: Iterable, relative_humidity__perc: Iterable
    ) -> list[float]: ...

    @overload
    def get_density_dry_air(
        self, temp__c: float, relative_humidity__perc: Iterable
    ) -> list[float]: ...

    @overload
    def get_density_dry_air(
        self, temp__c: Iterable, relative_humidity__perc: float
    ) -> list[float]: ...

    def get_density_dry_air(
        self,
        temp__c: Union[float, Iterable],
        relative_humidity__perc: Union[float, Iterable],
    ) -> Union[float, list[float]]:
        """
        Calculate density of dry air

        Parameters
        ----------
        temp__c : Union[float, Iterable]
            Temperature in Celsius
        relative_humidity__perc : Union[float, Iterable]
            Relative humidity in %

        Returns
        -------
        Union[float, List[float]]
            Density [kg dry air / m3 mixture]
        """
        self._validate_input(temp__c, relative_humidity__perc)
        temp__k = UnitConversions.celsius_to_kelvin(temp__c)
        relative_humidity__fraction = self._convert_percent_list_to_fraction(
            relative_humidity__perc
        )
        output_value = 1 / CP.HAPropsSI(
            self.VOLUME_DRY_AIR,
            self.TEMPERATURE,
            temp__k,
            self.PRESSURE,
            self._ambient_pressure__pa,
            self.RELATIVE_HUMIDITY,
            relative_humidity__fraction,
        )
        if isinstance(output_value, Iterable):
            return np.array(output_value, dtype=float).tolist()

        return float(output_value)
