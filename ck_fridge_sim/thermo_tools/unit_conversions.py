from collections.abc import Iterable
from typing import overload

import numpy as np


class UnitConversions:
    """
    A class for performing unit conversions.

    Usage:
    ------
    To use the `UnitConversions` class, simply import it into your code and call any of the available
    conversion methods:

    >>> from ck_joule_tools.thermo_tools.unit_conversions import UnitConversions
    >>> temp_f = UnitConversions.celsius_to_fahrenheit(25)
    >>> print(temp_f)
    77.0

    >>> temp_c = UnitConversions.fahrenheit_to_celsius(77)
    >>> print(temp_c)
    25.0

    >>> pressure_psig = UnitConversions.pascal_to_psig(100000)
    >>> print(pressure_psig)
    -0.19622622031412895

    >>> pressure_pa = UnitConversions.psig_to_pascal(50)
    >>> print(pressure_pa)
    446090.796663

    Note that both arrays and scalars are supported as inputs to the conversion methods.
    For more information on each method, see the docstrings for each individual method.
    """

    @overload
    @staticmethod
    def celsius_to_fahrenheit(temp__c: float) -> float:
        ...

    @overload
    @staticmethod
    def celsius_to_fahrenheit(temp__c: Iterable) -> Iterable[float]:
        ...

    @staticmethod
    def celsius_to_fahrenheit(temp__c: float | Iterable) -> float | Iterable[float]:
        """
        Convert Celsius to Fahrenheit

        Parameters
        ----------
        temp__c : float, Iterable
            Temperature in Celsius

        Returns
        -------
        float | Iterable[float]
            Temperature in Fahrenheit
        """
        output_value = np.array(temp__c) * 9 / 5 + 32
        if output_value.size == 1:
            return float(output_value)  # type: ignore

        return np.array(output_value, dtype=float)

    @overload
    @staticmethod
    def fahrenheit_to_celsius(temp__f: float) -> float:
        ...

    @overload
    @staticmethod
    def fahrenheit_to_celsius(temp__f: Iterable) -> Iterable[float]:
        ...

    @staticmethod
    def fahrenheit_to_celsius(temp__f: float | Iterable) -> float | Iterable[float]:
        """
        Convert Fahrenheit to Celsius

        Parameters
        ----------
        temp__f : float, Iterable
            Temperature in Fahrenheit

        Returns
        -------
        float | Iterable[float]
            Temperature in Celsius
        """
        output_value = (np.array(temp__f) - 32) * 5 / 9
        if output_value.size == 1:
            return float(output_value)  # type: ignore

        return np.array(output_value, dtype=float)

    @overload
    @staticmethod
    def celsius_to_kelvin(temp__c: float) -> float:
        ...

    @overload
    @staticmethod
    def celsius_to_kelvin(temp__c: Iterable) -> Iterable[float]:
        ...

    @staticmethod
    def celsius_to_kelvin(temp__c: float | Iterable) -> float | Iterable[float]:
        """
        Convert Celsius to Kelvin

        Parameters
        ----------
        temp__c : float, Iterable
            Temperature in Celsius

        Returns
        -------
        float | Iterable[float]
            Temperature in Kelvin
        """
        output_value = np.array(temp__c) + 273.15
        if output_value.size == 1:
            return float(output_value)  # type: ignore

        return np.array(output_value, dtype=float)

    @overload
    @staticmethod
    def kelvin_to_celsius(temp__k: float) -> float:
        ...

    @overload
    @staticmethod
    def kelvin_to_celsius(temp__k: Iterable) -> Iterable[float]:
        ...

    @staticmethod
    def kelvin_to_celsius(temp__k: float | Iterable) -> float | Iterable[float]:
        """
        Convert Kelvin to Celsius

        Parameters
        ----------
        temp__k : float, Iterable
            Temperature in Kelvin

        Returns
        -------
        float | Iterable[float]
            Temperature in Celsius
        """
        output_value = np.array(temp__k) - 273.15
        if output_value.size == 1:
            return float(output_value)  # type: ignore

        return np.array(output_value, dtype=float)

    @overload
    @staticmethod
    def fahrenheit_to_kelvin(temp__f: float) -> float:
        ...

    @overload
    @staticmethod
    def fahrenheit_to_kelvin(temp__f: Iterable) -> Iterable[float]:
        ...

    @staticmethod
    def fahrenheit_to_kelvin(temp__f: float | Iterable) -> float | Iterable[float]:
        """
        Convert Fahrenheit to Kelvin

        Parameters
        ----------
        temp__f : float, Iterable
            Temperature in Fahrenheit

        Returns
        -------
        float | Iterable[float]
            Temperature in Kelvin
        """
        return UnitConversions.celsius_to_kelvin(
            UnitConversions.fahrenheit_to_celsius(temp__f)
        )

    @overload
    @staticmethod
    def kelvin_to_fahrenheit(temp__k: float) -> float:
        ...

    @overload
    @staticmethod
    def kelvin_to_fahrenheit(temp__k: Iterable) -> Iterable[float]:
        ...

    @staticmethod
    def kelvin_to_fahrenheit(temp__k: float | Iterable) -> float | Iterable[float]:
        """
        Convert Kelvin to Fahrenheit

        Parameters
        ----------
        temp__k : float, Iterable
            Temperature in Kelvin

        Returns
        -------
        float | Iterable[float]
            Temperature in Fahrenheit
        """
        return UnitConversions.celsius_to_fahrenheit(
            UnitConversions.kelvin_to_celsius(temp__k)
        )

    @overload
    @staticmethod
    def rankine_to_kelvin(temp__r: float) -> float:
        ...

    @overload
    @staticmethod
    def rankine_to_kelvin(temp__r: Iterable) -> Iterable[float]:
        ...

    @staticmethod
    def rankine_to_kelvin(temp__r: float | Iterable) -> float | Iterable[float]:
        """
        Convert Rankine to Kelvin

        Parameters
        ----------
        temp__r : float, Iterable
            Temperature in Rankine

        Returns
        -------
        float | Iterable[float]
            Temperature in Kelvin
        """
        output_value = np.array(temp__r) * 5 / 9
        if output_value.size == 1:
            return float(output_value)  # type: ignore

        return np.array(output_value, dtype=float)

    @overload
    @staticmethod
    def kelvin_to_rankine(temp__k: float) -> float:
        ...

    @overload
    @staticmethod
    def kelvin_to_rankine(temp__k: Iterable) -> Iterable[float]:
        ...

    @staticmethod
    def kelvin_to_rankine(temp__k: float | Iterable) -> float | Iterable[float]:
        """
        Convert Kelvin to Rankine

        Parameters
        ----------
        temp__k : float, Iterable
            Temperature in Kelvin

        Returns
        -------
        float | Iterable[float]
            Temperature in Rankine
        """
        output_value = np.array(temp__k) * 9 / 5
        if output_value.size == 1:
            return float(output_value)  # type: ignore

        return np.array(output_value, dtype=float)

    @overload
    @staticmethod
    def pascal_to_psig(pressure__pa: float) -> float:
        ...

    @overload
    @staticmethod
    def pascal_to_psig(pressure__pa: Iterable) -> Iterable[float]:
        ...

    @staticmethod
    def pascal_to_psig(pressure__pa: float | Iterable) -> float | Iterable[float]:
        """
        Convert Pascal to psig

        Parameters
        ----------
        pressure__pa : float, Iterable
            Pressure in Pascal

        Returns
        -------
        float | Iterable[float]
            Pressure in psig
        """
        output_value = np.array(pressure__pa) / 6894.75729 - 14.7
        if output_value.size == 1:
            return float(output_value)  # type: ignore

        return np.array(output_value, dtype=float)

    @overload
    @staticmethod
    def psig_to_pascal(pressure__psig: float) -> float:
        ...

    @overload
    @staticmethod
    def psig_to_pascal(pressure__psig: Iterable) -> Iterable[float]:
        ...

    @staticmethod
    def psig_to_pascal(pressure__psig: float | Iterable) -> float | Iterable[float]:
        """
        Convert psig to Pascal

        Parameters
        ----------
        pressure__psig : float, Iterable
            Pressure in psig

        Returns
        -------
        float | Iterable[float]
            Pressure in Pascal
        """
        output_value = (np.array(pressure__psig) + 14.7) * 6894.75729
        if output_value.size == 1:
            return float(output_value)  # type: ignore

        return np.array(output_value, dtype=float)

    @overload
    @staticmethod
    def lbs_per_min_to_kg_per_s(lbs_per_min: float) -> float:
        ...

    @overload
    @staticmethod
    def lbs_per_min_to_kg_per_s(lbs_per_min: Iterable) -> Iterable[float]:
        ...

    @staticmethod
    def lbs_per_min_to_kg_per_s(
        lbs_per_min: float | Iterable,
    ) -> float | Iterable[float]:
        """
        Convert lbs/min to kg/s

        Parameters
        ----------
        lbs_per_min : float, Iterable
            Mass flow rate in lbs/min

        Returns
        -------
        float | Iterable[float]
            Mass flow rate in kg/s
        """
        output_value = np.array(lbs_per_min) / 132.277
        if output_value.size == 1:
            return float(output_value)  # type: ignore

        return np.array(output_value, dtype=float)

    @overload
    @staticmethod
    def kg_per_s_to_lbs_per_min(kg_per_s: float) -> float:
        ...

    @overload
    @staticmethod
    def kg_per_s_to_lbs_per_min(kg_per_s: Iterable) -> Iterable[float]:
        ...

    @staticmethod
    def kg_per_s_to_lbs_per_min(
        kg_per_s: float | Iterable,
    ) -> float | Iterable[float]:
        """
        Convert kg/s to lbs/min

        Parameters
        ----------
        kg_per_s : float, Iterable
            Mass flow rate in kg/s

        Returns
        -------
        float | Iterable[float]
            Mass flow rate in lbs/min
        """
        output_value = np.array(kg_per_s) * 132.277
        if output_value.size == 1:
            return float(output_value)  # type: ignore

        return np.array(output_value, dtype=float)

    @overload
    @staticmethod
    def hp_to_kw(hp: float) -> float:
        ...

    @overload
    @staticmethod
    def hp_to_kw(hp: Iterable) -> Iterable[float]:
        ...

    @staticmethod
    def hp_to_kw(hp: float | Iterable) -> float | Iterable[float]:
        """
        Convert horsepower to kilowatts

        Parameters
        ----------
        hp : float, Iterable
            Power in horsepower

        Returns
        -------
        float | Iterable[float]
            Power in kilowatts
        """
        output_value = np.array(hp) * 0.7457
        if output_value.size == 1:
            return float(output_value)  # type: ignore

        return np.array(output_value, dtype=float)
