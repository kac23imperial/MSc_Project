from CoolProp import CoolProp as CP

from ck_fridge_sim.thermo_tools.equations_of_state.equation_of_state_base import (
    EquationOfStateBase,
)
from ck_fridge_sim.thermo_tools.thermo_constants import ThermoConstants
from ck_fridge_sim.thermo_tools.unit_conversions import UnitConversions


class CoolPropRKS(EquationOfStateBase):
    """
    Read documentation here for CoolProp:
    http://www.coolprop.org/coolprop/HighLevelAPI.html#parameter-table
    CoolProp is based on the Redlick-Kwong-Stryjek (RKS) equation of state,
    and uses SI units (kg, m, s, K, Pa, J).
    """

    PRESSURE_KEY = "P"
    TEMPERATURE_KEY = "T"
    QUALITY_KEY = "Q"
    ENTHALPY_KEY = "H"
    ENTROPY_KEY = "S"
    DENSITY_KEY = "D"
    CRITICAL_PRESSURE_KEY = "PCRIT"
    MOLAR_MASS_KEY = "MOLARMASS"

    PROP_TO_KEY_DICT = {
        ThermoConstants.PRESSURE: PRESSURE_KEY,
        ThermoConstants.TEMPERATURE: TEMPERATURE_KEY,
        ThermoConstants.QUALITY: QUALITY_KEY,
        ThermoConstants.ENTHALPY: ENTHALPY_KEY,
        ThermoConstants.ENTROPY: ENTROPY_KEY,
        ThermoConstants.DENSITY: DENSITY_KEY,
    }

    def get_enthalpy_from_PT(self, pressure__pa: float, temp__c: float, fluid: str):
        """
        Returns the specific enthalpy given pressure and temperature.

        Parameters
        ----------
        pressure__pa : float
            Pressure in Pa
        temp__c : float
            Temperature in C
        fluid : str
            Name of the fluid

        Returns
        -------
        float
            Enthalpy in kJ/kg
        """
        temp__k = UnitConversions.celsius_to_kelvin(temp__c)
        return (
            CP.PropsSI(
                self.ENTHALPY_KEY,
                self.PRESSURE_KEY,
                pressure__pa,
                self.TEMPERATURE_KEY,
                temp__k,
                fluid,
            )
            / 1000
        )

    def get_enthalpy_from_PQ(self, pressure__pa: float, quality: float, fluid: str):
        """
        Returns the specific enthalpy given pressure and quality.

        Parameters
        ----------
        pressure__pa : float
            Pressure in Pa
        quality : float
            Quality
        fluid : str
            Name of the fluid

        Returns
        -------
        float
            Enthalpy in kJ/kg
        """
        return (
            CP.PropsSI(
                self.ENTHALPY_KEY,
                self.PRESSURE_KEY,
                pressure__pa,
                self.QUALITY_KEY,
                quality,
                fluid,
            )
            / 1000
        )

    def get_enthalpy_from_PS(
        self, pressure__pa: float, entropy__kj_kgk: float, fluid: str
    ):
        """
        Returns the specific enthalpy given pressure and entropy.

        Parameters
        ----------
        pressure__pa : float
            Pressure in Pa
        entropy__kj_kgk : float
            Entropy in kJ/kg-K
        fluid : str
            Name of the fluid

        Returns
        -------
        float
            Enthalpy in kJ/kg
        """
        return (
            CP.PropsSI(
                self.ENTHALPY_KEY,
                self.PRESSURE_KEY,
                pressure__pa,
                self.ENTROPY_KEY,
                entropy__kj_kgk * 1000,
                fluid,
            )
            / 1000
        )

    def get_enthalpy_from_TQ(self, temp__c: float, quality: float, fluid: str):
        """
        Returns the specific enthalpy given temperature and quality.

        Parameters
        ----------
        temp__c : float
            Temperature in C
        quality : float
            Quality
        fluid : str
            Name of the fluid

        Returns
        -------
        float
            Enthalpy in kJ/kg
        """
        temp__k = UnitConversions.celsius_to_kelvin(temp__c)
        return (
            CP.PropsSI(
                self.ENTHALPY_KEY,
                self.TEMPERATURE_KEY,
                temp__k,
                self.QUALITY_KEY,
                quality,
                fluid,
            )
            / 1000
        )

    def get_pressure_from_TQ(self, temp__c: float, quality: float, fluid: str):
        """
        Returns the pressure given temperature and enthalpy.

        Parameters
        ----------
        temp__c : float
            Temperature in C
        quality : float
            Vapor quality
        fluid : str
            Name of the fluid

        Returns
        -------
        float
            Pressure in kPa
        """
        temp__k = UnitConversions.celsius_to_kelvin(temp__c)
        return CP.PropsSI(
            self.PRESSURE_KEY,
            self.TEMPERATURE_KEY,
            temp__k,
            self.QUALITY_KEY,
            quality,
            fluid,
        )

    def get_enthalpy_from_TS(self, temp__c: float, entropy__kj_kgk: float, fluid: str):
        """
        Returns the specific enthalpy given temperature and entropy.

        Parameters
        ----------
        temp__c : float
            Temperature in C
        entropy__kj_kgk : float
            Entropy in kJ/kg-K
        fluid : str
            Name of the fluid

        Returns
        -------
        float
            Enthalpy in kJ/kg
        """
        temp__k = UnitConversions.celsius_to_kelvin(temp__c)
        return (
            CP.PropsSI(
                self.ENTHALPY_KEY,
                self.TEMPERATURE_KEY,
                temp__k,
                self.ENTROPY_KEY,
                entropy__kj_kgk * 1000,
                fluid,
            )
            / 1000
        )

    def get_pressure_from_HS(
        self, enthalpy__kj_kg: float, entropy__kj_kgk: float, fluid: str
    ):
        """
        Returns the pressure given enthalpy and entropy.

        Parameters
        ----------
        enthalpy__kj_kg : float
            Enthalpy in kJ/kg
        entropy__kj_kgk : float
            Entropy in kJ/kg-K
        fluid : str
            Name of the fluid

        Returns
        -------
        float
            Pressure in Pa
        """
        return CP.PropsSI(
            self.PRESSURE_KEY,
            self.ENTHALPY_KEY,
            enthalpy__kj_kg * 1000,
            self.ENTROPY_KEY,
            entropy__kj_kgk * 1000,
            fluid,
        )

    def get_enthalpy_from_PD(
        self, pressure__pa: float, density__kg_m3: float, fluid: str
    ):
        """
        Returns the specific enthalpy given pressure and density.

        Parameters
        ----------
        pressure__pa : float
            Pressure in Pa
        density__kg_m3 : float
            Density in kg/m3
        fluid : str
            Name of the fluid

        Returns
        -------
        float
            Enthalpy in kJ/kg
        """
        return (
            CP.PropsSI(
                self.ENTHALPY_KEY,
                self.PRESSURE_KEY,
                pressure__pa,
                self.DENSITY_KEY,
                density__kg_m3,
                fluid,
            )
            / 1000
        )

    def get_enthalpy_from_TD(self, temp__c: float, density__kg_m3: float, fluid: str):
        """
        Returns the specific enthalpy given temperature and density.

        Parameters
        ----------
        temp__c : float
            Temperature in C
        density__kg_m3 : float
            Density in kg/m3
        fluid : str
            Name of the fluid

        Returns
        -------
        float
            Enthalpy in kJ/kg
        """
        temp__k = UnitConversions.celsius_to_kelvin(temp__c)
        return (
            CP.PropsSI(
                self.ENTHALPY_KEY,
                self.TEMPERATURE_KEY,
                temp__k,
                self.DENSITY_KEY,
                density__kg_m3,
                fluid,
            )
            / 1000
        )

    def get_pressure_from_DH(
        self, density__kg_m3: float, enthalpy__kj_kg: float, fluid: str
    ):
        """
        Returns the pressure given density and enthalpy.

        Parameters
        ----------
        density__kg_m3 : float
            Density in kg/m3
        enthalpy__kj_kg : float
            Enthalpy in kJ/kg
        fluid : str
            Name of the fluid

        Returns
        -------
        float
            Pressure in kPa
        """
        return CP.PropsSI(
            self.PRESSURE_KEY,
            self.DENSITY_KEY,
            density__kg_m3,
            self.ENTHALPY_KEY,
            enthalpy__kj_kg * 1000,
            fluid,
        )

    def get_pressure_from_SD(
        self, entropy__kj_kgk: float, density__kg_m3: float, fluid: str
    ):
        """
        Returns the pressure given entropy and density.

        Parameters
        ----------
        entropy__kj_kgk : float
            Entropy in kJ/kg-K
        density__kg_m3 : float
            Density in kg/m3
        fluid : str
            Name of the fluid

        Returns
        -------
        float
            Pressure in kPa
        """
        return CP.PropsSI(
            self.PRESSURE_KEY,
            self.ENTROPY_KEY,
            entropy__kj_kgk * 1000,
            self.DENSITY_KEY,
            density__kg_m3,
            fluid,
        )

    def get_density_from_PQ(self, pressure__pa: float, quality: float, fluid: str):
        """
        Returns the density given pressure and quality.

        Parameters
        ----------
        pressure__pa : float
            Pressure in Pa
        quality : float
            Quality
        fluid : str
            Name of the fluid

        Returns
        -------
        float
            Density in kg/m3
        """
        return CP.PropsSI(
            self.DENSITY_KEY,
            self.PRESSURE_KEY,
            pressure__pa,
            self.QUALITY_KEY,
            quality,
            fluid,
        )

    def get_molar_mass(self, fluid: str) -> float:
        """
        Returns the molar mass of the fluid.

        Parameters
        ----------
        fluid : str
            Name of the fluid

        Returns
        -------
        float
            Molar mass in kg/mol
        """
        return CP.PropsSI(self.MOLAR_MASS_KEY, fluid)

    def get_property(
        self,
        property_name: str,
        pressure__pa: float,
        enthalpy__kj_kg: float,
        fluid: str,
    ):
        """
        Returns the property given pressure and enthalpy.

        Parameters
        ----------
        property_name : str
            Name of the property to be calculated
        pressure__pa : float
            Pressure in Pa
        enthalpy__kj_kg : float
            Enthalpy in kJ/kg
        fluid : str
            Name of the fluid

        Returns
        -------
        float
            Property in the units specified by the property name
        """
        property_value = CP.PropsSI(
            self.PROP_TO_KEY_DICT[property_name],
            self.PRESSURE_KEY,
            pressure__pa,
            self.ENTHALPY_KEY,
            enthalpy__kj_kg * 1000,
            fluid,
        )
        if (property_name == ThermoConstants.QUALITY) and (property_value == -1):
            if enthalpy__kj_kg <= self.get_enthalpy_from_PQ(pressure__pa, 0, fluid):
                property_value = 0.0
            else:
                property_value = 1.0
        elif property_name in [ThermoConstants.ENTROPY, ThermoConstants.ENTHALPY]:
            property_value /= 1000
        elif property_name == ThermoConstants.TEMPERATURE:
            property_value = UnitConversions.kelvin_to_celsius(property_value)

        return property_value

    def get_critical_pressure(self, fluid: str):
        return CP.PropsSI(self.CRITICAL_PRESSURE_KEY, fluid)
