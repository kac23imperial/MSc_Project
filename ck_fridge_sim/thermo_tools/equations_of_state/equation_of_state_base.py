from abc import ABC, abstractmethod
from typing import Union

from ck_fridge_sim.thermo_tools.thermo_constants import ThermoConstants


class EquationOfStateBase(ABC):
    """
    The purpose of this class is to provide a common interface for any
    equation of state. Rather than limiting to only using CoolProp, this
    will allow us to use any equation of state, but still have a common
    interface for computing the thermodynamic state.
    Note, the properties are given in the dictionary as follows. The units
    must be given in the units specified below (and set in the ThermConstants
    class).
    {
        "pressure": {
            "units": "pascals",
            "value": 101352.93216299999
        },
        "temperature": {
            "units": "celsius",
            "value": -31.666666666666668
        },
        "enthalpy": {
            "units": "kJ_kg",
            "value": 1567.8405433915957
        },
        "quality": {
            "units": "fraction",
            "value": 1.0
        },
        "entropy": {
            "units": "kJ_kg-K",
            "value": 6.621448429967412
        },
        "density": {
            "units": "kg_m3",
            "value": 0.8833255014572468
        }
    }
    """

    def __init__(self):
        pass

    @staticmethod
    def _format_dict(spec_name, value):
        return {
            ThermoConstants.UNITS: ThermoConstants.PROP_UNIT_DICT[spec_name],
            ThermoConstants.VALUE: value,
        }

    @staticmethod
    def _get_values(properties, spec_names: Union[str, list[str]]):
        if isinstance(spec_names, str):
            if spec_names not in properties:
                raise KeyError(
                    f"Missing property: {spec_names} not in {properties.keys()}"
                )
            return [properties[spec_names][ThermoConstants.VALUE]]
        elif isinstance(spec_names, list):
            if not all([spec in properties for spec in spec_names]):
                raise KeyError(
                    f"Missing property: {spec_names} not in {properties.keys()}"
                )
            return [properties[spec][ThermoConstants.VALUE] for spec in spec_names]

    def complete_thermodynamic_state(self, properties: dict, fluid: str):
        """
        Returns the value of the property specified by property_name
        given the properties and fluid.

        Parameters
        ----------
        properties : dict
            Dictionary of properties given for the calculation
        fluid : str
            Name of the fluid

        Returns
        -------
        dict
            Dictionary of properties with all properties filled in
        """
        self.normalize_properties_to_PH(properties, fluid)
        pressure__pa, enthalpy__kj_kg = self._get_values(
            properties, [ThermoConstants.PRESSURE, ThermoConstants.ENTHALPY]
        )
        for property_name in ThermoConstants.PROPERTIES:
            if property_name not in properties:
                try:
                    properties[property_name] = self._format_dict(
                        property_name,
                        self.get_property(
                            property_name, pressure__pa, enthalpy__kj_kg, fluid
                        ),
                    )
                except Exception as ex:
                    raise ex

        return properties

    def normalize_properties_to_PH(self, properties, fluid):
        """
        Gets the pressure and enthalpy from the properties given such that
        all the remaining properties can be calculated from these two.

        Parameters
        ----------
        properties : dict
            Dictionary of properties given for the calculation
        fluid : str
            Name of the fluid

        Returns
        -------
        float
            Value of the property
        """
        if (
            ThermoConstants.PRESSURE in properties
            and ThermoConstants.TEMPERATURE in properties
        ):
            pressure__pa, temp__c = self._get_values(
                properties, [ThermoConstants.PRESSURE, ThermoConstants.TEMPERATURE]
            )
            target_spec = ThermoConstants.ENTHALPY
            properties[target_spec] = self._format_dict(
                target_spec, self.get_enthalpy_from_PT(pressure__pa, temp__c, fluid)
            )
        elif (
            ThermoConstants.PRESSURE in properties
            and ThermoConstants.QUALITY in properties
        ):
            pressure__pa, quality = self._get_values(
                properties, [ThermoConstants.PRESSURE, ThermoConstants.QUALITY]
            )
            target_spec = ThermoConstants.ENTHALPY
            properties[target_spec] = self._format_dict(
                target_spec, self.get_enthalpy_from_PQ(pressure__pa, quality, fluid)
            )
        elif (
            ThermoConstants.PRESSURE in properties
            and ThermoConstants.ENTROPY in properties
        ):
            pressure__pa, entropy__kj_kgk = self._get_values(
                properties, [ThermoConstants.PRESSURE, ThermoConstants.ENTROPY]
            )
            target_spec = ThermoConstants.ENTHALPY
            properties[target_spec] = self._format_dict(
                target_spec,
                self.get_enthalpy_from_PS(pressure__pa, entropy__kj_kgk, fluid),
            )
        elif (
            ThermoConstants.TEMPERATURE in properties
            and ThermoConstants.QUALITY in properties
        ):
            temp__c, quality = self._get_values(
                properties, [ThermoConstants.TEMPERATURE, ThermoConstants.QUALITY]
            )
            target_spec = ThermoConstants.ENTHALPY
            properties[target_spec] = self._format_dict(
                target_spec, self.get_enthalpy_from_TQ(temp__c, quality, fluid)
            )
            target_spec = ThermoConstants.PRESSURE
            properties[target_spec] = self._format_dict(
                target_spec, self.get_pressure_from_TQ(temp__c, quality, fluid)
            )
        elif (
            ThermoConstants.TEMPERATURE in properties
            and ThermoConstants.ENTROPY in properties
        ):
            temp__c, entropy__kj_kgk = self._get_values(
                properties, [ThermoConstants.TEMPERATURE, ThermoConstants.ENTROPY]
            )
            target_spec = ThermoConstants.ENTHALPY
            properties[target_spec] = self._format_dict(
                target_spec, self.get_enthalpy_from_TS(temp__c, entropy__kj_kgk, fluid)
            )
            target_spec = ThermoConstants.PRESSURE
            enthalpy__kj_kg = self._get_values(properties, ThermoConstants.ENTHALPY)[0]
            properties[target_spec] = self._format_dict(
                target_spec,
                self.get_pressure_from_HS(enthalpy__kj_kg, entropy__kj_kgk, fluid),
            )
        elif (
            ThermoConstants.ENTHALPY in properties
            and ThermoConstants.ENTROPY in properties
        ):
            enthalpy__kj_kg, entropy__kj_kgk = self._get_values(
                properties, [ThermoConstants.ENTHALPY, ThermoConstants.ENTROPY]
            )
            target_spec = ThermoConstants.PRESSURE
            properties[target_spec] = self._format_dict(
                target_spec,
                self.get_pressure_from_HS(enthalpy__kj_kg, entropy__kj_kgk, fluid),
            )
        elif (
            ThermoConstants.PRESSURE in properties
            and ThermoConstants.DENSITY in properties
        ):
            pressure__pa, density__kg_m3 = self._get_values(
                properties, [ThermoConstants.PRESSURE, ThermoConstants.DENSITY]
            )
            target_spec = ThermoConstants.ENTHALPY
            properties[target_spec] = self._format_dict(
                target_spec,
                self.get_enthalpy_from_PD(pressure__pa, density__kg_m3, fluid),
            )
        elif (
            ThermoConstants.TEMPERATURE in properties
            and ThermoConstants.DENSITY in properties
        ):
            temp__c, density__kg_m3 = self._get_values(
                properties, [ThermoConstants.TEMPERATURE, ThermoConstants.DENSITY]
            )
            target_spec = ThermoConstants.ENTHALPY
            properties[target_spec] = self._format_dict(
                target_spec, self.get_enthalpy_from_TD(temp__c, density__kg_m3, fluid)
            )
            enthalpy__kj_kg = self._get_values(properties, ThermoConstants.ENTHALPY)[0]
            target_spec = ThermoConstants.PRESSURE
            properties[target_spec] = self._format_dict(
                target_spec,
                self.get_pressure_from_DH(density__kg_m3, enthalpy__kj_kg, fluid),
            )
        elif (
            ThermoConstants.ENTHALPY in properties
            and ThermoConstants.DENSITY in properties
        ):
            enthalpy__kj_kg, density__kg_m3 = self._get_values(
                properties, [ThermoConstants.ENTHALPY, ThermoConstants.DENSITY]
            )
            target_spec = ThermoConstants.PRESSURE
            properties[target_spec] = self._format_dict(
                target_spec,
                self.get_pressure_from_DH(density__kg_m3, enthalpy__kj_kg, fluid),
            )
        elif (
            ThermoConstants.ENTROPY in properties
            and ThermoConstants.DENSITY in properties
        ):
            entropy__kj_kgk, density__kg_m3 = self._get_values(
                properties, [ThermoConstants.ENTROPY, ThermoConstants.DENSITY]
            )
            target_spec = ThermoConstants.PRESSURE
            properties[target_spec] = self._format_dict(
                target_spec,
                self.get_pressure_from_SD(entropy__kj_kgk, density__kg_m3, fluid),
            )
            pressure__pa = self._get_values(properties, ThermoConstants.PRESSURE)[0]
            target_spec = ThermoConstants.ENTHALPY
            properties[target_spec] = self._format_dict(
                target_spec,
                self.get_enthalpy_from_PS(pressure__pa, entropy__kj_kgk, fluid),
            )
        elif (
            ThermoConstants.TEMPERATURE in properties
            and ThermoConstants.ENTHALPY in properties
        ):
            raise NotImplementedError(
                "Support for T, H specifications not implemented yet"
            )
        elif (
            ThermoConstants.QUALITY in properties
            and ThermoConstants.ENTHALPY in properties
        ):
            raise NotImplementedError(
                "Support for Q, H specifications not implemented yet"
            )
        elif (
            ThermoConstants.QUALITY in properties
            and ThermoConstants.ENTROPY in properties
        ):
            raise NotImplementedError(
                "Support for Q, S specifications not implemented yet"
            )
        elif (
            ThermoConstants.PRESSURE in properties
            and ThermoConstants.ENTHALPY in properties
        ):
            pass
        else:
            raise ValueError("Invalid properties: {}".format(properties))

        return properties

    @abstractmethod
    def get_enthalpy_from_PT(
        self, pressure__pa: float, temp__c: float, fluid: str
    ) -> float:
        pass

    @abstractmethod
    def get_enthalpy_from_PQ(
        self, pressure__pa: float, quality: float, fluid: str
    ) -> float:
        pass

    @abstractmethod
    def get_enthalpy_from_PS(
        self, pressure__pa: float, entropy__kg_kgk: float, fluid: str
    ) -> float:
        pass

    @abstractmethod
    def get_enthalpy_from_TQ(self, temp__c: float, quality: float, fluid: str) -> float:
        pass

    @abstractmethod
    def get_pressure_from_TQ(
        self, temp__c: float, enthalpy__kj_kg: float, fluid: str
    ) -> float:
        pass

    @abstractmethod
    def get_enthalpy_from_TS(
        self, temp__c: float, entropy__kg_kgk: float, fluid: str
    ) -> float:
        pass

    @abstractmethod
    def get_pressure_from_HS(
        self, enthalpy__kj_kg: float, entropy__kj_kgk: float, fluid: str
    ) -> float:
        pass

    @abstractmethod
    def get_enthalpy_from_PD(
        self, pressure__pa: float, density__kg_m3: float, fluid: str
    ) -> float:
        pass

    @abstractmethod
    def get_enthalpy_from_TD(
        self, temp__c: float, density__kg_m3: float, fluid: str
    ) -> float:
        pass

    @abstractmethod
    def get_pressure_from_DH(
        self, density__kg_m3: float, enthalpy__kj_kg: float, fluid: str
    ) -> float:
        pass

    @abstractmethod
    def get_pressure_from_SD(
        self, entropy__kj_kgk: float, density__kg_m3: float, fluid: str
    ) -> float:
        pass

    @abstractmethod
    def get_density_from_PQ(
        self, pressure__pa: float, quality: float, fluid: str
    ) -> float:
        pass

    @abstractmethod
    def get_molar_mass(self, fluid: str) -> float:
        pass

    @abstractmethod
    def get_property(
        self,
        property_name: str,
        pressure__pa: float,
        enthalpy__kj_kg: float,
        fluid: str,
    ) -> float:
        pass

    @abstractmethod
    def get_critical_pressure(self, fluid: str):
        pass
