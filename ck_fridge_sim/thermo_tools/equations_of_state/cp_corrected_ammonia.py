from CoolProp import CoolProp as CP

from ck_fridge_sim.thermo_tools.equations_of_state.cool_prop_rks import CoolPropRKS
from ck_fridge_sim.thermo_tools.thermo_constants import ThermoConstants
from ck_fridge_sim.thermo_tools.unit_conversions import UnitConversions


class CPCorrectedAmmonia(CoolPropRKS):
    HIGH_PRESSURE_LIMIT__PA = UnitConversions.psig_to_pascal(-1.6466035982827165)
    LOW_PRESSURE_LIMIT__PA = UnitConversions.psig_to_pascal(-4.300069011276246)

    def get_enthalpy_from_PQ(self, pressure__pa: float, quality: float, fluid: str):
        assert fluid in ["Ammonia", "NH3"], f"Fluid must be ammonia, not {fluid}"
        if pressure__pa < self.HIGH_PRESSURE_LIMIT__PA:
            h_low__kj_kg = (
                CP.PropsSI(
                    self.ENTHALPY_KEY,
                    self.PRESSURE_KEY,
                    self.LOW_PRESSURE_LIMIT__PA,
                    self.QUALITY_KEY,
                    quality,
                    fluid,
                )
                / 1000
            )
            h_high__kj_kg = (
                CP.PropsSI(
                    self.ENTHALPY_KEY,
                    self.PRESSURE_KEY,
                    self.HIGH_PRESSURE_LIMIT__PA,
                    self.QUALITY_KEY,
                    quality,
                    fluid,
                )
                / 1000
            )
            return h_low__kj_kg + (pressure__pa - self.LOW_PRESSURE_LIMIT__PA) / (
                self.HIGH_PRESSURE_LIMIT__PA - self.LOW_PRESSURE_LIMIT__PA
            ) * (h_high__kj_kg - h_low__kj_kg)

        return super().get_enthalpy_from_PQ(pressure__pa, quality, fluid)

    def get_property(
        self,
        property_name: str,
        pressure__pa: float,
        enthalpy__kj_kg: float,
        fluid: str,
    ):
        assert fluid in ["Ammonia", "NH3"], f"Fluid must be ammonia, not {fluid}"
        if pressure__pa < self.HIGH_PRESSURE_LIMIT__PA:
            property_low = CP.PropsSI(
                self.PROP_TO_KEY_DICT[property_name],
                self.PRESSURE_KEY,
                self.LOW_PRESSURE_LIMIT__PA,
                self.ENTHALPY_KEY,
                enthalpy__kj_kg * 1000,
                fluid,
            )
            property_high = CP.PropsSI(
                self.PROP_TO_KEY_DICT[property_name],
                self.PRESSURE_KEY,
                self.HIGH_PRESSURE_LIMIT__PA,
                self.ENTHALPY_KEY,
                enthalpy__kj_kg * 1000,
                fluid,
            )
            property_value = property_low + (
                pressure__pa - self.LOW_PRESSURE_LIMIT__PA
            ) / (self.HIGH_PRESSURE_LIMIT__PA - self.LOW_PRESSURE_LIMIT__PA) * (
                property_high - property_low
            )
        else:
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
