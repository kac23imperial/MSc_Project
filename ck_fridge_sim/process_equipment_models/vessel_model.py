from datetime import datetime
from typing import TYPE_CHECKING, Callable, Literal

import numpy as np
from scipy.optimize import fsolve

from ck_fridge_sim.base_model_classes import (
    MeasurementBase,
    ModelBase,
    StateBase,
    StateDerivativeBase,
    TimePointBase,
)
from ck_fridge_sim.thermo_tools.ammonia_thermo import AmmoniaThermo
from ck_fridge_sim.thermo_tools.unit_conversions import UnitConversions

if TYPE_CHECKING:
    from ck_fridge_sim.equipment_models import Pipe


class VesselState(StateBase):
    kind: Literal["vessel"] = "vessel"
    m_l__kg: float
    m_v__kg: float
    h_l__kj_kg: float
    h_v__kj_kg: float


class VesselStateDerivative(VesselState, StateDerivativeBase):
    pass


class VesselMeasurement(MeasurementBase):
    kind: Literal["vessel"] = "vessel"
    pressure__psig: float
    level__m: float
    level__perc: float
    interface_pressure__psig: float


class VesselModel(ModelBase):
    """
    Parameters for a vessel model.
    """

    name: str
    diameter__m: float
    length__m: float
    mass_transfer_coefficient__m_s: float
    default_temp__f: float
    shape: str = "cylinder"
    max_pressure__psig: float | None = None
    orientation: str = "vertical"  # vertical or horizontal
    insulation_thickness__m: float = 0.0254
    insulation_thermal_conductivity__w_m_k: float = 0.05

    @property
    def volume__m3(self) -> float:
        if self.shape == "cylinder":
            volume__m3 = np.pi * (self.diameter__m / 2) ** 2 * self.length__m
        else:
            raise NotImplementedError("The shape must be a cylinder.")

        return volume__m3

    @property
    def surface_area__m2(self) -> float:
        if self.shape == "cylinder":
            if self.orientation == "vertical":
                surface_area__m2 = np.pi * (self.diameter__m / 2) ** 2
            elif self.orientation == "horizontal":
                surface_area__m2 = self.diameter__m * self.length__m
            else:
                raise NotImplementedError(
                    "The orientation must be vertical or horizontal."
                )
        else:
            raise NotImplementedError("The shape must be a cylinder.")

        return surface_area__m2

    @property
    def external_surface_area__m2(self) -> float:
        if self.shape == "cylinder":
            surface_area__m2 = np.pi * self.diameter__m * self.length__m
        else:
            raise NotImplementedError("The orientation must be vertical or horizontal.")

        return surface_area__m2

    @property
    def heat_transfer_coefficient__kw_m2_k(self) -> float:
        return (
            self.insulation_thermal_conductivity__w_m_k
            / self.insulation_thickness__m
            / 1000
        )

    def _get_temp__c_func(self, h_l__kj_kg: float) -> Callable[[float], float]:
        def rhs(temp__c):
            return (
                h_l__kj_kg - AmmoniaThermo(temp__c=temp__c, quality=0.0).enthalpy__kj_kg
            )

        return rhs

    def _solve_for_temperature(self, h_l__kj_mol: float) -> float:
        lambda_func = self._get_temp__c_func(h_l__kj_mol)
        temp__c, infodict, ier, mesg = fsolve(
            lambda_func,
            UnitConversions.fahrenheit_to_celsius(self.default_temp__f),
            full_output=True,
            xtol=1e-3,
        )
        assert ier == 1, mesg
        return temp__c[0]

    def get_liquid_state(self, state: VesselState) -> AmmoniaThermo:
        temp__c = self._solve_for_temperature(state.h_l__kj_kg)
        self.default_temp__f = UnitConversions.celsius_to_fahrenheit(temp__c)
        return AmmoniaThermo(temp__c=temp__c, quality=0.0)

    def get_vapor_state(
        self, state: VesselState, liquid_state: AmmoniaThermo
    ) -> AmmoniaThermo:
        rho_v__kg_m3 = state.m_v__kg * (
            liquid_state.density__kg_m3
            / (liquid_state.density__kg_m3 * self.volume__m3 - state.m_l__kg)
        )
        try:
            vapor_thermo = AmmoniaThermo(
                enthalpy__kj_kg=state.h_v__kj_kg, density__kg_m3=rho_v__kg_m3
            )
        except Exception:
            # try again but add random noise to the density
            vapor_thermo = AmmoniaThermo(
                enthalpy__kj_kg=state.h_v__kj_kg,
                density__kg_m3=float(rho_v__kg_m3 + np.random.normal(0, 0.1)),
            )
        return vapor_thermo

    @staticmethod
    def _get_level__m_func(f: float, r: float) -> Callable[[float], float]:
        def rhs(h: float) -> float:
            h = max(1e-3, min(2 * r - 1e-3, h))
            return (
                r**2 * np.arccos((r - h) / r)
                - (r - h) * np.sqrt(2 * r * h - h**2)
                - f * r**2 * np.pi
            )

        return rhs

    def level__perc(self, level__m: float) -> float:
        if self.orientation.lower() == "vertical":
            level__perc = level__m / self.length__m * 100
        elif self.orientation.lower() == "horizontal":
            level__perc = level__m / self.diameter__m * 100
        else:
            raise NotImplementedError("The orientation must be vertical or horizontal.")

        return float(level__perc)

    def level__m(self, liquid_density__kg_m3: float, m_l__kg: float) -> float:
        volume_liquid__m3 = m_l__kg / liquid_density__kg_m3
        if self.orientation == "vertical":
            level__m = volume_liquid__m3 / self.volume__m3 * self.length__m
        elif self.orientation == "horizontal":
            fill_fraction = volume_liquid__m3 / self.volume__m3
            rad = self.diameter__m / 2
            func = self._get_level__m_func(fill_fraction, rad)
            level__m = fsolve(func, rad)[0]
        else:
            raise NotImplementedError("The orientation must be vertical or horizontal.")

        return float(level__m)

    def get_state_from_level(
        self,
        h: float,
        liquid_density__kg_m3: float,
        vapor_density__kg_m3: float,
    ) -> tuple[float, float]:
        if self.orientation == "vertical":
            fill_fraction = h / self.length__m
        elif self.orientation == "horizontal":
            r = self.diameter__m / 2
            fill_fraction = (
                r**2 * np.arccos((r - h) / r) - (r - h) * np.sqrt(2 * r * h - h**2)
            ) / (r**2 * np.pi)
        else:
            raise NotImplementedError("The orientation must be vertical or horizontal.")

        m_l__kg = fill_fraction * self.volume__m3 * liquid_density__kg_m3
        m_v__kg = (1 - fill_fraction) * self.volume__m3 * vapor_density__kg_m3
        return m_l__kg, m_v__kg

    def get_vessel_heat_rate__kw(
        self, liquid_thermo: AmmoniaThermo, external_temp__c: float
    ) -> float:
        return (
            self.heat_transfer_coefficient__kw_m2_k
            * self.external_surface_area__m2
            * (external_temp__c - liquid_thermo.temp__c)
        )

    def get_state_derivative(
        self,
        state: VesselState,
        simulation_point: TimePointBase | None,
        inlets: list["Pipe"],
        vapor_thermo: AmmoniaThermo,
        liquid_thermo: AmmoniaThermo,
        liquid_flow_out__kg_s: float,
        vapor_flow_out__kg_s: float,
        heat_transfer_rate__kw: float,
    ) -> VesselStateDerivative:
        inlet_flow__kg_s, inlet_enthalpy__kj_kg = self._mix_inlet_streams(inlets)
        # flash the inlet stream
        if inlet_flow__kg_s > 1e-3:
            flash = AmmoniaThermo(
                enthalpy__kj_kg=inlet_enthalpy__kj_kg,
                pressure__pa=vapor_thermo.pressure__pa,
            ).quality
            inlet_liquid_flow__kg_s = inlet_flow__kg_s * (1 - flash)
            inlet_vapor_flow__kg_s = inlet_flow__kg_s * flash
            flash_liquid_thermo = AmmoniaThermo(
                pressure__pa=vapor_thermo.pressure__pa, quality=0.0
            )
            inlet_liquid_enthalpy__kw = (
                flash_liquid_thermo.enthalpy__kj_kg * inlet_liquid_flow__kg_s
            )
            if flash > 0:
                flash_vapor_enthalpy__kj_kq = (
                    inlet_enthalpy__kj_kg
                    - flash_liquid_thermo.enthalpy__kj_kg * (1 - flash)
                ) / (flash + 1e-6)
                inlet_vapor_enthalpy__kw = (
                    flash_vapor_enthalpy__kj_kq * inlet_vapor_flow__kg_s
                )
            else:
                inlet_vapor_enthalpy__kw = 0

        else:
            inlet_liquid_flow__kg_s = 0
            inlet_vapor_flow__kg_s = 0
            inlet_liquid_enthalpy__kw = 0
            inlet_vapor_enthalpy__kw = 0

        interface_thermo = AmmoniaThermo(
            pressure__pa=liquid_thermo.pressure__pa, quality=1.0
        )
        delta_c__kg_m3 = interface_thermo.density__kg_m3 - vapor_thermo.density__kg_m3
        m_dot_evap__kg_s = (
            self.surface_area__m2 * self.mass_transfer_coefficient__m_s * delta_c__kg_m3
        )

        # State derivatives
        dm_l_dt = inlet_liquid_flow__kg_s - liquid_flow_out__kg_s - m_dot_evap__kg_s
        dm_v_dt = inlet_vapor_flow__kg_s - vapor_flow_out__kg_s + m_dot_evap__kg_s

        dh_l_dt = (
            inlet_liquid_enthalpy__kw
            - inlet_liquid_flow__kg_s * state.h_l__kj_kg
            + heat_transfer_rate__kw
            - m_dot_evap__kg_s
            * (vapor_thermo.enthalpy__kj_kg - liquid_thermo.enthalpy__kj_kg)
        ) / state.m_l__kg
        dh_v_dt = (
            inlet_vapor_enthalpy__kw - inlet_vapor_flow__kg_s * state.h_v__kj_kg
        ) / state.m_v__kg
        return VesselStateDerivative(
            name=state.name,
            date_time=state.date_time,
            m_l__kg=dm_l_dt,
            m_v__kg=dm_v_dt,
            h_l__kj_kg=dh_l_dt,
            h_v__kj_kg=dh_v_dt,
        )

    def get_state_from_measurement(
        self,
        measurement: VesselMeasurement,
        simulation_point: None = None,
    ) -> VesselState:
        liquid_thermo = AmmoniaThermo(
            pressure__pa=UnitConversions.psig_to_pascal(
                measurement.interface_pressure__psig
            ),
            quality=0.0,
        )
        vapor_thermo = AmmoniaThermo(
            pressure__pa=UnitConversions.psig_to_pascal(measurement.pressure__psig),
            quality=1.0,
        )
        m_l__kg, m_v__kg = self.get_state_from_level(
            measurement.level__m,
            liquid_thermo.density__kg_m3,
            vapor_thermo.density__kg_m3,
        )
        return VesselState(
            name=measurement.name,
            date_time=measurement.date_time,
            m_l__kg=m_l__kg,
            m_v__kg=m_v__kg,
            h_l__kj_kg=liquid_thermo.enthalpy__kj_kg,
            h_v__kj_kg=vapor_thermo.enthalpy__kj_kg,
        )

    def initialize(
        self,
        date_time: datetime,
    ) -> tuple[VesselState, VesselMeasurement]:
        vapor_thermo = AmmoniaThermo(
            temp__c=UnitConversions.fahrenheit_to_celsius(self.default_temp__f),
            quality=1.0,
        )
        if self.orientation == "vertical":
            level__m = self.length__m / 3
        else:
            level__m = self.diameter__m / 3

        vessel_measurement = VesselMeasurement(
            name=self.name,
            date_time=date_time,
            pressure__psig=vapor_thermo.pressure__psig,
            level__m=level__m,
            level__perc=self.level__perc(level__m),
            interface_pressure__psig=vapor_thermo.pressure__psig,
        )
        return self.get_state_from_measurement(vessel_measurement), vessel_measurement
