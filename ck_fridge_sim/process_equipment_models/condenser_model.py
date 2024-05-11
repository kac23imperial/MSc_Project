from datetime import datetime
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel

from ck_fridge_sim.base_model_classes import (
    MeasurementBase,
    ModelBase,
    StateBase,
    StateDerivativeBase,
    TimePointBase,
)
from ck_fridge_sim.thermo_tools.ammonia_thermo import AmmoniaThermo
from ck_fridge_sim.thermo_tools.humid_air_properties import HumidAirProperties
from ck_fridge_sim.thermo_tools.unit_conversions import UnitConversions

if TYPE_CHECKING:
    from ck_fridge_sim.equipment_models import Pipe


class CondenserState(StateBase):
    kind: Literal["condenser"] = "condenser"
    m__kg: float
    h__kj_kg: float
    fan_speed_1__rpm: float
    fan_speed_2__rpm: float | None = None
    fan_speed_3__rpm: float | None = None
    fan_speed_4__rpm: float | None = None
    pump_1_on: float
    pump_2_on: float | None = None
    pump_3_on: float | None = None
    pump_4_on: float | None = None


class CondenserStateDerivative(CondenserState, StateDerivativeBase):
    pass


class CondenserTimePoint(TimePointBase):
    kind: Literal["condenser"] = "condenser"
    fan_speed_1__rpm: float
    fan_speed_2__rpm: float | None = None
    fan_speed_3__rpm: float | None = None
    fan_speed_4__rpm: float | None = None
    pump_1_on: bool
    pump_2_on: bool | None = None
    pump_3_on: bool | None = None
    pump_4_on: bool | None = None


class CondenserMeasurement(MeasurementBase):
    kind: Literal["condenser"] = "condenser"
    condenser_power__kw: float
    ambient_temp__c: float
    ambient_rh__perc: float
    pressure__psig: float
    fan_run_proof_1: bool
    fan_run_proof_2: bool | None = None
    fan_run_proof_3: bool | None = None
    fan_run_proof_4: bool | None = None
    pump_run_proof_1: bool
    pump_run_proof_2: bool | None = None
    pump_run_proof_3: bool | None = None
    pump_run_proof_4: bool | None = None


class CondenserModel(ModelBase):
    name: str
    surface_area__m2: float
    delta_p__pa: float
    volume__m3: float
    natural_htc__kw_m2_K: float
    forced_htc__kw_m2_K_rpm_hp: float
    fan_motors__hp: list[float]
    pump_motors__hp: list[float]
    fan_time_constants__s: list[float]
    pump_time_constants__s: list[float]

    @property
    def ha_props(self) -> HumidAirProperties:
        return HumidAirProperties()

    def _get_htc__kw_m2_K(self, fan_speed__rpm: float, fan__hp: float) -> float:
        return (
            self.natural_htc__kw_m2_K
            + self.forced_htc__kw_m2_K_rpm_hp
            * 10
            * 3600
            * (fan_speed__rpm * fan__hp / (10 * 3600)) ** 0.6
        )

    def get_vapor_state(self, state: CondenserState) -> AmmoniaThermo:
        try:
            inlet_thermo = AmmoniaThermo(
                enthalpy__kj_kg=state.h__kj_kg,
                density__kg_m3=state.m__kg / self.volume__m3,
            )
        except Exception:
            # try again but add random noise to the density
            inlet_thermo = AmmoniaThermo(
                enthalpy__kj_kg=state.h__kj_kg,
                density__kg_m3=float(
                    state.m__kg / self.volume__m3 + np.random.normal(0, 0.1)
                ),
            )
        return inlet_thermo

    def get_condensate_state(self, inlet_thermo: AmmoniaThermo) -> AmmoniaThermo:
        return AmmoniaThermo(
            pressure__pa=inlet_thermo.pressure__pa - self.delta_p__pa, quality=0.0
        )

    def _get_pumps_fans_list(
        self, pydantic_model: BaseModel
    ) -> tuple[list[float], list[bool]]:
        pumps_on = [getattr(pydantic_model, f"pump_{i}_on") for i in range(1, 5)]
        fan_speeds__rpm = [
            getattr(pydantic_model, f"fan_speed_{i}__rpm") for i in range(1, 5)
        ]

        pumps_on = [pump_on for pump_on in pumps_on if pump_on is not None]
        fan_speeds__rpm = [
            fan_speed__rpm
            for fan_speed__rpm in fan_speeds__rpm
            if fan_speed__rpm is not None
        ]
        if len(fan_speeds__rpm) != len(self.fan_motors__hp):
            raise ValueError(
                f"fan speeds length: {len(fan_speeds__rpm)} differs from fan hp length: {len(self.fan_motors__hp)}"
            )
        if len(pumps_on) != len(self.pump_motors__hp):
            raise ValueError(
                f"pumps_on length: {len(pumps_on)} differs from pump hp length: {len(self.pump_motors__hp)}"
            )
        return pumps_on, fan_speeds__rpm

    def solve_for_out_flow__kg_s(
        self,
        inlet_thermo: AmmoniaThermo,
        condensate_state: AmmoniaThermo,
        simulation_point: CondenserTimePoint,
        ambient_temp__c: float,
        ambient_rh__perc: float,
        state: CondenserState | None = None,
    ) -> float:
        pumps_on, fan_speeds__rpm = self._get_pumps_fans_list(state or simulation_point)
        wet_bulb_temp__c = self.ha_props.get_wet_bulb_temp(
            temp__c=ambient_temp__c,
            relative_humidity__perc=ambient_rh__perc,
        )
        saturation_fraction = sum(pumps_on) / len(pumps_on)
        external_temp__c = wet_bulb_temp__c * saturation_fraction + ambient_temp__c * (
            1 - saturation_fraction
        )
        delta_t__k = condensate_state.temp__c - external_temp__c

        htcs__kw_m2_K = [
            self._get_htc__kw_m2_K(fan_speed__rpm, fan__hp)
            for (fan_speed__rpm, fan__hp) in zip(fan_speeds__rpm, self.fan_motors__hp)
        ]
        heat_transfer_rate__kw = (
            sum(htcs__kw_m2_K) / len(htcs__kw_m2_K) * self.surface_area__m2 * delta_t__k
        )
        return heat_transfer_rate__kw / (
            inlet_thermo.enthalpy__kj_kg - condensate_state.enthalpy__kj_kg
        )

    def get_power__kw(self, simulation_point: CondenserTimePoint) -> float:
        pumps_on, fan_speeds__rpm = self._get_pumps_fans_list(simulation_point)
        fan_power__kw = sum(
            UnitConversions.hp_to_kw(fan__hp) * fan_speed__rpm / 3600
            for (fan_speed__rpm, fan__hp) in zip(fan_speeds__rpm, self.fan_motors__hp)
        )
        pump_power__kw = sum(
            UnitConversions.hp_to_kw(pump__hp) * pump_on
            for (pump__hp, pump_on) in zip(self.pump_motors__hp, pumps_on)
        )
        return fan_power__kw + pump_power__kw

    def get_state_derivative(
        self,
        state: CondenserState,
        simulation_point: CondenserTimePoint,
        inlets: list["Pipe"],
        flow_out__kg_s: float,
    ) -> CondenserStateDerivative:
        target_pumps_on, target_fan_speeds__rpm = self._get_pumps_fans_list(
            simulation_point
        )
        pumps_on, fan_speeds__rpm = self._get_pumps_fans_list(state)
        inlet_flow__kg_s, inlet_enthalpy__kj_kg = self._mix_inlet_streams(inlets)
        dm_dt = inlet_flow__kg_s - flow_out__kg_s
        dh_dt = (
            inlet_flow__kg_s / state.m__kg * (inlet_enthalpy__kj_kg - state.h__kj_kg)
        )
        dfan_dt = [
            self.first_order_dynamics(
                target=target_speed, state=current_speed, time_constant__s=time_constant
            )
            for (target_speed, current_speed, time_constant) in zip(
                target_fan_speeds__rpm,
                fan_speeds__rpm,
                self.fan_time_constants__s,
            )
        ]
        dpump_dt = [
            self.first_order_dynamics(
                target=target_pump, state=current_pump, time_constant__s=time_constant
            )
            for (target_pump, current_pump, time_constant) in zip(
                target_pumps_on,
                pumps_on,
                self.pump_time_constants__s,
            )
        ]
        return CondenserStateDerivative(
            name=state.name,
            date_time=simulation_point.date_time,
            m__kg=dm_dt,
            h__kj_kg=dh_dt,
            fan_speed_1__rpm=dfan_dt[0],
            fan_speed_2__rpm=None if len(dfan_dt) < 2 else dfan_dt[1],
            fan_speed_3__rpm=None if len(dfan_dt) < 3 else dfan_dt[2],
            fan_speed_4__rpm=None if len(dfan_dt) < 4 else dfan_dt[3],
            pump_1_on=dpump_dt[0],
            pump_2_on=None if len(dpump_dt) < 2 else dpump_dt[1],
            pump_3_on=None if len(dpump_dt) < 3 else dpump_dt[2],
            pump_4_on=None if len(dpump_dt) < 4 else dpump_dt[3],
        )

    def get_state_from_measurement(
        self,
        measurement: CondenserMeasurement,
        simulation_point: CondenserTimePoint,
    ) -> CondenserState:
        vapor_thermo = AmmoniaThermo(
            pressure__pa=UnitConversions.psig_to_pascal(measurement.pressure__psig),
            quality=1.0,
        )
        return CondenserState(
            name=simulation_point.name,
            date_time=simulation_point.date_time,
            m__kg=self.volume__m3 * vapor_thermo.density__kg_m3,
            h__kj_kg=vapor_thermo.enthalpy__kj_kg,
            fan_speed_1__rpm=simulation_point.fan_speed_1__rpm,
            fan_speed_2__rpm=simulation_point.fan_speed_2__rpm,
            fan_speed_3__rpm=simulation_point.fan_speed_3__rpm,
            fan_speed_4__rpm=simulation_point.fan_speed_4__rpm,
            pump_1_on=simulation_point.pump_1_on,
            pump_2_on=simulation_point.pump_2_on,
            pump_3_on=simulation_point.pump_3_on,
            pump_4_on=simulation_point.pump_4_on,
        )

    def initialize(
        self,
        date_time: datetime,
    ) -> tuple[CondenserState, CondenserMeasurement, CondenserTimePoint]:
        n_fans = len(self.fan_motors__hp)
        n_pumps = len(self.pump_motors__hp)
        fan_run_proofs = [True] * n_fans + [None] * (4 - n_fans)
        pump_run_proofs = [True] * n_pumps + [None] * (4 - n_pumps)
        condenser_measurement = CondenserMeasurement(
            name=self.name,
            date_time=date_time,
            condenser_power__kw=0,
            ambient_temp__c=25,
            ambient_rh__perc=50,
            pressure__psig=125,
            fan_run_proof_1=fan_run_proofs[0],
            fan_run_proof_2=fan_run_proofs[1],
            fan_run_proof_3=fan_run_proofs[2],
            fan_run_proof_4=fan_run_proofs[3],
            pump_run_proof_1=pump_run_proofs[0],
            pump_run_proof_2=pump_run_proofs[1],
            pump_run_proof_3=pump_run_proofs[2],
            pump_run_proof_4=pump_run_proofs[3],
        )
        actuator = CondenserTimePoint(
            name=self.name,
            date_time=date_time,
            fan_speed_1__rpm=3600,
            fan_speed_2__rpm=None if fan_run_proofs[1] is None else 3600,
            fan_speed_3__rpm=None if fan_run_proofs[2] is None else 3600,
            fan_speed_4__rpm=None if fan_run_proofs[3] is None else 3600,
            pump_1_on=pump_run_proofs[0],
            pump_2_on=None if pump_run_proofs[1] is None else True,
            pump_3_on=None if pump_run_proofs[2] is None else True,
            pump_4_on=None if pump_run_proofs[3] is None else True,
        )
        return (
            self.get_state_from_measurement(condenser_measurement, actuator),
            condenser_measurement,
            actuator,
        )
