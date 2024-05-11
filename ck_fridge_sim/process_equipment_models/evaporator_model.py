from datetime import datetime
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

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


class EvaporatorState(StateBase):
    kind: Literal["evaporator"] = "evaporator"
    h__kj_kg: float
    fan_speed_1__rpm: float
    fan_speed_2__rpm: float | None = None
    fan_speed_3__rpm: float | None = None
    fan_speed_4__rpm: float | None = None
    fan_speed_5__rpm: float | None = None
    fan_speed_6__rpm: float | None = None
    liquid_solenoid_on: float


class EvaporatorStateDerivative(EvaporatorState, StateDerivativeBase):
    pass


class EvaporatorTimePoint(TimePointBase):
    kind: Literal["evaporator"] = "evaporator"
    fan_speed_1__rpm: float
    fan_speed_2__rpm: float | None = None
    fan_speed_3__rpm: float | None = None
    fan_speed_4__rpm: float | None = None
    fan_speed_5__rpm: float | None = None
    fan_speed_6__rpm: float | None = None
    liquid_solenoid_on: bool | float | int


class EvaporatorMeasurement(MeasurementBase):
    kind: Literal["evaporator"] = "evaporator"
    evaporator_power__kw: float
    suction_pressure__psig: float
    room_temp__c: float
    fan_run_proof_1: bool
    fan_run_proof_2: bool | None = None
    fan_run_proof_3: bool | None = None
    fan_run_proof_4: bool | None = None
    fan_run_proof_5: bool | None = None
    fan_run_proof_6: bool | None = None


class EvaporatorModel(ModelBase):
    name: str
    surface_area__m2: float
    holdup__kg: float
    natural_htc__kw_m2_K: float
    forced_htc__kw_m2_K_rpm_hp: float
    liquid_flow__kg_s: float
    fan_motors__hp: float
    number_of_fans: int
    fan_time_constants__s: float = 20
    liquid_solenoid_time_constant__s: float = 20

    def _get_fans_list(self, pydantic_model: BaseModel) -> list[float]:
        fan_speeds__rpm = [
            getattr(pydantic_model, f"fan_speed_{i}__rpm") for i in range(1, 7)
        ]

        fan_speeds__rpm = [
            fan_speed__rpm
            for fan_speed__rpm in fan_speeds__rpm
            if fan_speed__rpm is not None
        ]
        return fan_speeds__rpm

    def _get_htc__kw_m2_K(self, fan_speed__rpm: float, fan__hp: float) -> float:
        return (
            self.natural_htc__kw_m2_K
            + self.forced_htc__kw_m2_K_rpm_hp
            * 10
            * 3600
            * (fan_speed__rpm * fan__hp / (10 * 3600)) ** 0.6
        )

    def get_liquid_state(
        self, state: EvaporatorState, inlet_pressure__psig: float
    ) -> AmmoniaThermo:
        return AmmoniaThermo(
            enthalpy__kj_kg=state.h__kj_kg,
            pressure__pa=UnitConversions.psig_to_pascal(inlet_pressure__psig),
        )

    def get_heat_transfer_rate__kw(
        self,
        inlet_thermo: AmmoniaThermo,
        simulation_point: EvaporatorTimePoint,
        room_temp__c: float,
        state: EvaporatorState | None = None,
    ) -> float:
        fan_speeds__rpm = self._get_fans_list(state or simulation_point)
        delta_t__k = room_temp__c - inlet_thermo.temp__c
        htcs__kw_m2_K = [
            self._get_htc__kw_m2_K(fan_speed__rpm, self.fan_motors__hp)
            for fan_speed__rpm in fan_speeds__rpm
        ]
        htc__kw_m2_K = sum(htcs__kw_m2_K) / len(htcs__kw_m2_K)
        return htc__kw_m2_K * self.surface_area__m2 * delta_t__k

    def solve_for_out_flow__kg_s(
        self,
        simulation_point: EvaporatorTimePoint,
        pump_on: bool,
        state: EvaporatorState | None = None,
    ) -> float:
        liquid_solenoid_on = (
            state.liquid_solenoid_on if state is not None else simulation_point
        )
        return liquid_solenoid_on * pump_on * self.liquid_flow__kg_s

    def get_power__kw(self, simulation_point: EvaporatorTimePoint) -> float:
        fan_speeds__rpm = self._get_fans_list(simulation_point)
        return sum(
            UnitConversions.hp_to_kw(self.fan_motors__hp) * speed / 3600
            for speed in fan_speeds__rpm
        )

    def get_state_derivative(
        self,
        state: EvaporatorState,
        simulation_point: EvaporatorTimePoint,
        inlets: list["Pipe"],
        heat_transfer_rate__kw: float,
    ) -> EvaporatorStateDerivative:
        target_fan_speeds__rpm = self._get_fans_list(simulation_point)
        fan_speeds__rpm = self._get_fans_list(state)
        inlet_flow__kg_s, inlet_enthalpy__kj_kg = self._mix_inlet_streams(inlets)
        dh_dt = (
            inlet_flow__kg_s * (inlet_enthalpy__kj_kg - state.h__kj_kg)
            + heat_transfer_rate__kw
        ) / self.holdup__kg

        dfan_dt = [
            self.first_order_dynamics(
                target=target_speed,
                state=current_speed,
                time_constant__s=self.fan_time_constants__s,
            )
            for (target_speed, current_speed) in zip(
                target_fan_speeds__rpm,
                fan_speeds__rpm,
            )
        ]
        dliquid_solenoid_dt = self.first_order_dynamics(
            target=simulation_point.liquid_solenoid_on,
            state=state.liquid_solenoid_on,
            time_constant__s=self.liquid_solenoid_time_constant__s,
        )
        return EvaporatorStateDerivative(
            name=state.name,
            date_time=simulation_point.date_time,
            h__kj_kg=dh_dt,
            fan_speed_1__rpm=dfan_dt[0],
            fan_speed_2__rpm=None if len(dfan_dt) < 2 else dfan_dt[1],
            fan_speed_3__rpm=None if len(dfan_dt) < 3 else dfan_dt[2],
            fan_speed_4__rpm=None if len(dfan_dt) < 4 else dfan_dt[3],
            fan_speed_5__rpm=None if len(dfan_dt) < 5 else dfan_dt[4],
            fan_speed_6__rpm=None if len(dfan_dt) < 6 else dfan_dt[5],
            liquid_solenoid_on=dliquid_solenoid_dt,
        )

    def get_state_from_measurement(
        self,
        measurement: EvaporatorMeasurement,
        simulation_point: EvaporatorTimePoint,
    ) -> EvaporatorState:
        thermo = AmmoniaThermo(
            pressure__pa=UnitConversions.psig_to_pascal(
                measurement.suction_pressure__psig
            ),
            quality=0.0,
        )
        return EvaporatorState(
            name=simulation_point.name,
            date_time=simulation_point.date_time,
            h__kj_kg=thermo.enthalpy__kj_kg,
            fan_speed_1__rpm=simulation_point.fan_speed_1__rpm,
            fan_speed_2__rpm=simulation_point.fan_speed_2__rpm,
            fan_speed_3__rpm=simulation_point.fan_speed_3__rpm,
            fan_speed_4__rpm=simulation_point.fan_speed_4__rpm,
            fan_speed_5__rpm=simulation_point.fan_speed_5__rpm,
            fan_speed_6__rpm=simulation_point.fan_speed_6__rpm,
            liquid_solenoid_on=simulation_point.liquid_solenoid_on,
        )

    def initialize(
        self,
        date_time: datetime,
        default_temp__f: float,
    ) -> tuple[EvaporatorState, EvaporatorMeasurement, EvaporatorTimePoint]:
        vapor_thermo = AmmoniaThermo(
            temp__c=UnitConversions.fahrenheit_to_celsius(default_temp__f),
            quality=1.0,
        )
        fan_run_proofs = [True] * self.number_of_fans + [None] * (
            6 - self.number_of_fans
        )
        evaporator_measurement = EvaporatorMeasurement(
            name=self.name,
            date_time=date_time,
            evaporator_power__kw=0.0,
            suction_pressure__psig=vapor_thermo.pressure__psig,
            room_temp__c=UnitConversions.fahrenheit_to_celsius(default_temp__f + 10),
            fan_run_proof_1=fan_run_proofs[0],
            fan_run_proof_2=fan_run_proofs[1],
            fan_run_proof_3=fan_run_proofs[2],
            fan_run_proof_4=fan_run_proofs[3],
            fan_run_proof_5=fan_run_proofs[4],
            fan_run_proof_6=fan_run_proofs[5],
        )
        evaporator_actuator = EvaporatorTimePoint(
            name=self.name,
            date_time=date_time,
            fan_speed_1__rpm=3600 if fan_run_proofs[0] else None,
            fan_speed_2__rpm=3600 if fan_run_proofs[1] else None,
            fan_speed_3__rpm=3600 if fan_run_proofs[2] else None,
            fan_speed_4__rpm=3600 if fan_run_proofs[3] else None,
            fan_speed_5__rpm=3600 if fan_run_proofs[4] else None,
            fan_speed_6__rpm=3600 if fan_run_proofs[5] else None,
            liquid_solenoid_on=1.0,
        )
        return (
            self.get_state_from_measurement(
                evaporator_measurement,
                evaporator_actuator,
            ),
            evaporator_measurement,
            evaporator_actuator,
        )
