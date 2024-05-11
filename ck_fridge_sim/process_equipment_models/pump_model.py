from datetime import datetime
from typing import TYPE_CHECKING, Literal

from ck_fridge_sim.base_model_classes import (
    MeasurementBase,
    ModelBase,
    StateBase,
    StateDerivativeBase,
    TimePointBase,
)
from ck_fridge_sim.thermo_tools.unit_conversions import UnitConversions

if TYPE_CHECKING:
    from ck_fridge_sim.equipment_models import Pipe


class PumpState(StateBase):
    pump_on: float
    kind: Literal["pump"] = "pump"


class PumpStateDerivative(PumpState, StateDerivativeBase):
    pass


class PumpTimePoint(TimePointBase):
    kind: Literal["pump"] = "pump"
    pump_on: bool


class PumpMeasurement(MeasurementBase):
    kind: Literal["pump"] = "pump"
    pump_power__kw: float
    run_proof: bool


class PumpModel(ModelBase):
    name: str
    pump__hp: float
    time_constant__s: float = 5

    def is_flowing(
        self, simulation_point: PumpTimePoint, state: PumpState | None = None
    ) -> bool:
        return state.pump_on if state is not None else simulation_point.pump_on

    def get_power__kw(self, simulation_point: PumpTimePoint) -> float:
        return UnitConversions.hp_to_kw(self.pump__hp) * simulation_point.pump_on

    def get_state_derivative(
        self,
        state: PumpState,
        simulation_point: PumpTimePoint,
        inlets: list["Pipe"] = None,
    ) -> StateDerivativeBase:
        dpump_dt = self.first_order_dynamics(
            target=simulation_point.pump_on,
            state=state.pump_on,
            time_constant__s=self.time_constant__s,
        )
        return PumpStateDerivative(
            name=state.name, date_time=simulation_point.date_time, pump_on=dpump_dt
        )

    def get_state_from_measurement(
        self,
        measurement: PumpMeasurement,
        simulation_point: PumpTimePoint,
    ) -> PumpState:
        return PumpState(
            name=simulation_point.name,
            date_time=simulation_point.date_time,
            pump_on=simulation_point.pump_on,
        )

    def initialize(
        self, date_time: datetime
    ) -> tuple[PumpState, PumpMeasurement, PumpTimePoint]:
        pump_actuator = PumpTimePoint(name=self.name, date_time=date_time, pump_on=True)
        pump_measurement = PumpMeasurement(
            name=self.name,
            date_time=date_time,
            pump_power__kw=self.get_power__kw(pump_actuator),
            run_proof=True,
        )
        return (
            self.get_state_from_measurement(pump_measurement, pump_actuator),
            pump_measurement,
            pump_actuator,
        )
