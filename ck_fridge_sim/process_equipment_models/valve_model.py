from datetime import datetime
from typing import TYPE_CHECKING, Literal

from ck_fridge_sim.base_model_classes import (
    MeasurementBase,
    ModelBase,
    StateBase,
    StateDerivativeBase,
    TimePointBase,
)

if TYPE_CHECKING:
    from ck_fridge_sim.equipment_models import Pipe


class ValveState(StateBase):
    kind: Literal["valve"] = "valve"
    valve_position__perc: float


class ValveStateDerivative(ValveState, StateDerivativeBase):
    pass


class ValveTimePoint(TimePointBase):
    kind: Literal["valve"] = "valve"
    valve_position__perc: float


class ValveMeasurement(MeasurementBase):
    kind: Literal["valve"] = "valve"
    valve_flow_rate__kg_s: float


class ValveModel(ModelBase):
    name: str
    flow_rate__kg_s: float
    time_constant__s: float = 15.0

    def get_flow_rate__kg_s(
        self, simulation_point: ValveTimePoint, state: ValveState | None = None
    ) -> float:
        valve_position__perc = (
            state.valve_position__perc
            if state is not None
            else simulation_point.valve_position__perc
        )
        return self.flow_rate__kg_s * valve_position__perc / 100.0

    def get_state_derivative(
        self,
        state: ValveState,
        simulation_point: ValveTimePoint,
        inlets: list["Pipe"] = None,
    ) -> StateDerivativeBase:
        if (
            abs(simulation_point.valve_position__perc - state.valve_position__perc)
            < 0.01
        ):
            return ValveStateDerivative(
                name=state.name,
                date_time=simulation_point.date_time,
                valve_position__perc=0.0,
            )

        dvalve_position_dt = self.first_order_dynamics(
            target=simulation_point.valve_position__perc,
            state=state.valve_position__perc,
            time_constant__s=self.time_constant__s,
        )
        return ValveStateDerivative(
            name=state.name,
            date_time=simulation_point.date_time,
            valve_position__perc=dvalve_position_dt,
        )

    def get_state_from_measurement(
        self,
        measurement: ValveMeasurement,
        simulation_point: ValveTimePoint,
    ) -> ValveState:
        return ValveState(
            name=simulation_point.name,
            date_time=simulation_point.date_time,
            valve_position__perc=simulation_point.valve_position__perc,
        )

    def initialize(
        self, date_time: datetime
    ) -> tuple[ValveState, ValveMeasurement, ValveTimePoint]:
        valve_actuator = ValveTimePoint(
            name=self.name, date_time=date_time, valve_position__perc=50.0
        )
        valve_measurement = ValveMeasurement(
            name=self.name, date_time=date_time, valve_flow_rate__kg_s=2.0
        )
        return (
            self.get_state_from_measurement(valve_measurement, valve_actuator),
            valve_measurement,
            valve_actuator,
        )
