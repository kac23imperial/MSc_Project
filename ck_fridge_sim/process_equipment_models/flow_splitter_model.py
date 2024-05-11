from datetime import datetime
from typing import TYPE_CHECKING, Literal

from ck_fridge_sim.base_model_classes import (
    MeasurementBase,
    ModelBase,
    StateBase,
    StateDerivativeBase,
)

if TYPE_CHECKING:
    from ck_fridge_sim.equipment_models import Pipe


class FlowSplitterState(StateBase):
    kind: Literal["flow_splitter"] = "flow_splitter"
    flow_fraction_1: float
    flow_fraction_2: float | None
    flow_fraction_3: float | None
    flow_fraction_4: float | None


class FlowSplitterStateDerivative(FlowSplitterState, StateDerivativeBase):
    pass


class FlowSplitterMeasurement(MeasurementBase):
    kind: Literal["flow_splitter"] = "flow_splitter"
    flow_fraction_1: float
    flow_fraction_2: float | None
    flow_fraction_3: float | None
    flow_fraction_4: float | None


class FlowSplitterModel(ModelBase):
    name: str
    tau__psig: float = 3000

    def flow_distribution(self, state: FlowSplitterState) -> list[float]:
        scores = [state.flow_fraction_1]
        if state.flow_fraction_2 is not None:
            scores.append(state.flow_fraction_2)
        if state.flow_fraction_3 is not None:
            scores.append(state.flow_fraction_3)
        if state.flow_fraction_4 is not None:
            scores.append(state.flow_fraction_4)

        limited = [min(max(score, 0), 1) for score in scores]
        return [score / sum(limited) for score in limited]

    def mix_flows(self, pipe_list: list["Pipe"]) -> tuple[float, float]:
        total_mass_flow__kg_s, total_enthalpy_flow__kw = 0, 0
        for pipe in pipe_list:
            total_mass_flow__kg_s += pipe.flowrate__kg_s
            total_enthalpy_flow__kw += pipe.flowrate__kg_s * pipe.enthalpy__kj_kg

        enthalpy__kj_kg = total_enthalpy_flow__kw / (total_mass_flow__kg_s + 1e-6)
        return total_mass_flow__kg_s, enthalpy__kj_kg

    def split_flow(
        self,
        state: FlowSplitterState,
        inlet_pipe_list: list["Pipe"],
        outlet_pipe_list: list["Pipe"],
    ) -> None:
        flow_distribution = self.flow_distribution(state)
        total_mass_flow__kg_s, enthalpy__kj_kg = self.mix_flows(inlet_pipe_list)
        for i, pipe in enumerate(outlet_pipe_list):
            pipe.flowrate__kg_s = flow_distribution[i] * total_mass_flow__kg_s
            pipe.enthalpy__kj_kg = enthalpy__kj_kg

    def get_state_derivative(
        self,
        state: FlowSplitterState,
        simulation_point: None,
        inlets: list["Pipe"],
        pressures: list[float],
    ) -> FlowSplitterStateDerivative:
        dynamics = []
        for pressure in pressures:
            dynamics.append(
                self.first_order_dynamics(
                    pressures[0], pressure, time_constant__s=self.tau__psig
                )
            )
        return FlowSplitterStateDerivative(
            name=state.name,
            date_time=state.date_time,
            flow_fraction_1=dynamics[0],
            flow_fraction_2=dynamics[1] if len(dynamics) > 1 else None,
            flow_fraction_3=dynamics[2] if len(dynamics) > 2 else None,
            flow_fraction_4=dynamics[3] if len(dynamics) > 3 else None,
        )

    def get_state_from_measurement(
        self,
        measurement: FlowSplitterMeasurement,
        simulation_point: None,
    ) -> FlowSplitterState:
        return FlowSplitterState(
            name=measurement.name,
            date_time=measurement.date_time,
            flow_fraction_1=measurement.flow_fraction_1,
            flow_fraction_2=measurement.flow_fraction_2,
            flow_fraction_3=measurement.flow_fraction_3,
            flow_fraction_4=measurement.flow_fraction_4,
        )

    def initialize(
        self, date_time: datetime, n_condensers: int
    ) -> tuple[FlowSplitterState, FlowSplitterMeasurement]:
        flow_fraction = 1.0 / n_condensers
        flow_fraction_list = [flow_fraction] * n_condensers + [None] * (
            4 - n_condensers
        )
        flow_splitter_measurement = FlowSplitterMeasurement(
            name=self.name,
            date_time=date_time,
            flow_fraction_1=flow_fraction_list[0],
            flow_fraction_2=flow_fraction_list[1],
            flow_fraction_3=flow_fraction_list[2],
            flow_fraction_4=flow_fraction_list[3],
        )
        return (
            self.get_state_from_measurement(flow_splitter_measurement, None),
            flow_splitter_measurement,
        )
