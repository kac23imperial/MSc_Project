from datetime import datetime
from pathlib import Path
from typing import Annotated

from pydantic import Field

from ck_fridge_sim.base_model_classes import (
    BaseModel,
    FullProcessStateDerivative,
    dump_json,
    load_json,
)
from ck_fridge_sim.process_equipment_models.compressor_model import (
    CompressorState,
)
from ck_fridge_sim.process_equipment_models.condenser_model import (
    CondenserState,
)
from ck_fridge_sim.process_equipment_models.evaporator_model import (
    EvaporatorState,
)
from ck_fridge_sim.process_equipment_models.flow_splitter_model import (
    FlowSplitterState,
)
from ck_fridge_sim.process_equipment_models.pump_model import PumpState
from ck_fridge_sim.process_equipment_models.refrigeration_model import RoomState
from ck_fridge_sim.process_equipment_models.valve_model import ValveState
from ck_fridge_sim.process_equipment_models.vessel_model import VesselState

ModelState = Annotated[
    VesselState
    | PumpState
    | ValveState
    | EvaporatorState
    | CondenserState
    | CompressorState
    | FlowSplitterState
    | RoomState,
    Field(discriminator="kind"),
]


class FullProcessState(BaseModel):
    simulation_time: datetime
    equipment_state_list: list[ModelState] = []

    def get_equipment_state(self, name: str) -> ModelState | None:
        for state in self.equipment_state_list:
            if state.name == name:
                return state

        return None

    def set_equipment_state(self, name: str, new_state: ModelState) -> None:
        for i, state in enumerate(self.equipment_state_list):
            if state.name == name:
                self.equipment_state_list[i] = new_state
                return

        self.equipment_state_list.append(new_state)

    def advance_states(
        self,
        state_derivative: FullProcessStateDerivative,
        target_time: datetime,
    ) -> "FullProcessState":
        time_step__s = (target_time - self.simulation_time).total_seconds()
        full_state = self.model_copy(deep=True)
        full_state.simulation_time = target_time
        for equipment_state in full_state.equipment_state_list:
            equipment_state_derivative = (
                state_derivative.get_equipment_state_derivative(equipment_state.name)
            )
            equipment_state = equipment_state.advance_state(
                equipment_state_derivative, time_step__s
            )
            equipment_state.date_time = full_state.simulation_time
            full_state.set_equipment_state(equipment_state.name, equipment_state)

        return full_state

    @property
    def sorted_equipment_state_list(
        self,
    ) -> list[ModelState]:
        return sorted(
            self.equipment_state_list,
            key=lambda equipment_state: equipment_state.name,
        )

    @property
    def state_vector(self) -> list[float]:
        return [
            value
            for equipment_state in self.sorted_equipment_state_list
            for value in equipment_state.state_vector
        ]

    @property
    def state_args_vector(self) -> list[tuple[str, str]]:
        return [
            (equipment_state.name, arg)
            for equipment_state in self.sorted_equipment_state_list
            for arg in equipment_state.state_args_vector
        ]

    def set_from_vector(
        self,
        state_vector: list[float],
        date_time: datetime,
    ) -> list[float]:
        # copy the state vector so we don't modify the original
        state_vector = state_vector.copy()
        for equipment_state in self.sorted_equipment_state_list:
            state_vector = equipment_state.set_from_vector(state_vector, date_time)

        return state_vector

    def dump_to_json(self, file_path: Path) -> None:
        dump_json(self, file_path)

    @classmethod
    def load_from_json(cls, file_path: Path) -> "FullProcessState":
        return cls(**load_json(file_path))
