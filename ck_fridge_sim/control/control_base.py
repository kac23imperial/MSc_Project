from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from ck_fridge_sim.base_model_classes import dump_json, load_json
from ck_fridge_sim.equipment_models import TimePoint
from ck_fridge_sim.sources.sources import Source


class Actuators(BaseModel):
    actuator_list: list[TimePoint]

    def get_actuator_by_name(self, name: str) -> TimePoint | None:
        for actuator in self.actuator_list:
            if actuator.name == name:
                return actuator

        return None

    def set_actuator(self, name: str, key: str, value: float, time: datetime) -> None:
        for actuator in self.actuator_list:
            if actuator.name == name:
                setattr(actuator, key, value)
                actuator.date_time = time
                return

        raise ValueError(f"Actuator point with name {name} and key {key} not found.")


class Disturbance(BaseModel):
    name: str
    source: Source

    def value(self, time: datetime) -> float:
        return self.source.get_value(time)


class Disturbances(BaseModel):
    disturbance_list: list[Disturbance]

    def get_disturbance_by_name(self, name: str) -> Disturbance:
        for disturbance in self.disturbance_list:
            if disturbance.name == name:
                return disturbance

        raise ValueError(f"Disturbance with name {name} not found.")


class ControlState(BaseModel):
    name: str
    controller_name: str
    date_time: datetime
    value: Any
    target_value: Any | None = None
    derivative: float | None = None

    def update(self, time_step_length__s: float) -> None:
        self.date_time += timedelta(seconds=time_step_length__s)
        if self.target_value is not None:
            self.value = self.target_value

        if self.derivative is not None:
            self.value = 0.0 if self.value is None else self.value
            self.value += self.derivative * time_step_length__s


ControllerSettingType = TypeVar("ControllerSettingType", bound="ControllerSetting")


class ControllerSetting(BaseModel):
    name: str
    controller_name: str
    setting_name: str
    nested_setting_names: list[str | int] | None = None
    value: str | float | bool
    source: Source | None = None

    def update_value(self, time: datetime) -> None:
        if self.source is None:
            return

        self.value = self.source.get_value(time)


class Setpoint(BaseModel):
    name: str
    equipment_name: str
    measurement_key: str
    value: float


class ControlSystemState(BaseModel):
    control_states: list[ControlState]

    def update(self, step_length__s: float) -> None:
        for state in self.control_states:
            state.update(step_length__s)

    def get_control_state(self, name: str, controller_name: str) -> Any | None:
        for state in self.control_states:
            if state.name == name and state.controller_name == controller_name:
                return state.value

        return None

    def set_control_state(
        self,
        name: str,
        controller_name: str,
        target_value: Any,
        time: datetime,
        derivative: float | None = None,
    ) -> None:
        for state in self.control_states:
            if state.name == name and state.controller_name == controller_name:
                state.target_value = target_value
                state.date_time = time
                state.derivative = derivative
                return

        self.control_states.append(
            ControlState(
                name=name,
                controller_name=controller_name,
                date_time=time,
                value=target_value,
                derivative=derivative,
            )
        )

    def remove_control_state(self, name: str, controller_name: str) -> None:
        self.control_states = [
            state
            for state in self.control_states
            if not (state.name == name and state.controller_name == controller_name)
        ]

    def dump_to_json(self, file_path: Path) -> None:
        dump_json(self, file_path)

    @classmethod
    def load_from_json(cls, file_path: Path) -> "ControlSystemState":
        return cls(**load_json(file_path))
