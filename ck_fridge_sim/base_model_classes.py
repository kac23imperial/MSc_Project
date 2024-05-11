import json
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel

if TYPE_CHECKING:
    from ck_fridge_sim.equipment_models import Pipe


class BaseBaseModel(BaseModel, ABC):
    name: str
    date_time: datetime


class StateDerivativeBase(BaseBaseModel, ABC):
    pass

    @property
    def state_model_fields(self) -> list[str]:
        return sorted(
            [
                key
                for key in self.model_fields
                if key not in ["name", "date_time", "kind"]
            ]
        )

    @property
    def state_derivative_vector(self) -> list[float]:
        values = [getattr(self, key) for key in self.state_model_fields]
        return [value for value in values if value is not None]


class StateBase(BaseBaseModel, ABC):
    pass

    @property
    def state_model_fields(self) -> list[str]:
        return sorted(
            [
                key
                for key in self.model_fields
                if key not in ["name", "date_time", "kind"]
            ]
        )

    @classmethod
    def state_length(cls) -> int:
        return len(cls.model_fields) - 2

    @property
    def state_vector(self) -> list[float]:
        values = [getattr(self, key) for key in self.state_model_fields]
        return [value for value in values if value is not None]

    @property
    def state_args_vector(self) -> list[str]:
        return [
            key for key in self.state_model_fields if getattr(self, key) is not None
        ]

    def set_from_vector(
        self,
        state_vector: list[float],
        date_time: datetime,
    ) -> list[float]:
        self.date_time = date_time
        for field in self.state_model_fields:
            if getattr(self, field) is None:
                continue

            setattr(self, field, state_vector.pop(0))

        return state_vector

    def __eq__(self, other: "StateBase") -> bool:
        for key in self.state_model_fields:
            current_val = getattr(self, key)
            other_val = getattr(other, key)
            if current_val != other_val:
                return False

        return True

    def __add__(self, other: "StateBase") -> "StateBase":
        state = self.model_copy(deep=True)
        for key in self.state_model_fields:
            current_val = getattr(self, key)
            other_val = getattr(other, key)
            if current_val is None:
                if other_val is not None:
                    raise ValueError(
                        f"current val {current_val} and other val {other_val} are not both None for {key}."
                    )

                setattr(state, key, None)
                continue

            setattr(state, key, current_val + other_val)

        return state

    def __sub__(self, other: "StateBase") -> "StateBase":
        state = self.model_copy(deep=True)
        for key in self.state_model_fields:
            current_val = getattr(self, key)
            other_val = getattr(other, key)
            if current_val is None:
                if other_val is not None:
                    raise ValueError(
                        f"current val {current_val} and other val {other_val} are not both None for {key}."
                    )

                setattr(state, key, None)
                continue

            setattr(state, key, current_val - other_val)

        return state

    def __mul__(self, gain: float) -> "StateBase":
        state = self.model_copy(deep=True)
        for key in self.state_model_fields:
            current_val = getattr(self, key)
            if current_val is None:
                setattr(state, key, None)
                continue

            setattr(state, key, current_val * gain)

        return state

    def advance_state(
        self,
        state_derivative: StateDerivativeBase,
        time_step__s: float,
    ) -> "StateBase":
        state = self.model_copy(deep=True)
        return state + state_derivative * time_step__s


class TimePointBase(BaseBaseModel, ABC):
    @property
    def time_point_model_fields(self) -> list[str]:
        return sorted(
            [
                key
                for key in self.model_fields
                if key not in ["name", "date_time", "kind"]
            ]
        )


class MeasurementBase(BaseBaseModel, ABC):
    @property
    def measurement_model_fields(self) -> list[str]:
        return sorted(
            [
                key
                for key in self.model_fields
                if key not in ["name", "date_time", "kind"]
            ]
        )


class ModelBase(BaseModel, ABC):
    @staticmethod
    def first_order_dynamics(
        target: float | bool, state: float, time_constant__s: float
    ) -> float:
        return (target - state) / time_constant__s

    @staticmethod
    def _mix_inlet_streams(inlets: list["Pipe"]) -> tuple[float]:
        flow__kg_s = sum([inlet.flowrate__kg_s for inlet in inlets]) + 1e-6
        heat_flow__kw = sum(
            [inlet.flowrate__kg_s * inlet.enthalpy__kj_kg for inlet in inlets]
        )
        return flow__kg_s, heat_flow__kw / flow__kg_s

    @abstractmethod
    def get_state_derivative(
        self,
        state: StateBase,
        simulation_point: TimePointBase,
        inlets: list["Pipe"],
        *args,
        **kwargs,
    ) -> StateDerivativeBase:
        pass

    @abstractmethod
    def get_state_from_measurement(
        self,
        measurement: MeasurementBase,
        simulation_point: TimePointBase,
    ) -> StateBase:
        pass

    @abstractmethod
    def initialize(
        self,
        date_time: datetime,
        *args,
    ) -> tuple[StateBase, MeasurementBase, TimePointBase]:
        pass


class PointMap(BaseModel):
    equipment_name: str
    point_name: str
    cp_alias: str
    point_scaling: float = 1.0


class PowerModelBase(BaseModel, ABC):
    name: str

    @property
    def snake_name(self) -> str:
        return self.name.replace(" ", "_").lower()

    @abstractmethod
    def get_power__kw(
        self,
        simulation_point: TimePointBase,
        *args,
        **kwargs,
    ) -> MeasurementBase:
        pass

    @abstractmethod
    def build_time_point(self, date_time: datetime, data: dict) -> TimePointBase:
        pass


class FullProcessStateDerivative(BaseModel):
    equipment_state_derivative_list: list[StateDerivativeBase]

    @property
    def sorted_equipment_state_derivative_list(self) -> list[StateDerivativeBase]:
        return sorted(
            self.equipment_state_derivative_list,
            key=lambda equipment_state_derivative: equipment_state_derivative.name,
        )

    def get_equipment_state_derivative(
        self, equipment_name: str
    ) -> StateDerivativeBase:
        for state_derivative in self.equipment_state_derivative_list:
            if state_derivative.name == equipment_name:
                return state_derivative

        raise ValueError(
            f"Equipment state derivative with name {equipment_name} not found."
        )

    def __add__(
        self, other: "FullProcessStateDerivative"
    ) -> "FullProcessStateDerivative":
        full_state_derivative = self.model_copy(deep=True)
        for state_derivative in full_state_derivative.equipment_state_derivative_list:
            other_state_derivative = other.get_equipment_state_derivative(
                state_derivative.name
            )
            state_derivative += other_state_derivative

        return full_state_derivative

    def __sub__(
        self, other: "FullProcessStateDerivative"
    ) -> "FullProcessStateDerivative":
        full_state_derivative = self.model_copy(deep=True)
        for state_derivative in full_state_derivative.equipment_state_derivative_list:
            other_state_derivative = other.get_equipment_state_derivative(
                state_derivative.name
            )
            state_derivative -= other_state_derivative

        return full_state_derivative

    def __mul__(self, other: float) -> "FullProcessStateDerivative":
        full_state_derivative = self.model_copy(deep=True)
        for state_derivative in full_state_derivative.equipment_state_derivative_list:
            state_derivative *= other

        return full_state_derivative

    @property
    def state_derivative_vector(self) -> list[float]:
        return [
            value
            for state_derivative in self.sorted_equipment_state_derivative_list
            for value in state_derivative.state_derivative_vector
        ]


class ProcessModelEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, datetime):
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)


def dump_json(model: BaseModel | list[BaseModel], json_path: Path) -> None:
    with open(str(json_path), "w") as json_file:
        if isinstance(model, list):
            json.dump(
                [model_item.model_dump() for model_item in model],
                json_file,
                indent=2,
                cls=ProcessModelEncoder,
            )
        else:
            json.dump(model.model_dump(), json_file, indent=2, cls=ProcessModelEncoder)


def load_json(json_path: Path) -> dict:
    with open(str(json_path), "r") as json_file:
        json_dict = json.load(json_file)

    return json_dict


def dump_yaml(model: BaseModel | list[BaseModel], yaml_path: Path) -> None:
    with open(str(yaml_path), "w") as yaml_file:
        if isinstance(model, list):
            yaml.dump(
                [model_item.model_dump() for model_item in model],
                yaml_file,
                default_flow_style=False,
            )
        else:
            yaml.dump(model.model_dump(), yaml_file, default_flow_style=False)


def load_yaml(yaml_path: Path) -> dict:
    with open(str(yaml_path), "r") as yaml_file:
        yaml_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)

    return yaml_dict
