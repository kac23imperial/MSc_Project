import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, Field
from pysolar.solar import get_altitude, get_azimuth


class SourceBase(BaseModel, ABC):
    kind: Literal[
        "constant",
        "step",
        "step_sequence",
        "ramp",
        "sinusoid",
        "sun_angle_altitude",
        "sun_angle_azimuth",
        # "random",
    ]

    @abstractmethod
    def get_value(self, time: datetime) -> float:
        pass


class Constant(SourceBase):
    kind: Literal["constant"] = "constant"
    value: float

    def get_value(self, time: datetime) -> float:
        return self.value


class Step(SourceBase):
    kind: Literal["step"] = "step"
    start_value: float
    end_value: float
    step_time_str: str

    @property
    def step_time(self) -> datetime:
        return datetime.fromisoformat(self.step_time_str)

    def get_value(self, time: datetime) -> float:
        if time < self.step_time:
            return self.start_value
        else:
            return self.end_value


class StepSequence(SourceBase):
    kind: Literal["step_sequence"] = "step_sequence"
    initial_value: float
    values: list[float]
    step_times_str: list[str]

    @property
    def step_times(self) -> list[datetime]:
        return [
            datetime.fromisoformat(step_time_str)
            for step_time_str in self.step_times_str
        ]

    def get_value(self, time: datetime) -> float:
        if len(self.values) != len(self.step_times_str):
            raise ValueError(
                "The number of values and step times must be the same in StepSequence"
            )
        if len(self.values) == 0:
            return self.initial_value

        if time < self.step_times[0]:
            return self.initial_value
        else:
            for i, step_time in enumerate(self.step_times):
                if time < step_time:
                    return self.values[i - 1]
            return self.values[-1]


class Ramp(SourceBase):
    kind: Literal["ramp"] = "ramp"
    start_value: float
    end_value: float
    ramp_start_time_str: str
    ramp_end_time_str: str

    @property
    def ramp_start_time(self) -> datetime:
        return datetime.fromisoformat(self.ramp_start_time_str)

    @property
    def ramp_end_time(self) -> datetime:
        return datetime.fromisoformat(self.ramp_end_time_str)

    def get_value(self, time: datetime) -> float:
        if time < self.ramp_start_time:
            return self.start_value
        elif time > self.ramp_end_time:
            return self.end_value
        else:
            return (
                self.start_value
                + (self.end_value - self.start_value)
                * (time - self.ramp_start_time).total_seconds()
                / (self.ramp_end_time - self.ramp_start_time).total_seconds()
            )


class Sinusoid(SourceBase):
    kind: Literal["sinusoid"] = "sinusoid"
    amplitude: float
    offset: float
    shift_time_str: str
    period__hr: float = 24

    @property
    def shift_time(self) -> datetime:
        return datetime.fromisoformat(self.shift_time_str)

    def get_value(self, time: datetime) -> float:
        return (
            self.amplitude
            * np.sin(
                2
                * np.pi
                * (time - self.shift_time).total_seconds()
                / self.period__hr
                / 3600
            )
            + self.offset
        )


class SunAngleAltitude(SourceBase):
    kind: Literal["sun_angle_altitude"] = "sun_angle_altitude"
    latitude__deg: float
    longitude__deg: float

    def get_value(self, time: datetime) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return (
                get_altitude(self.latitude__deg, self.longitude__deg, time)
                * np.pi
                / 180
            )


class SunAngleAzimuth(SunAngleAltitude):
    kind: Literal["sun_angle_azimuth"] = "sun_angle_azimuth"

    def get_value(self, time: datetime) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return (
                get_azimuth(self.latitude__deg, self.longitude__deg, time) * np.pi / 180
            )

# class RandomSource(SourceBase):
#     kind: Literal['random'] = 'random'
#     low: float
#     high: float

#     def get_value(self, time: datetime):
#         return np.random.uniform(self.low, self.high)

Source = Annotated[
    Constant
    | Step
    | StepSequence
    | Ramp
    | Sinusoid
    | SunAngleAltitude
    | SunAngleAzimuth,
    # | RandomSource,
    Field(discriminator="kind"),
]
