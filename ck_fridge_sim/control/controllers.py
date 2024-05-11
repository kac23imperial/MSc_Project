from abc import ABC, abstractmethod
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from ck_fridge_sim.base_model_classes import MeasurementBase
from ck_fridge_sim.control.control_base import (
    Actuators,
    ControllerSetting,
    ControlSystemState,
)
from ck_fridge_sim.process_equipment_models.compressor_model import (
    CompressorTimePoint,
)


class MeasurementKey(BaseModel):
    equipment_name: str
    measurement_key: str


class PointKey(BaseModel):
    equipment_name: str
    point_key: str


class ControllerBase(BaseModel, ABC):
    kind: Literal[
        "OnOff", "PI", "OpenLoop", "CompressorPI", "EvaporatorOnOff", "Sequencer"
    ]
    name: str
    measurement_keys: list[MeasurementKey]
    point_keys: list[PointKey]

    def update_setting(self, setting: ControllerSetting) -> None:
        if setting.nested_setting_names is not None:
            # Access the first attribute based on setting_name
            nested_setting = getattr(self, setting.setting_name)

            # Traverse all but the last item in nested_setting_names to reach the target
            for nested_setting_name in setting.nested_setting_names[:-1]:
                if isinstance(nested_setting_name, int):
                    nested_setting = nested_setting[nested_setting_name]
                else:
                    nested_setting = getattr(nested_setting, nested_setting_name)

            # Use the last item in nested_setting_names to set the value
            last_setting_name = setting.nested_setting_names[-1]
            if isinstance(last_setting_name, int):
                nested_setting[last_setting_name] = setting.value
            else:
                setattr(nested_setting, last_setting_name, setting.value)
        else:
            # Directly set the attribute if there are no nested settings
            setattr(self, setting.setting_name, setting.value)

    def measurements_dict(
        self, measurements: list[MeasurementBase]
    ) -> dict[str, MeasurementBase]:
        return {
            m.name: m
            for m in measurements
            if m.name in [mk.equipment_name for mk in self.measurement_keys]
        }

    @staticmethod
    def _error(measured: float, setpoint: float) -> float:
        return measured - setpoint

    def get_measurement(
        self, measurements: list[MeasurementBase], measurement_key: MeasurementKey
    ) -> float:
        measurement_dict = self.measurements_dict(measurements)
        equipment_measurement = measurement_dict.get(measurement_key.equipment_name)
        if equipment_measurement is not None:
            return getattr(equipment_measurement, measurement_key.measurement_key)

        raise ValueError(f"Measurement {measurement_key} not found")

    def get_measurements(self, measurements: list[MeasurementBase]) -> list[float]:
        measurement_values = []
        for measurement_key in self.measurement_keys:
            measurement_values.append(
                self.get_measurement(measurements, measurement_key)
            )

        return measurement_values

    def _validate_single_measurement(
        self, measurements: list[MeasurementBase]
    ) -> float:
        measurement_list = self.get_measurements(measurements)
        if len(measurement_list) != 1:
            raise ValueError(
                f"{self.name} controller requires exactly one measurement, got {len(measurement_list)}"
            )
        return measurement_list[0]

    def set_actuator_values(
        self,
        actuators: Actuators,
        values: list[float | bool],
        time: datetime,
    ) -> None:
        if len(values) != len(self.point_keys):
            raise ValueError(
                f"Number of values ({len(values)}) does not match number of point keys ({len(self.point_keys)}) for {self.name} controller"
            )

        for point_key, value in zip(self.point_keys, values):
            actuators.set_actuator(
                point_key.equipment_name, point_key.point_key, value, time
            )

    def set_actuator_value(
        self,
        actuators: Actuators,
        point_key_idx: int,
        value: float | bool,
        time: datetime,
    ) -> None:
        point_key = self.point_keys[point_key_idx]
        actuators.set_actuator(
            point_key.equipment_name, point_key.point_key, value, time
        )

    def set_all_actuators_to_value(
        self, actuators: Actuators, value: float | bool, time: datetime
    ) -> None:
        for point_key in self.point_keys:
            actuators.set_actuator(
                point_key.equipment_name, point_key.point_key, value, time
            )

    @abstractmethod
    def action(
        self,
        control_states: "ControlSystemState",
        measurements: list[MeasurementBase],
        actuators: Actuators,
        time: datetime,
    ) -> None:
        pass


class OnOff(ControllerBase):
    """
    A simple on/off controller.
    """

    kind: Literal["OnOff"] = "OnOff"
    min_output: float
    max_output: float
    process_gain: float | int = -1  # -1 for cooling, 1 for heating
    deadband_low: float
    deadband_high: float
    setpoint: float
    last_output_key: str = "last_output"

    def get_output(self, last_output: float | None, error: float) -> float:
        if error > self.deadband_high:
            output = self.min_output if self.process_gain > 0 else self.max_output
        elif error < -self.deadband_low:
            output = self.max_output if self.process_gain > 0 else self.min_output
        elif last_output is None:
            output = self.min_output if self.process_gain > 0 else self.max_output
        else:
            output = last_output

        return output

    def action(
        self,
        control_state: ControlSystemState,
        measurements: list[MeasurementBase],
        actuators: Actuators,
        time: datetime,
    ) -> None:
        last_output = control_state.get_control_state(self.last_output_key, self.name)
        measurement = self._validate_single_measurement(measurements)
        error = self._error(measurement, self.setpoint)
        output = self.get_output(last_output, error)
        control_state.set_control_state(
            name=self.last_output_key,
            controller_name=self.name,
            target_value=output,
            time=time,
        )
        self.set_all_actuators_to_value(actuators, output, time)


class PI(ControllerBase):
    kind: Literal["PI"] = "PI"
    min_output: float
    max_output: float
    kp: float
    tau_i: float
    setpoint: float
    _ki: float | None = None
    integral_error_key: str = "integral_error"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ki = 0 if self.tau_i == 0 else self.kp / self.tau_i

    def _control_output(self, error: float, error_int: float) -> float:
        return self.kp * error + self._ki * error_int

    def _limit_integrator(self, error: float, error_int: float) -> float:
        if (
            self._control_output(error, error_int) > self.max_output
            and self.kp * error > 0
        ):
            return 0

        if (
            self._control_output(error, error_int) < self.min_output
            and self.kp * error < 0
        ):
            return 0

        return error

    def _add_error_to_state(
        self, error: float, control_state: ControlSystemState, time: datetime
    ) -> None:
        integral_error = control_state.get_control_state(
            self.integral_error_key, self.name
        )

        control_state.set_control_state(
            name=self.integral_error_key,
            controller_name=self.name,
            target_value=None,
            time=time,
            derivative=self._limit_integrator(
                error, 0 if integral_error is None else integral_error
            ),
        )

    def _apply_saturation(self, u: float) -> float:
        return max(min(u, self.max_output), self.min_output)

    def _get_output(
        self,
        control_state: ControlSystemState,
        measurement: float,
        setpoint: float,
        time: datetime,
    ) -> float:
        error = self._error(measurement, setpoint)
        self._add_error_to_state(error, control_state, time)
        integral_error = (
            control_state.get_control_state(self.integral_error_key, self.name) or 0
        )
        action = self._control_output(error, integral_error)
        return self._apply_saturation(action)

    def action(
        self,
        control_state: ControlSystemState,
        measurements: list[MeasurementBase],
        actuators: Actuators,
        time: datetime,
    ) -> None:
        measurement = self._validate_single_measurement(measurements)
        output = self._get_output(control_state, measurement, self.setpoint, time)
        self.set_all_actuators_to_value(actuators, output, time)


class OpenLoop(ControllerBase):
    kind: Literal["OpenLoop"] = "OpenLoop"
    values: list[float]

    def action(
        self,
        control_state: ControlSystemState,
        measurements: list[MeasurementBase],
        actuators: Actuators,
        time: datetime,
    ) -> None:
        self.set_actuator_values(actuators, self.values, time)


class CompressorPI(PI):
    kind: Literal["CompressorPI"] = "CompressorPI"
    min_rpm: float

    def action(
        self,
        control_state: ControlSystemState,
        measurements: list[MeasurementBase],
        actuators: Actuators,
        time: datetime,
    ) -> None:
        measurement = self._validate_single_measurement(measurements)
        output = self._get_output(control_state, measurement, self.setpoint, time)
        rpm, sv = CompressorTimePoint.rpm_and_sv_from_capacity(output, self.min_rpm)
        self.set_actuator_values(actuators, [rpm, sv], time)


class EvaporatorOnOff(OnOff):
    kind: Literal["EvaporatorOnOff"] = "EvaporatorOnOff"

    def action(
        self,
        control_state: ControlSystemState,
        measurements: list[MeasurementBase],
        actuators: Actuators,
        time: datetime,
    ) -> None:
        last_output = control_state.get_control_state(self.last_output_key, self.name)
        measurement = self._validate_single_measurement(measurements)
        error = self._error(measurement, self.setpoint)
        output = self.get_output(last_output, error)
        control_state.set_control_state(
            name=self.last_output_key,
            controller_name=self.name,
            target_value=output,
            time=time,
        )
        if output == self.max_output:
            solenoid_on, fan_speed = True, 3600.0
        else:
            solenoid_on, fan_speed = False, 0.0

        self.set_actuator_values(
            actuators, [solenoid_on] + [fan_speed] * (len(self.point_keys) - 1), time
        )


class SequencerStage(BaseModel):
    idx: int
    equipment_on_list: list[bool | float]
    trim_idx: list[int] | None = None


class Sequencer(ControllerBase, ABC):
    stages: list[SequencerStage]
    deadband_low: float
    deadband_high: float
    initial_stage_idx: int = 0
    stage_up_timer__s: int
    stage_down_timer__s: int
    pi_controller: PI
    stage_up_timer_key: str = "stage_up_timer"
    stage_down_timer_key: str = "stage_down_timer"
    active_stage_key: str = "active_stage"

    def _get_stage_by_idx(self, idx: int) -> SequencerStage:
        return next(stage for stage in self.stages if stage.idx == idx)

    def _get_states(
        self, control_state: ControlSystemState
    ) -> tuple[float, float, int]:
        stage_up_timer = control_state.get_control_state(
            self.stage_up_timer_key, self.name
        )
        stage_up_timer = 0 if stage_up_timer is None else stage_up_timer

        stage_down_timer = control_state.get_control_state(
            self.stage_down_timer_key, self.name
        )
        stage_down_timer = 0 if stage_down_timer is None else stage_down_timer

        current_stage_idx = control_state.get_control_state(
            self.active_stage_key, self.name
        )
        current_stage_idx = (
            self.initial_stage_idx if current_stage_idx is None else current_stage_idx
        )
        return stage_up_timer, stage_down_timer, current_stage_idx

    def _update_states(
        self,
        control_state: ControlSystemState,
        time: datetime,
        measurement: float,
        current_stage_idx: int,
    ) -> None:
        if (
            measurement > self.pi_controller.setpoint + self.deadband_high
            and current_stage_idx < len(self.stages) - 1
        ):
            control_state.set_control_state(
                name=self.stage_up_timer_key,
                controller_name=self.name,
                target_value=None,
                time=time,
                derivative=1,
            )
            control_state.set_control_state(
                name=self.stage_down_timer_key,
                controller_name=self.name,
                target_value=0,
                time=time,
                derivative=0,
            )

        elif (
            measurement < self.pi_controller.setpoint - self.deadband_low
            and current_stage_idx > 0
        ):
            control_state.set_control_state(
                name=self.stage_up_timer_key,
                controller_name=self.name,
                target_value=0,
                time=time,
            )
            control_state.set_control_state(
                name=self.stage_down_timer_key,
                controller_name=self.name,
                target_value=None,
                time=time,
                derivative=1,
            )

        else:
            control_state.set_control_state(
                name=self.stage_up_timer_key,
                controller_name=self.name,
                target_value=0,
                time=time,
            )
            control_state.set_control_state(
                name=self.stage_down_timer_key,
                controller_name=self.name,
                target_value=0,
                time=time,
            )

    def _change_sequencer_stage(
        self,
        control_state: ControlSystemState,
        stage_up_timer: float,
        stage_down_timer: float,
        current_stage_idx: int,
        time: datetime,
    ) -> bool:
        staged_up = False
        if (
            stage_up_timer > self.stage_up_timer__s
            and current_stage_idx < len(self.stages) - 1
        ):
            stage = current_stage_idx + 1
            control_state.set_control_state(
                name=self.active_stage_key,
                controller_name=self.name,
                target_value=stage,
                time=time,
            )
            control_state.set_control_state(
                name=self.stage_up_timer_key,
                controller_name=self.name,
                target_value=0,
                time=time,
            )
            staged_up = stage
        elif stage_down_timer > self.stage_down_timer__s and current_stage_idx > 0:
            stage = current_stage_idx - 1
            control_state.set_control_state(
                name=self.active_stage_key,
                controller_name=self.name,
                target_value=stage,
                time=time,
            )
            control_state.set_control_state(
                name=self.stage_down_timer_key,
                controller_name=self.name,
                target_value=0,
                time=time,
            )
        else:
            stage = current_stage_idx
            control_state.set_control_state(
                name=self.active_stage_key,
                controller_name=self.name,
                target_value=None,
                time=time,
            )

        return staged_up


class CompressorSequencer(Sequencer):
    kind: Literal["CompressorSequencer"] = "CompressorSequencer"
    pi_controller: CompressorPI
    min_rpm_dict: dict[int, float]
    suction_shutdown_pressure__psig: float = -3

    def action(
        self,
        control_state: ControlSystemState,
        measurements: list[MeasurementBase],
        actuators: Actuators,
        time: datetime,
    ) -> None:
        stage_up_timer, stage_down_timer, current_stage_idx = self._get_states(
            control_state
        )

        # update control state
        pressure = self._validate_single_measurement(measurements)
        self._update_states(control_state, time, pressure, current_stage_idx)

        # update stage
        staged_up = self._change_sequencer_stage(
            control_state, stage_up_timer, stage_down_timer, current_stage_idx, time
        )

        # apply action
        # set rpm to 0 for all off stages and 3600 for all on stages
        stage = self._get_stage_by_idx(current_stage_idx)
        if pressure < self.suction_shutdown_pressure__psig:
            stage = self._get_stage_by_idx(0)

        actuator_values = []
        for equipment_on in stage.equipment_on_list:
            if not equipment_on:
                actuator_values.extend([0, 0])
            else:
                rpm, sv = CompressorTimePoint.rpm_and_sv_from_capacity(
                    100, self.min_rpm_dict[current_stage_idx]
                )
                actuator_values.extend([rpm, sv])

        self.set_actuator_values(actuators, actuator_values, time)
        # set pi controller to the active stage's trim index
        if stage.trim_idx is not None:
            self.pi_controller.point_keys = [
                self.point_keys[idx] for idx in stage.trim_idx
            ]
            self.pi_controller.min_rpm = self.min_rpm_dict[stage.idx]
            if staged_up:
                control_state.set_control_state(
                    name=self.pi_controller.integral_error_key,
                    controller_name=self.pi_controller.name,
                    target_value=0,
                    time=time,
                )

            self.pi_controller.action(control_state, measurements, actuators, time)


class CondenserSequencer(Sequencer):
    kind: Literal["CondenserSequencer"] = "CondenserSequencer"
    pi_controller: PI
    shutdown_pressure__psig: float = 100

    def action(
        self,
        control_state: ControlSystemState,
        measurements: list[MeasurementBase],
        actuators: Actuators,
        time: datetime,
    ) -> None:
        stage_up_timer, stage_down_timer, current_stage_idx = self._get_states(
            control_state
        )

        # update control state
        pressure = self._validate_single_measurement(measurements)
        self._update_states(control_state, time, pressure, current_stage_idx)

        # update stage
        _ = self._change_sequencer_stage(
            control_state, stage_up_timer, stage_down_timer, current_stage_idx, time
        )

        # apply action
        stage = self._get_stage_by_idx(current_stage_idx)
        if pressure < self.shutdown_pressure__psig:
            stage = self._get_stage_by_idx(0)

        actuator_values = []
        for equipment_on in stage.equipment_on_list:
            actuator_values.append(equipment_on)

        self.set_actuator_values(actuators, actuator_values, time)
        # set pi controller to the active stage's trim index
        if stage.trim_idx is not None:
            self.pi_controller.point_keys = [
                self.point_keys[idx] for idx in stage.trim_idx
            ]
            self.pi_controller.action(control_state, measurements, actuators, time)


class LocalCompressorController(ControllerBase):
    kind: Literal["LocalCompressorController"] = "LocalCompressorController"
    max_discharge_pressure__psig: float
    unload_rate__perc_per_s: float
    sv_unload_key: str = "sv_unload"

    def action(
        self,
        control_state: ControlSystemState,
        measurements: list[MeasurementBase],
        actuators: Actuators,
        time: datetime,
    ) -> None:
        (
            discharge_pressure,
            sv_position__perc,
            compressor_speed__rpm,
        ) = self.get_measurements(measurements)
        if discharge_pressure < self.max_discharge_pressure__psig:
            control_state.remove_control_state(self.sv_unload_key, self.name)
            return

        if compressor_speed__rpm <= 0:
            return

        if sv_position__perc <= 0:
            return

        sv_unload = control_state.get_control_state(self.sv_unload_key, self.name)
        if sv_unload is None:
            sv_unload = sv_position__perc
        else:
            self.set_actuator_value(actuators, 0, sv_unload, time)

        if sv_unload <= 10:
            control_state.remove_control_state(self.sv_unload_key, self.name)
            return

        control_state.set_control_state(
            name=self.sv_unload_key,
            controller_name=self.name,
            target_value=None,
            time=time,
            derivative=-self.unload_rate__perc_per_s,
        )


Controller = Annotated[
    OnOff
    | PI
    | OpenLoop
    | CompressorPI
    | EvaporatorOnOff
    | CompressorSequencer
    | CondenserSequencer
    | LocalCompressorController,
    Field(discriminator="kind"),
]
