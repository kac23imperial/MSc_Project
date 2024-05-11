from ck_fridge_sim.control.control_system import ControlSystem
from ck_fridge_sim.control.controllers import (
    PI,
    CompressorPI,
    CompressorSequencer,
    CondenserSequencer,
    LocalCompressorController,
    MeasurementKey,
    OnOff,
    OpenLoop,
    PointKey,
    SequencerStage,
)
from ck_fridge_sim.process_model import ProcessModel


def get_vessel_name(process: ProcessModel, suction_side: str) -> str:
    vessel_identifiers = (
        ("ltr", "ltpr", "lpr", "lta")
        if "low" in suction_side
        else ("htr", "htpr", "mtpr", "hta")
    )
    vessels = [
        vessel
        for vessel in process.vessel_list
        if any(identifier in vessel.name for identifier in vessel_identifiers)
    ]
    if len(vessels) != 1:
        raise ValueError(f"Expected one vessel with identifier {vessel_identifiers}")

    return vessels[0].name


def build_compressor_sequencer(
    process: ProcessModel, vessel_name: str, sequence_order: list[str] | None = None
) -> CompressorSequencer:
    point_keys = []
    if sequence_order is None:
        compressor_list = sorted(
            process.compressor_list.copy(), key=lambda c: c.model.motor_size__hp
        )
    else:
        compressor_list = [
            c for c in process.compressor_list.copy() if c.name in sequence_order
        ]
        compressor_list.sort(key=lambda c: sequence_order.index(c.name))

    for compressor in compressor_list:
        point_keys.append(
            PointKey(equipment_name=compressor.name, point_key="compressor_speed__rpm")
        )
        point_keys.append(
            PointKey(equipment_name=compressor.name, point_key="sv_position__perc")
        )
    stages = []
    for i in range(len(compressor_list) + 1):
        equipment_on_list = [True] * i + [False] * (len(compressor_list) - i)
        trim_idx = [2 * (i - 1), 2 * (i - 1) + 1] if i > 0 else None
        stages.append(
            SequencerStage(
                idx=i, equipment_on_list=equipment_on_list, trim_idx=trim_idx
            )
        )

    pressure_measurement = MeasurementKey(
        equipment_name=vessel_name,
        measurement_key="pressure__psig",
    )
    compressor_sequencer = CompressorSequencer(
        name="compressor_sequencer",
        measurement_keys=[pressure_measurement],
        point_keys=point_keys,
        stages=stages,
        pi_controller=CompressorPI(
            name="compressor_pi",
            measurement_keys=[pressure_measurement],
            point_keys=[],
            min_output=0,
            max_output=100,
            setpoint=5,
            kp=5,
            tau_i=300,
            min_rpm=3600,
        ),
        min_rpm_dict={i + 1: c.model.min_rpm for i, c in enumerate(compressor_list)},
        deadband_low=2,
        deadband_high=2,
        initial_stage_idx=1,
        stage_up_timer__s=600,
        stage_down_timer__s=300,
    )
    return compressor_sequencer


def configure_vessel_level_control(vessel_name: str) -> OnOff:
    return OnOff(
        name="level_control",
        measurement_keys=[
            MeasurementKey(
                equipment_name=vessel_name,
                measurement_key="level__perc",
            ),
        ],
        point_keys=[
            PointKey(
                equipment_name="feed_valve",
                point_key="valve_position__perc",
            )
        ],
        process_gain=1.0,
        setpoint=33.0,
        min_output=0.0,
        max_output=100.0,
        deadband_low=3.0,
        deadband_high=3.0,
    )


def configure_suction_control_system(
    process: ProcessModel,
    suction_side: str,
    sequence_order: list[str] | None = None,
) -> ControlSystem:
    vessel_name = get_vessel_name(process, suction_side)
    compressor_sequencer = build_compressor_sequencer(
        process, vessel_name, sequence_order
    )
    level_control = configure_vessel_level_control(vessel_name)
    return ControlSystem(controllers=[compressor_sequencer, level_control])


def make_condenser_sequencer_stages(
    process: ProcessModel,
    n_fans: dict[str, dict[str, int]],
    n_pumps: dict[str, dict[str, int]],
    max_fan_speed: float,
):
    total_fans = sum([n["length"] for n in n_fans.values()])
    total_pumps = sum([n["length"] for n in n_pumps.values()])
    stages = []
    stages.append(
        SequencerStage(
            idx=0, equipment_on_list=[0.0] * total_fans + [False] * total_pumps
        )
    )
    # first turn on all the pumps
    fan_on_list = [0.0] * total_fans
    pump_on_list = [False] * total_pumps
    stage_idx = 1
    for pump_idx in range(max([n["length"] for n in n_pumps.values()])):
        for condenser in process.condenser_list:
            if pump_idx >= n_pumps[condenser.name]["length"]:
                continue

            start_idx = n_pumps[condenser.name]["start_idx"]
            pump_on_list[start_idx + pump_idx] = True
            stages.append(
                SequencerStage(
                    idx=stage_idx,
                    equipment_on_list=fan_on_list + pump_on_list.copy(),
                )
            )
            stage_idx += 1

    # then turn on the fans
    for fan_idx in range(max([n["length"] for n in n_fans.values()])):
        for condenser in process.condenser_list:
            if fan_idx >= n_fans[condenser.name]["length"]:
                continue

            start_idx = n_fans[condenser.name]["start_idx"]
            fan_on_list[start_idx + fan_idx] = max_fan_speed
            # trim all fans that are on
            stages.append(
                SequencerStage(
                    idx=stage_idx,
                    equipment_on_list=fan_on_list.copy() + pump_on_list,
                    trim_idx=[idx for idx, val in enumerate(fan_on_list) if val > 0],
                )
            )
            stage_idx += 1

    return stages


def make_condenser_sequencer(
    process: ProcessModel, min_fan_speed: float = 900, max_fan_speed: float = 3600
) -> CondenserSequencer:
    n_fans, n_pumps = dict(), dict()
    fan_start_idx, pump_start_idx = 0, 0
    for condenser in process.condenser_list:
        n_fans[condenser.name] = {
            "length": len(condenser.model.fan_motors__hp),
            "start_idx": fan_start_idx,
        }
        n_pumps[condenser.name] = {
            "length": len(condenser.model.pump_motors__hp),
            "start_idx": pump_start_idx,
        }
        fan_start_idx += n_fans[condenser.name]["length"]
        pump_start_idx += n_pumps[condenser.name]["length"]

    fan_point_keys = []
    for condenser in process.condenser_list:
        for i in range(1, n_fans[condenser.name]["length"] + 1):
            fan_point_keys.append(
                PointKey(equipment_name=condenser.name, point_key=f"fan_speed_{i}__rpm")
            )
    pump_point_keys = []
    for condenser in process.condenser_list:
        for i in range(1, n_pumps[condenser.name]["length"] + 1):
            pump_point_keys.append(
                PointKey(equipment_name=condenser.name, point_key=f"pump_{i}_on")
            )

    condenser_sequencer = CondenserSequencer(
        name="condenser_sequencer",
        measurement_keys=[
            MeasurementKey(
                equipment_name=process.condenser_list[0].name,
                measurement_key="pressure__psig",
            )
        ],
        point_keys=fan_point_keys + pump_point_keys,
        stages=make_condenser_sequencer_stages(process, n_fans, n_pumps, max_fan_speed),
        deadband_low=5.0,
        deadband_high=5.0,
        initial_stage_idx=1,
        stage_down_timer__s=300,
        stage_up_timer__s=300,
        pi_controller=PI(
            name="condenser_pi",
            measurement_keys=[
                MeasurementKey(
                    equipment_name=process.condenser_list[0].name,
                    measurement_key="pressure__psig",
                )
            ],
            setpoint=125.0,
            point_keys=[],
            min_output=min_fan_speed,
            max_output=max_fan_speed,
            kp=50.0,
            tau_i=30.0,
        ),
    )
    return condenser_sequencer


def configure_local_compressor_controller(
    process: ProcessModel, max_discharge_pressure__psig: float
) -> list[LocalCompressorController]:
    return [
        LocalCompressorController(
            name=compressor.name + "_local",
            max_discharge_pressure__psig=max_discharge_pressure__psig,
            measurement_keys=[
                MeasurementKey(
                    equipment_name=process.condenser_list[0].name,
                    measurement_key="pressure__psig",
                ),
                MeasurementKey(
                    equipment_name=compressor.name,
                    measurement_key="sv_position__perc",
                ),
                MeasurementKey(
                    equipment_name=compressor.name,
                    measurement_key="compressor_speed__rpm",
                ),
            ],
            point_keys=[
                PointKey(equipment_name=compressor.name, point_key="sv_position__perc"),
            ],
            unload_rate__perc_per_s=0.5,
        )
        for compressor in process.compressor_list
    ]


def configure_condenser_control_system(
    process: ProcessModel,
    min_fan_speed: float = 900,
    max_fan_speed: float = 3600,
    max_discharge_pressure__psig: float = 190.0,
) -> ControlSystem:
    condenser_sequencer = make_condenser_sequencer(
        process, min_fan_speed, max_fan_speed
    )

    compressor_controllers = [
        OpenLoop(
            name=compressor.name,
            measurement_keys=[],
            point_keys=[
                PointKey(
                    equipment_name=compressor.name, point_key="compressor_speed__rpm"
                ),
                PointKey(equipment_name=compressor.name, point_key="sv_position__perc"),
            ],
            values=[3600.0, 100.0],
        )
        for compressor in process.compressor_list
    ]
    local_compressor_controllers = configure_local_compressor_controller(
        process, max_discharge_pressure__psig
    )

    return ControlSystem(
        controllers=[condenser_sequencer]
        + compressor_controllers
        + local_compressor_controllers
    )
