from datetime import datetime
from pathlib import Path

from ck_fridge_sim.base_model_classes import load_json, load_yaml
from ck_fridge_sim.control.control_base import (
    ControllerSetting,
    ControlSystemState,
    Disturbances,
)
from ck_fridge_sim.control.control_system import ControlSystem
from ck_fridge_sim.process_model import ProcessModel
from ck_fridge_sim.solver.dynamic_simulation import DynamicSimulation
from ck_fridge_sim.solver.solvers import RK12, FixedStepEuler


def configure_example_simulation_from_path(
    process_config_path: Path,
    control_config_path: Path,
    disturbances: Disturbances,
    settings: list[ControllerSetting],
    start_datetime: datetime,
    end_datetime: datetime,
    solver: FixedStepEuler | RK12,
) -> DynamicSimulation:
    process_config = load_json(process_config_path)
    process = ProcessModel(**process_config)
    control_states = ControlSystemState(control_states=[])
    full_process_state = process.configure_actuators_and_measurements(start_datetime)
    control_config = load_yaml(control_config_path)
    control_system = ControlSystem(**control_config)
    return DynamicSimulation(
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        process_model=process,
        control_system=control_system,
        process_states=full_process_state,
        control_states=control_states,
        disturbances=disturbances,
        settings=settings,
        solver=solver,
    )
