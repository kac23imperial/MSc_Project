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
    process_config = load_json(process_config_path) #1
    process = ProcessModel(**process_config) #1
    control_states = ControlSystemState(control_states=[]) #2
    full_process_state = process.configure_actuators_and_measurements(start_datetime) #1
    control_config = load_yaml(control_config_path) #2
    control_system = ControlSystem(**control_config) #2
    return DynamicSimulation(
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        process_model=process, #yes
        control_system=control_system, #yes
        process_states=full_process_state, #yes
        control_states=control_states, #yes
        disturbances=disturbances,
        settings=settings,
        solver=solver,
    )
