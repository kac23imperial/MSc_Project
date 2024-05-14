from datetime import datetime
from typing import Literal

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pydantic import BaseModel

from ck_fridge_sim.base_model_classes import (
    FullProcessStateDerivative,
    MeasurementBase,
)
from ck_fridge_sim.control.control_base import Actuators, Disturbances
from ck_fridge_sim.equipment_models import (
    Compressor,
    Condenser,
    Equipment,
    Evaporator,
    FlowSplitter,
    Measurement,
    Pipe,
    Pump,
    Room,
    TimePoint,
    Valve,
    Vessel,
)
from ck_fridge_sim.process_equipment_models.compressor_model import (
    CompressionReturn,
)
from ck_fridge_sim.process_equipment_models.condenser_model import (
    CondenserMeasurement,
    CondenserTimePoint,
)
from ck_fridge_sim.process_equipment_models.evaporator_model import (
    EvaporatorMeasurement,
    EvaporatorState,
)
from ck_fridge_sim.process_equipment_models.flow_splitter_model import (
    FlowSplitterMeasurement,
    FlowSplitterModel,
    FlowSplitterState,
)
from ck_fridge_sim.process_equipment_models.pump_model import (
    PumpMeasurement,
)
from ck_fridge_sim.process_equipment_models.refrigeration_model import (
    RoomMeasurement,
)
from ck_fridge_sim.process_equipment_models.valve_model import (
    ValveMeasurement,
)
from ck_fridge_sim.process_equipment_models.vessel_model import (
    VesselMeasurement,
    VesselState,
)
from ck_fridge_sim.process_states import FullProcessState
from ck_fridge_sim.thermo_tools.ammonia_thermo import AmmoniaThermo
from ck_fridge_sim.thermo_tools.unit_conversions import UnitConversions


class BuildingMeasurement(MeasurementBase):
    total_refrigeration_power__kw: float
    machine_room_temperature__f: float


class ProcessModel(BaseModel):
    name: str
    kind: Literal["process_with_rooms"] = "process_with_rooms"
    equipment_list: list[Equipment]
    pipe_list: list[Pipe] = []
    refrigerated_rooms: list[Room] | None = None
    building_measurement: BuildingMeasurement | None = None

    @property
    def sorted_equipment_list(
        self,
    ) -> list[Equipment]:
        return sorted(self.equipment_list, key=lambda x: x.name)

    def get_equipment_by_name(self, name: str) -> Equipment | None:
        for equipment in self.equipment_list:
            if equipment.name == name:
                return equipment

        return None

    def get_device_by_name(self, device_name: str) -> Equipment | None:
        return self.get_equipment_by_name(device_name)

    def get_room_by_name(self, name: str) -> Room:
        for room in self.refrigerated_rooms:
            if room.name == name:
                return room

        raise ValueError(f"Room with name {name} not found.")

    def get_upstream_pipes(self, equipment_name: str) -> list[Pipe]:
        return [pipe for pipe in self.pipe_list if pipe.target == equipment_name]

    def get_downstream_pipes(self, equipment_name: str) -> list[Pipe]:
        return [pipe for pipe in self.pipe_list if pipe.source == equipment_name]

    def get_upstream_pipe(self, equipment_name: str) -> Pipe:
        upstream_pipes = self.get_upstream_pipes(equipment_name)
        if len(upstream_pipes) != 1:
            raise ValueError(f"More than one upstream pipe found on {equipment_name}")
        return upstream_pipes[0]

    def get_downstream_pipe(self, equipment_name: str) -> Pipe:
        downstream_pipes = self.get_downstream_pipes(equipment_name)
        if len(downstream_pipes) != 1:
            raise ValueError(f"More than one downstream pipe found on {equipment_name}")
        return downstream_pipes[0]

    @property
    def compressor_list(self) -> list[Compressor]:
        return [
            equipment
            for equipment in self.sorted_equipment_list
            if equipment.kind == "compressor"
        ]

    @property
    def condenser_list(self) -> list[Condenser]:
        return [
            equipment
            for equipment in self.sorted_equipment_list
            if equipment.kind == "condenser"
        ]

    @property
    def evaporator_list(self) -> list[Evaporator]:
        return [
            equipment
            for equipment in self.sorted_equipment_list
            if equipment.kind == "evaporator"
        ]

    @property
    def vessel_list(self) -> list[Vessel]:
        return [
            equipment
            for equipment in self.sorted_equipment_list
            if equipment.kind == "vessel"
        ]

    @property
    def pump_list(self) -> list[Pump]:
        return [
            equipment
            for equipment in self.sorted_equipment_list
            if equipment.kind == "pump"
        ]

    @property
    def valve_list(self) -> list[Valve]:
        return [
            equipment
            for equipment in self.sorted_equipment_list
            if equipment.kind == "valve" or equipment.kind == "local_control_valve"
        ]

    @staticmethod
    def _assert_condenser_sources(process_model: "ProcessModel") -> None:
        source_list = [
            pipe.source
            for pipe in process_model.get_upstream_pipes(
                process_model.condenser_list[0].name
            )
        ]
        for condenser in process_model.condenser_list:
            upstream_sources = [
                pipe.source for pipe in process_model.get_upstream_pipes(condenser.name)
            ]
            if sorted(source_list) != sorted(upstream_sources):
                raise ValueError(
                    f"Upstream sources do not match for condenser {condenser.name}"
                )

    @staticmethod
    def _add_flow_distributor(process_model: "ProcessModel") -> "ProcessModel":
        process_model._assert_condenser_sources(process_model)
        flow_splitter = FlowSplitter(
            name="flow_splitter", model=FlowSplitterModel(name="flow_splitter")
        )
        process_model.equipment_list.append(flow_splitter)
        for i, condenser in enumerate(process_model.condenser_list):
            current_upstream_pipes = process_model.get_upstream_pipes(condenser.name)
            if i == 0:
                # only add the first time since all condensers have the same sources
                for pipe in current_upstream_pipes:
                    process_model.pipe_list.append(
                        Pipe(source=pipe.source, target=flow_splitter.name)
                    )

            pipe = Pipe(source=flow_splitter.name, target=condenser.name)
            process_model.pipe_list.append(pipe)
            # now drop current upstream pipes
            for pipe in current_upstream_pipes:
                process_model.pipe_list.remove(pipe)

        return process_model

    @staticmethod
    def _identify_suction_vessels(
        upstream_vessels: list[Vessel],
    ) -> tuple[Vessel, Vessel]:
        if (
            upstream_vessels[0].inlet_pressure__psig
            > upstream_vessels[1].inlet_pressure__psig
        ):
            main_suction_vessel = upstream_vessels[1]
            economizer_vessel = upstream_vessels[0]
        else:
            main_suction_vessel = upstream_vessels[0]
            economizer_vessel = upstream_vessels[1]

        return main_suction_vessel, economizer_vessel

    def identify_evaporator_vessels(self, evaporator_name: str) -> str:
        upstream_pipes = self.get_upstream_pipes(evaporator_name)
        upstream_pumps = [
            self.get_equipment_by_name(pipe.source) for pipe in upstream_pipes
        ]
        further_upstream_pipes = [
            self.get_upstream_pipe(pump.name) for pump in upstream_pumps
        ]
        upstream_vessels = set(
            [
                self.get_equipment_by_name(pipe.source).name
                for pipe in further_upstream_pipes
            ]
        )
        if len(upstream_vessels) != 1:
            raise ValueError(
                f"More than one upstream vessel found for evaporator {evaporator_name}"
            )
        return list(upstream_vessels)[0]

    def _vessel_level_protection(
        self, vessel: Vessel, vessel_state: VesselState
    ) -> float:
        if vessel.model.orientation == "vertical":
            if vessel_state.m_l__kg < 0.01:
                level__m = 0.01 * vessel.model.length__m
            else:
                level__m = 0.99 * vessel.model.length__m
        else:
            if vessel_state.m_l__kg < 0.01:
                level__m = 0.01 * vessel.model.diameter__m
            else:
                level__m = 0.99 * vessel.model.diameter__m

        return level__m

    def solve_for_pressures(
        self, state: FullProcessState, disturbances: Disturbances
    ) -> None:
        # condensers
        for condenser in self.condenser_list:
            condenser_state = state.get_equipment_state(condenser.name)
            vapor_thermo = condenser.model.get_vapor_state(condenser_state)
            condenser.inlet_pressure__psig = vapor_thermo.pressure__psig

        # vessels
        for vessel in self.vessel_list:
            vessel_state = state.get_equipment_state(vessel.name)
            liquid_thermo = vessel.model.get_liquid_state(vessel_state)
            vapor_thermo = vessel.model.get_vapor_state(vessel_state, liquid_thermo)
            vessel.inlet_pressure__psig = vapor_thermo.pressure__psig

        # evaporators
        for evaporator in self.evaporator_list:
            downstream_pipe = self.get_downstream_pipe(evaporator.name)
            downstream_vessel: Vessel = self.get_equipment_by_name(
                downstream_pipe.target
            )
            evaporator.inlet_pressure__psig = downstream_vessel.inlet_pressure__psig

        # flow splitter
        flow_splitter = self.get_equipment_by_name("flow_splitter")
        downstream_pipes = self.get_downstream_pipes(flow_splitter.name)
        downstream_equipment = [
            self.get_equipment_by_name(pipe.target) for pipe in downstream_pipes
        ]
        downstream_pressures__psig = [
            equipment.inlet_pressure__psig for equipment in downstream_equipment
        ]
        flow_splitter.inlet_pressure__psig = max(downstream_pressures__psig)

        # compressors
        for compressor in self.compressor_list:
            upstream_pipes = self.get_upstream_pipes(compressor.name)
            upstream_vessels: list[Vessel] = [
                self.get_equipment_by_name(pipe.source) for pipe in upstream_pipes
            ]
            if len(upstream_vessels) == 1:
                upstream_vessel = upstream_vessels[0]
                compressor.inlet_pressure__psig = upstream_vessel.inlet_pressure__psig
            elif len(upstream_vessels) == 2:
                main_suction_vessel, economizer_vessel = self._identify_suction_vessels(
                    upstream_vessels
                )
                compressor.inlet_pressure__psig = (
                    main_suction_vessel.inlet_pressure__psig
                )
                compressor.economizer_pressure__psig = (
                    economizer_vessel.inlet_pressure__psig
                )
            elif len(upstream_vessels) == 0:
                raise ValueError(
                    f"No upstream vessels found for compressor {compressor.name}"
                )
            else:
                raise ValueError(
                    f"More than two upstream vessels found for compressor {compressor.name}"
                )

            downstream_pipe = self.get_downstream_pipe(compressor.name)
            downstream_equipment = self.get_equipment_by_name(downstream_pipe.target)
            compressor.discharge_pressure__psig = (
                downstream_equipment.inlet_pressure__psig
            )

    def solve_for_flowrates_and_enthalpies(
        self, state: FullProcessState, disturbances: Disturbances
    ) -> None:
        # TODO: This method is too long, consider refactoring

        # ensure all flowrates and enthalpies are None to start
        for pipe in self.pipe_list:
            pipe.flowrate__kg_s = None
            pipe.enthalpy__kj_kg = None

        # first solve for the condenser drain
        for condenser in self.condenser_list:
            condenser_state = state.get_equipment_state(condenser.name)
            condenser_point: CondenserTimePoint = condenser.actuator
            vapor_thermo = condenser.model.get_vapor_state(condenser_state)
            condensate_thermo = condenser.model.get_condensate_state(vapor_thermo)
            cd_flow__kg_s = condenser.model.solve_for_out_flow__kg_s(
                vapor_thermo,
                condensate_thermo,
                condenser_point,
                disturbances.get_disturbance_by_name("ambient_temp__c").value(
                    condenser_state.date_time
                ),
                disturbances.get_disturbance_by_name("ambient_rh__perc").value(
                    condenser_state.date_time
                ),
                condenser_state,
            )
            pipe = self.get_downstream_pipe(condenser.name)
            pipe.flowrate__kg_s = cd_flow__kg_s
            pipe.enthalpy__kj_kg = condensate_thermo.enthalpy__kj_kg
            condenser.measurements = CondenserMeasurement(
                name=condenser.name,
                date_time=condenser_state.date_time,
                condenser_power__kw=condenser.model.get_power__kw(condenser_point),
                ambient_temp__c=disturbances.get_disturbance_by_name(
                    "ambient_temp__c"
                ).value(condenser_state.date_time),
                ambient_rh__perc=disturbances.get_disturbance_by_name(
                    "ambient_rh__perc"
                ).value(condenser_state.date_time),
                pressure__psig=vapor_thermo.pressure__psig,
                fan_run_proof_1=condenser_point.fan_speed_1__rpm > 10,
                fan_run_proof_2=None
                if condenser_point.fan_speed_2__rpm is None
                else condenser_point.fan_speed_2__rpm > 10,
                fan_run_proof_3=None
                if condenser_point.fan_speed_3__rpm is None
                else condenser_point.fan_speed_3__rpm > 10,
                fan_run_proof_4=None
                if condenser_point.fan_speed_4__rpm is None
                else condenser_point.fan_speed_4__rpm > 10,
                pump_run_proof_1=condenser_point.pump_1_on,
                pump_run_proof_2=condenser_point.pump_2_on,
                pump_run_proof_3=condenser_point.pump_3_on,
            )

        # then solve for the valve flows
        for valve in self.valve_list:
            valve_state = state.get_equipment_state(valve.name)
            flow__kg_s = valve.model.get_flow_rate__kg_s(valve.actuator, valve_state)
            upstream_pipe = self.get_upstream_pipe(valve.name)
            upstream_pipe.flowrate__kg_s = flow__kg_s
            source_vessel: Vessel = self.get_equipment_by_name(upstream_pipe.source)
            if source_vessel.kind != "vessel":
                raise ValueError(
                    f"Upstream equipment is not a vessel: {upstream_pipe.source}"
                )
            source_vessel_state: VesselState = state.get_equipment_state(
                source_vessel.name
            )
            upstream_pipe.enthalpy__kj_kg = source_vessel_state.h_l__kj_kg
            downstream_pipe = self.get_downstream_pipe(valve.name)
            downstream_pipe.flowrate__kg_s = flow__kg_s
            downstream_pipe.enthalpy__kj_kg = upstream_pipe.enthalpy__kj_kg
            valve.measurements = ValveMeasurement(
                name=valve.name,
                date_time=valve_state.date_time,
                valve_flow_rate__kg_s=flow__kg_s,
            )

        # then solve for the vessel flows
        for vessel in self.vessel_list:
            vessel_state = state.get_equipment_state(vessel.name)
            liquid_state = vessel.model.get_liquid_state(vessel_state)
            try:
                level__m = vessel.model.level__m(
                    liquid_density__kg_m3=liquid_state.density__kg_m3,
                    m_l__kg=vessel_state.m_l__kg,
                )
            except Exception:
                level__m = self._vessel_level_protection(vessel, vessel_state)

            vessel.measurements = VesselMeasurement(
                name=vessel.name,
                date_time=vessel_state.date_time,
                pressure__psig=vessel.inlet_pressure__psig,
                level__m=level__m,
                level__perc=vessel.model.level__perc(level__m),
                interface_pressure__psig=liquid_state.pressure__psig,
            )

        # then solve for the evaporator flows
        for room in self.refrigerated_rooms:
            room_state = state.get_equipment_state(room.name)
            evaporators = [
                self.get_equipment_by_name(evap) for evap in room.model.evaporators
            ]
            for evaporator in evaporators:
                assert isinstance(evaporator, Evaporator)
                evaporator_state = state.get_equipment_state(evaporator.name)
                upstream_pipes = self.get_upstream_pipes(evaporator.name)
                if not all(
                    self.get_device_by_name(pipe.source).kind == "pump"
                    for pipe in upstream_pipes
                ):
                    raise ValueError(
                        f"Not all upstream equipment is a pump: {upstream_pipes}"
                    )
                upstream_pumps: list[Pump] = [
                    self.get_equipment_by_name(pipe.source) for pipe in upstream_pipes
                ]
                pump_points = [
                    self.get_actuator_by_name(pump.name) for pump in upstream_pumps
                ]
                pump_states = [
                    state.get_equipment_state(pump.name) for pump in upstream_pumps
                ]
                pumps_flowing = [
                    pump.model.is_flowing(pump_point, pump_state)
                    for (pump, pump_point, pump_state) in zip(
                        upstream_pumps, pump_points, pump_states
                    )
                ]
                evaporator_flow__kg_s = evaporator.model.solve_for_out_flow__kg_s(
                    evaporator.actuator, any(pumps_flowing), evaporator_state
                )
                downstream_pipe = self.get_downstream_pipe(evaporator.name)
                downstream_pipe.flowrate__kg_s = evaporator_flow__kg_s
                evaporator_state: EvaporatorState = state.get_equipment_state(
                    evaporator.name
                )
                downstream_pipe.enthalpy__kj_kg = evaporator_state.h__kj_kg

                # now set the upstream pipe flowrates and enthalpies
                for pipe in upstream_pipes:
                    upstream_pump: Pump = self.get_equipment_by_name(pipe.source)
                    upstream_pump_point = self.get_actuator_by_name(upstream_pump.name)
                    upstream_pump_state = state.get_equipment_state(upstream_pump.name)
                    further_upstream_pipe = self.get_upstream_pipe(upstream_pump.name)
                    vessel: Vessel = self.get_equipment_by_name(
                        further_upstream_pipe.source
                    )
                    vessel_state: VesselState = state.get_equipment_state(vessel.name)
                    further_upstream_pipe.enthalpy__kj_kg = vessel_state.h_l__kj_kg
                    pipe.enthalpy__kj_kg = further_upstream_pipe.enthalpy__kj_kg
                    pump_on = upstream_pump.model.is_flowing(
                        upstream_pump_point, upstream_pump_state
                    )
                    pipe.flowrate__kg_s = evaporator_flow__kg_s * pump_on
                    if further_upstream_pipe.flowrate__kg_s is None:
                        further_upstream_pipe.flowrate__kg_s = pipe.flowrate__kg_s
                    else:
                        further_upstream_pipe.flowrate__kg_s += pipe.flowrate__kg_s

                    if upstream_pump.measurements is None:
                        upstream_pump.measurements = PumpMeasurement(
                            name=upstream_pump.name,
                            date_time=evaporator_state.date_time,
                            pump_power__kw=upstream_pump.model.get_power__kw(
                                upstream_pump_point
                            ),
                            run_proof=pump_on,
                        )

                evaporator.measurements = EvaporatorMeasurement(
                    name=evaporator.name,
                    date_time=evaporator_state.date_time,
                    evaporator_power__kw=evaporator.model.get_power__kw(
                        evaporator.actuator
                    ),
                    suction_pressure__psig=evaporator.inlet_pressure__psig,
                    room_temp__c=room_state.air_temp__c,
                    fan_run_proof_1=evaporator.actuator.fan_speed_1__rpm > 10,
                    fan_run_proof_2=None
                    if evaporator.actuator.fan_speed_2__rpm is None
                    else evaporator.actuator.fan_speed_2__rpm > 10,
                    fan_run_proof_3=None
                    if evaporator.actuator.fan_speed_3__rpm is None
                    else evaporator.actuator.fan_speed_3__rpm > 10,
                )

            # now set the room measurements
            room.measurements = RoomMeasurement(
                name=room.name,
                date_time=evaporator_state.date_time,
                air_temp__f=UnitConversions.celsius_to_fahrenheit(
                    np.median(
                        [
                            evaporator.measurements.room_temp__c
                            for evaporator in evaporators
                        ]
                    )
                ),
                int_product_temp__f=UnitConversions.celsius_to_fahrenheit(
                    room_state.internal_product_temp__c
                ),
                ext_product_temp__f=UnitConversions.celsius_to_fahrenheit(
                    room_state.external_product_temp__c
                ),
            )

        # then solve for the compressor flows
        for compressor in self.compressor_list:
            compressor_state = state.get_equipment_state(compressor.name)
            suction_state = AmmoniaThermo(
                pressure__pa=UnitConversions.psig_to_pascal(
                    compressor.inlet_pressure__psig
                ),
                quality=1.0,
            )
            condensate_state = AmmoniaThermo(
                pressure__pa=UnitConversions.psig_to_pascal(
                    compressor.discharge_pressure__psig
                ),
                quality=0.0,
            )

            compression_return = compressor.model.run_compression(
                compressor.actuator,
                suction_state,
                condensate_state,
                compressor_state,
            )
            if compression_return.suction_mass_flow__kg_s <= 0:
                discharge_enthalpy__kj_kg = suction_state.enthalpy__kj_kg
            else:
                discharge_enthalpy__kj_kg = (
                    compression_return.shaft_work__kw
                    / compression_return.suction_mass_flow__kg_s
                    + suction_state.enthalpy__kj_kg
                )
            upstream_pipe = self.get_upstream_pipe(compressor.name)
            downstream_pipe = self.get_downstream_pipe(compressor.name)
            upstream_pipe.flowrate__kg_s = compression_return.suction_mass_flow__kg_s
            upstream_pipe.enthalpy__kj_kg = suction_state.enthalpy__kj_kg
            downstream_pipe.flowrate__kg_s = compression_return.suction_mass_flow__kg_s
            downstream_pipe.enthalpy__kj_kg = discharge_enthalpy__kj_kg

            compressor.measurements = compression_return

        # now redistribute flow to the condensers to equilibrate pressures
        flow_splitter: FlowSplitter = self.get_equipment_by_name("flow_splitter")
        flow_splitter_state: FlowSplitterState = state.get_equipment_state(
            flow_splitter.name
        )
        upstream_pipes = self.get_upstream_pipes(flow_splitter.name)
        downstream_pipes = self.get_downstream_pipes(flow_splitter.name)
        flow_splitter.model.split_flow(
            state=flow_splitter_state,
            inlet_pipe_list=upstream_pipes,
            outlet_pipe_list=downstream_pipes,
        )
        flow_distribution = flow_splitter.model.flow_distribution(
            state=flow_splitter_state
        )
        flow_splitter.measurements = FlowSplitterMeasurement(
            name=flow_splitter.name,
            date_time=flow_splitter_state.date_time,
            flow_fraction_1=flow_distribution[0],
            flow_fraction_2=flow_distribution[1]
            if len(flow_distribution) > 1
            else None,
            flow_fraction_3=flow_distribution[2]
            if len(flow_distribution) > 2
            else None,
            flow_fraction_4=flow_distribution[3]
            if len(flow_distribution) > 3
            else None,
        )

        total_refrigeration_power__kw = 0
        for condenser_measurement in self.get_condenser_measurements():
            total_refrigeration_power__kw += condenser_measurement.condenser_power__kw

        for evaporator_measurement in self.get_evaporator_measurements():
            total_refrigeration_power__kw += evaporator_measurement.evaporator_power__kw

        for pump_measurement in self.get_pump_measurements():
            total_refrigeration_power__kw += pump_measurement.pump_power__kw

        for compressor_measurement in self.get_compressor_measurements():
            total_refrigeration_power__kw += compressor_measurement.power_use__kw

        self.building_measurement = BuildingMeasurement(
            name=self.name,
            date_time=flow_splitter_state.date_time,
            total_refrigeration_power__kw=total_refrigeration_power__kw,
            machine_room_temperature__f=UnitConversions.celsius_to_fahrenheit(
                disturbances.get_disturbance_by_name("machine_room_temp__c").value(
                    flow_splitter_state.date_time
                )
            ),
        )

        # assert that all flowrates and enthalpies are set
        for pipe in self.pipe_list:
            if pipe.flowrate__kg_s is None:
                raise ValueError(f"{pipe} flowrate is None")
            if pipe.enthalpy__kj_kg is None:
                raise ValueError(f"{pipe} enthalpy is None")

    def solve_for_state_derivatives(
        self, state: FullProcessState, disturbances: Disturbances
    ) -> FullProcessStateDerivative:
        process_state_derivatives = []
        for vessel in self.vessel_list:
            vessel_state = state.get_equipment_state(vessel.name)
            inlet_streams = self.get_upstream_pipes(vessel.name)
            vapor_pipes = [
                pipe
                for pipe in self.get_downstream_pipes(vessel.name)
                if self.get_equipment_by_name(pipe.target).kind == "compressor"
            ]
            vapor_flow__kg_s = sum([pipe.flowrate__kg_s for pipe in vapor_pipes])
            liquid_pipes = [
                pipe
                for pipe in self.get_downstream_pipes(vessel.name)
                if pipe not in vapor_pipes
            ]
            liquid_flow__kg_s = sum([pipe.flowrate__kg_s for pipe in liquid_pipes])
            liquid_thermo = vessel.model.get_liquid_state(vessel_state)
            vapor_thermo = vessel.model.get_vapor_state(vessel_state, liquid_thermo)
            heat_transfer_rate__kw = vessel.model.get_vessel_heat_rate__kw(
                liquid_thermo,
                disturbances.get_disturbance_by_name("machine_room_temp__c").value(
                    vessel_state.date_time
                ),
            )
            process_state_derivatives.append(
                vessel.model.get_state_derivative(
                    state=vessel_state,
                    simulation_point=None,
                    inlets=inlet_streams,
                    vapor_thermo=vapor_thermo,
                    liquid_thermo=liquid_thermo,
                    liquid_flow_out__kg_s=liquid_flow__kg_s,
                    vapor_flow_out__kg_s=vapor_flow__kg_s,
                    heat_transfer_rate__kw=heat_transfer_rate__kw,
                )
            )

        for condenser in self.condenser_list:
            condenser_state = state.get_equipment_state(condenser.name)
            inlet_streams = self.get_upstream_pipes(condenser.name)
            cd_flow__kg_s = self.get_downstream_pipe(condenser.name).flowrate__kg_s
            process_state_derivatives.append(
                condenser.model.get_state_derivative(
                    state=condenser_state,
                    simulation_point=condenser.actuator,
                    inlets=inlet_streams,
                    flow_out__kg_s=cd_flow__kg_s,
                )
            )

        for room in self.refrigerated_rooms:
            room_state = state.get_equipment_state(room.name)
            evaporators: list[Evaporator] = [
                self.get_equipment_by_name(evap) for evap in room.model.evaporators
            ]
            room_heat_transfer_rate__kw = 0
            for evaporator in evaporators:
                evaporator_state = state.get_equipment_state(evaporator.name)
                inlet_streams = self.get_upstream_pipes(evaporator.name)
                liquid_thermo = evaporator.model.get_liquid_state(
                    evaporator_state, evaporator.inlet_pressure__psig
                )
                heat_transfer_rate__kw = evaporator.model.get_heat_transfer_rate__kw(
                    inlet_thermo=liquid_thermo,
                    simulation_point=evaporator.actuator,
                    room_temp__c=room_state.air_temp__c,
                    state=evaporator_state,
                )
                room_heat_transfer_rate__kw += heat_transfer_rate__kw
                process_state_derivatives.append(
                    evaporator.model.get_state_derivative(
                        state=evaporator_state,
                        simulation_point=evaporator.actuator,
                        inlets=inlet_streams,
                        heat_transfer_rate__kw=heat_transfer_rate__kw,
                    )
                )

            room_heat_load__kw = room.model.get_heat_load__kw(
                state=room_state,
                ambient_temp__c=disturbances.get_disturbance_by_name(
                    "ambient_temp__c"
                ).value(room_state.date_time),
                ambient_rh__perc=disturbances.get_disturbance_by_name(
                    "ambient_rh__perc"
                ).value(room_state.date_time),
                sun_angle__deg=disturbances.get_disturbance_by_name(
                    "sun_angle_altitude"
                ).value(room_state.date_time),
                cloud_cover__perc=disturbances.get_disturbance_by_name(
                    "cloud_cover__perc"
                ).value(room_state.date_time),
                date_time=room_state.date_time,
            )
            process_state_derivatives.append(
                room.model.get_state_derivative(
                    state=room_state,
                    heat_load__kw=room_heat_load__kw,
                    evap_q__kw=room_heat_transfer_rate__kw,
                )
            )

        for compressor in self.compressor_list:
            compressor_state = state.get_equipment_state(compressor.name)
            process_state_derivatives.append(
                compressor.model.get_state_derivative(
                    state=compressor_state,
                    simulation_point=compressor.actuator,
                )
            )

        for pump in self.pump_list:
            pump_state = state.get_equipment_state(pump.name)
            process_state_derivatives.append(
                pump.model.get_state_derivative(
                    state=pump_state,
                    simulation_point=pump.actuator,
                )
            )

        for valve in self.valve_list:
            valve_state = state.get_equipment_state(valve.name)
            process_state_derivatives.append(
                valve.model.get_state_derivative(
                    state=valve_state,
                    simulation_point=valve.actuator,
                )
            )

        flow_splitter: FlowSplitter = self.get_equipment_by_name("flow_splitter")
        flow_splitter_state = state.get_equipment_state(flow_splitter.name)
        inlet_streams = self.get_upstream_pipes(flow_splitter.name)
        downstream_pipes = self.get_downstream_pipes(flow_splitter.name)
        downstream_equipment = [
            self.get_equipment_by_name(pipe.target) for pipe in downstream_pipes
        ]
        downstream_pressures__psig = [
            equipment.inlet_pressure__psig for equipment in downstream_equipment
        ]
        process_state_derivatives.append(
            flow_splitter.model.get_state_derivative(
                state=flow_splitter_state,
                simulation_point=None,
                inlets=inlet_streams,
                pressures=downstream_pressures__psig,
            )
        )

        return FullProcessStateDerivative(
            equipment_state_derivative_list=process_state_derivatives
        )

    def solve_flowsheet(
        self,
        state: FullProcessState,
        disturbances: Disturbances,
    ) -> None:
        self.solve_for_pressures(state, disturbances)
        self.solve_for_flowrates_and_enthalpies(state, disturbances)

    def get_state_derivatives(
        self,
        state: FullProcessState,
        disturbances: Disturbances,
    ) -> FullProcessStateDerivative:
        self.solve_flowsheet(state, disturbances)
        return self.solve_for_state_derivatives(state, disturbances)

    def get_measurements(self) -> list[Measurement | BuildingMeasurement]:
        measurements = []
        for equipment in self.sorted_equipment_list + self.refrigerated_rooms:
            if equipment.measurements is not None:
                measurements.append(equipment.measurements)

        return measurements + [self.building_measurement]

    def set_measurements(
        self,
        measurements: list[Measurement | BuildingMeasurement],
    ) -> None:
        for measurement in measurements:
            if measurement.name in [
                equipment.name for equipment in self.equipment_list
            ]:
                equipment = self.get_equipment_by_name(measurement.name)
                equipment.measurements = measurement

            elif measurement.name in [room.name for room in self.refrigerated_rooms]:
                room = self.get_room_by_name(measurement.name)
                room.measurements = measurement

            elif measurement.name == self.name:
                self.building_measurement = measurement

    def get_vessel_measurements(self) -> list[VesselMeasurement]:
        measurement_list = self.get_measurements()
        return [
            measurement
            for measurement in measurement_list
            if isinstance(measurement, VesselMeasurement)
        ]

    def get_condenser_measurements(self) -> list[CondenserMeasurement]:
        measurement_list = self.get_measurements()
        return [
            measurement
            for measurement in measurement_list
            if isinstance(measurement, CondenserMeasurement)
        ]

    def get_evaporator_measurements(self) -> list[EvaporatorMeasurement]:
        measurement_list = self.get_measurements()
        return [
            measurement
            for measurement in measurement_list
            if isinstance(measurement, EvaporatorMeasurement)
        ]

    def get_pump_measurements(self) -> list[PumpMeasurement]:
        measurement_list = self.get_measurements()
        return [
            measurement
            for measurement in measurement_list
            if isinstance(measurement, PumpMeasurement)
        ]

    def get_valve_measurements(self) -> list[ValveMeasurement]:
        measurement_list = self.get_measurements()
        return [
            measurement
            for measurement in measurement_list
            if isinstance(measurement, ValveMeasurement)
        ]

    def get_compressor_measurements(self) -> list[CompressionReturn]:
        measurement_list = self.get_measurements()
        return [
            measurement
            for measurement in measurement_list
            if isinstance(measurement, CompressionReturn)
        ]

    def get_actuators(self) -> Actuators:
        actuator_list = []
        for equipment in self.sorted_equipment_list:
            if equipment.actuator is not None:
                actuator_list.append(equipment.actuator)

        return Actuators(actuator_list=actuator_list)

    def get_actuator_by_name(self, name: str) -> TimePoint:
        for actuator in self.get_actuators().actuator_list:
            if actuator.name == name:
                return actuator

        raise ValueError(f"Actuator not found: {name}")

    def set_actuators(self, actuators: Actuators) -> None:
        for actuator in actuators.actuator_list:
            equipment = self.get_equipment_by_name(actuator.name)
            equipment.actuator = actuator

    def get_state_from_measurements(
        self,
        measurements: list[Measurement],
        inputs: Actuators,
    ) -> FullProcessState:
        equipment_states = []
        for measurement in measurements:
            actuator = inputs.get_actuator_by_name(measurement.name)
            equipment = self.get_equipment_by_name(measurement.name)
            equipment_state = equipment.model.get_state_from_measurement(
                measurement, actuator
            )
            equipment_states.append(equipment_state)

        return FullProcessState(equipment_state_list=equipment_states)

    def initialize_building_measurement(
        self, date_time: datetime
    ) -> BuildingMeasurement:
        return BuildingMeasurement(
            name=self.name,
            date_time=date_time,
            total_refrigeration_power__kw=0,
            machine_room_temperature__f=70,
        )

    def configure_actuators_and_measurements(
        self,
        start_datetime: datetime,
    ) -> FullProcessState:
        actuators = Actuators(actuator_list=[])
        full_process_state = FullProcessState(simulation_time=start_datetime)
        measurements = []
        for vessel in self.vessel_list:
            vessel_state, vessel_measurement = vessel.model.initialize(start_datetime)
            measurements.append(vessel_measurement)
            full_process_state.equipment_state_list.append(vessel_state)

        for condenser in self.condenser_list:
            (
                condenser_state,
                condenser_measurement,
                condenser_actuator,
            ) = condenser.model.initialize(start_datetime)
            measurements.append(condenser_measurement)
            actuators.actuator_list.append(condenser_actuator)
            full_process_state.equipment_state_list.append(condenser_state)

        for evaporator in self.evaporator_list:
            source_vessel_name = self.identify_evaporator_vessels(evaporator.name)
            source_vessel: Vessel = self.get_equipment_by_name(source_vessel_name)
            (
                evaporator_state,
                evaporator_measurement,
                evaporator_actuator,
            ) = evaporator.model.initialize(
                start_datetime, source_vessel.model.default_temp__f
            )
            measurements.append(evaporator_measurement)
            actuators.actuator_list.append(evaporator_actuator)
            full_process_state.equipment_state_list.append(evaporator_state)

        for compressor in self.compressor_list:
            (
                compressor_state,
                compressor_measurement,
                compressor_actuator,
            ) = compressor.model.initialize(start_datetime)
            measurements.append(compressor_measurement)
            actuators.actuator_list.append(compressor_actuator)
            full_process_state.equipment_state_list.append(compressor_state)

        for pump in self.pump_list:
            pump_state, pump_measurement, pump_actuator = pump.model.initialize(
                start_datetime
            )
            measurements.append(pump_measurement)
            full_process_state.equipment_state_list.append(pump_state)
            actuators.actuator_list.append(pump_actuator)

        for valve in self.valve_list:
            valve_state, valve_measurement, valve_actuator = valve.model.initialize(
                start_datetime
            )
            measurements.append(valve_measurement)
            full_process_state.equipment_state_list.append(valve_state)
            actuators.actuator_list.append(valve_actuator)

        for room in self.refrigerated_rooms:
            room_state, room_measurement = room.model.initialize(start_datetime)
            measurements.append(room_measurement)
            full_process_state.equipment_state_list.append(room_state)

        flow_splitter = self.get_device_by_name("flow_splitter")
        if flow_splitter is not None:
            (
                flow_splitter_state,
                flow_splitter_measurement,
            ) = flow_splitter.model.initialize(start_datetime, len(self.condenser_list))
            measurements.append(flow_splitter_measurement)
            full_process_state.equipment_state_list.append(flow_splitter_state)

        measurements.append(
            self.initialize_building_measurement(date_time=start_datetime)
        )

        self.set_measurements(measurements)
        self.set_actuators(actuators)
        return full_process_state

    def current_graph(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        for equipment in self.equipment_list:
            graph.add_node(equipment.name)
        for pipe in self.pipe_list:
            graph.add_edge(pipe.source, pipe.target)

        return graph

    def plot_graph(self, graph: nx.DiGraph) -> None:
        for device_name in graph.nodes:
            try:
                device = self.get_device_by_name(device_name)
            except ValueError:
                graph.nodes[device_name]["color"] = "black"
                graph.nodes[device_name]["weight"] = 1
            if "compressor" in device.kind:
                graph.nodes[device_name]["color"] = "red"
                graph.nodes[device_name]["weight"] = 3
            elif "condenser" in device.kind:
                graph.nodes[device_name]["color"] = "blue"
                graph.nodes[device_name]["weight"] = 2
            elif "evaporator" in device.kind:
                graph.nodes[device_name]["color"] = "green"
                graph.nodes[device_name]["weight"] = 1
            elif "vessel" in device.kind:
                graph.nodes[device_name]["color"] = "yellow"
                graph.nodes[device_name]["weight"] = 4
            elif "valve" in device.kind or "pump" in device.kind:
                graph.nodes[device_name]["color"] = "magenta"
                graph.nodes[device_name]["weight"] = 2
            else:
                graph.nodes[device_name]["color"] = "black"
                graph.nodes[device_name]["weight"] = 1

        pos = nx.spring_layout(graph)
        node_color = [graph.nodes[node]["color"] for node in graph]
        nx.draw(graph, pos, node_color=node_color, with_labels=True)
        plt.show()
