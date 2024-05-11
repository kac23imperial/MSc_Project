from abc import ABC
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from ck_fridge_sim.process_equipment_models.compressor_model import (
    CompressionReturn,
    CompressorModel,
    CompressorTimePoint,
)
from ck_fridge_sim.process_equipment_models.condenser_model import (
    CondenserMeasurement,
    CondenserModel,
    CondenserTimePoint,
)
from ck_fridge_sim.process_equipment_models.evaporator_model import (
    EvaporatorMeasurement,
    EvaporatorModel,
    EvaporatorTimePoint,
)
from ck_fridge_sim.process_equipment_models.flow_splitter_model import (
    FlowSplitterMeasurement,
    FlowSplitterModel,
)
from ck_fridge_sim.process_equipment_models.pump_model import (
    PumpMeasurement,
    PumpModel,
    PumpTimePoint,
)
from ck_fridge_sim.process_equipment_models.refrigeration_model import (
    RefrigeratedRoom,
    RoomMeasurement,
)
from ck_fridge_sim.process_equipment_models.valve_model import (
    ValveMeasurement,
    ValveModel,
    ValveTimePoint,
)
from ck_fridge_sim.process_equipment_models.vessel_model import (
    VesselMeasurement,
    VesselModel,
)


class EquipmentBase(BaseModel, ABC):
    name: str
    kind: str
    model: (
        VesselModel
        | CompressorModel
        | CondenserModel
        | EvaporatorModel
        | RefrigeratedRoom
        | FlowSplitterModel
        | PumpModel
        | ValveModel
        | None
    ) = None
    inlet_pressure__psig: None | float = None
    measurements: (
        None
        | list[
            VesselMeasurement
            | PumpMeasurement
            | ValveMeasurement
            | EvaporatorMeasurement
            | CondenserMeasurement
            | RoomMeasurement
            | FlowSplitterMeasurement
        ]
    ) = None
    actuator: (
        ValveTimePoint
        | PumpTimePoint
        | CompressorTimePoint
        | EvaporatorTimePoint
        | CondenserTimePoint
        | None
    ) = None


class Pump(EquipmentBase):
    kind: Literal["pump"] = "pump"
    model: PumpModel
    measurements: None | PumpMeasurement = None
    actuator: None | PumpTimePoint = None


class Valve(EquipmentBase):
    kind: Literal["valve"] = "valve"
    model: ValveModel
    measurements: None | ValveMeasurement = None
    actuator: None | ValveTimePoint = None


class Compressor(EquipmentBase):
    kind: Literal["compressor"] = "compressor"
    model: CompressorModel
    discharge_pressure__psig: None | float = None
    economizer_pressure__psig: None | float = None
    measurements: None | CompressionReturn = None
    actuator: None | CompressorTimePoint = None


class Condenser(EquipmentBase):
    kind: Literal["condenser"] = "condenser"
    model: CondenserModel
    measurements: None | CondenserMeasurement = None
    actuator: None | CondenserTimePoint = None


class Vessel(EquipmentBase):
    kind: Literal["vessel"] = "vessel"
    model: VesselModel
    measurements: None | VesselMeasurement = None
    actuator: None = None


class Evaporator(EquipmentBase):
    kind: Literal["evaporator"] = "evaporator"
    model: EvaporatorModel
    measurements: None | EvaporatorMeasurement = None
    actuator: None | EvaporatorTimePoint = None


class Room(EquipmentBase):
    kind: Literal["refrigerated_room"] = "refrigerated_room"
    model: RefrigeratedRoom
    measurements: None | RoomMeasurement = None
    actuator: None = None


class Pipe(BaseModel):
    source: str
    target: str
    flowrate__kg_s: None | float = None
    enthalpy__kj_kg: None | float = None


class FlowSplitter(EquipmentBase):
    kind: Literal["flow_splitter"] = "flow_splitter"
    model: FlowSplitterModel
    measurements: None | FlowSplitterMeasurement = None
    actuator: None = None


Equipment = Annotated[
    Pump | Valve | Compressor | Condenser | Vessel | Evaporator | Room | FlowSplitter,
    Field(discriminator="kind"),
]

Measurement = Annotated[
    PumpMeasurement
    | ValveMeasurement
    | VesselMeasurement
    | EvaporatorMeasurement
    | CondenserMeasurement
    | RoomMeasurement
    | FlowSplitterMeasurement,
    Field(discriminator="kind"),
]

TimePoint = Annotated[
    PumpTimePoint
    | ValveTimePoint
    | CompressorTimePoint
    | EvaporatorTimePoint
    | CondenserTimePoint,
    Field(discriminator="kind"),
]
