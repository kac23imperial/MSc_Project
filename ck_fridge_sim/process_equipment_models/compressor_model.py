from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, field_validator

from ck_fridge_sim.base_model_classes import (
    MeasurementBase,
    ModelBase,
    StateBase,
    StateDerivativeBase,
    TimePointBase,
)
from ck_fridge_sim.thermo_tools.ammonia_thermo import AmmoniaThermo
from ck_fridge_sim.thermo_tools.thermo_processes.irreversible_compression import (
    IrreversibleCompression,
)
from ck_fridge_sim.thermo_tools.unit_conversions import UnitConversions

COMPRESSOR_FITS = (
    Path(__file__).parent.parent / "compressor_data/global_compressor_model.json"
)


class CompressorTimePoint(TimePointBase):
    kind: Literal["compressor"] = "compressor"
    compressor_speed__rpm: float
    sv_position__perc: float
    compressor_current__amps: float | None = None
    side_port: bool = False

    @field_validator("compressor_speed__rpm", mode="before")
    def prevent_null_speed(cls, v):
        assert (
            not np.isnan(v) and v is not None
        ), f"compressor_speed__rpm may not be null, got {v}"
        return v

    @field_validator("sv_position__perc", mode="before")
    def prevent_null_sv(cls, v):
        assert (
            not np.isnan(v) and v is not None
        ), f"sv_position__perc may not be null, got {v}"
        return v

    @field_validator("compressor_speed__rpm", mode="before")
    def rpm_speed_range(cls, v):
        assert (
            v >= 0 and v <= 3600
        ), f"compressor_speed__rpm must be between 0 and 3600, got {v}"
        return v

    @field_validator("sv_position__perc", mode="before")
    def sv_position_range(cls, v):
        assert (
            v >= 0 and v <= 100
        ), f"sv_position__perc must be between 0 and 100, got {v}"
        return v

    @classmethod
    def rpm_and_sv_from_capacity(
        cls, capacity_cmd: float, min_rpm: float = 1200
    ) -> tuple[float, float]:
        rpm = max(
            min_rpm,
            (capacity_cmd - 50) * 2 / 100 * (3600 - min_rpm) + min_rpm,
        )
        if (3600 - min_rpm) == 0:
            sv = capacity_cmd
        else:
            sv = min(50, capacity_cmd) * 2

        return rpm, sv


class CompressorState(StateBase):
    kind: Literal["compressor"] = "compressor"
    compressor_speed__rpm: float
    sv_position__perc: float


class CompressorStateDerivative(CompressorState, StateDerivativeBase):
    pass


class OptimizationVar(BaseModel):
    value: float
    var_name: str
    upper_bound: float
    lower_bound: float


class CompressionReturn(MeasurementBase):
    kind: Literal["compressor"] = "compressor"
    shaft_work__kw: float
    evap_heat_removed__kw: float
    oil_heat_removal__kw: float
    power_use__kw: float
    suction_mass_flow__kg_s: float
    econ_mass_flow__kg_s: float = 0.0
    econ_evap_heat_removed__kw: float = 0.0
    compressor_speed__rpm: float | None = None
    sv_position__perc: float | None = None
    compressor_current__amps: float | None = None


class CompressorModel(ModelBase):
    name: str
    swept_vol__m3_rotation: float = 0.0
    motor_size__hp: int
    max_voltage: float = 460
    min_rpm: float = 1200
    max_rpm: float = 3600
    frac_superheat_oil_removed: float = 0.6
    motor_efficiency: float = 0.96781605
    motor_eff_intercept: float = 0.01107
    rpm_time_constant__s: float = 20
    sv_time_constant__s: float = 10

    @staticmethod
    def _capacity_model(sv: float):
        return (10 + sv * 30 / 85 + 1.6 * np.e ** (0.25 * (sv - 90))) / 0.648

    @staticmethod
    def _eff_model(cap):
        return 130 * cap / (6000 + 37.34 * cap + 1.84976 * cap**2 - 0.0123317 * cap**3)

    def get_discharge_state(
        self,
        simulation_point: CompressorTimePoint,
        suction_state: AmmoniaThermo,
        condensate_state: AmmoniaThermo,
        state: CompressorState | None = None,
    ) -> AmmoniaThermo:
        sv_position__perc = (
            state.sv_position__perc
            if state is not None
            else simulation_point.sv_position__perc
        )
        capacity__perc = self._capacity_model(sv_position__perc)
        isentropic_efficiency = self._eff_model(capacity__perc)
        compression = IrreversibleCompression(
            UnitConversions.psig_to_pascal(condensate_state.pressure__psig),
            max(0.3, min(1.0, isentropic_efficiency)),
            initial_state=suction_state,
        )
        return compression.run()

    def compute_mass_flow__kg_s(
        self,
        simulation_point: CompressorTimePoint,
        suction_state: AmmoniaThermo,
        discharge_state: AmmoniaThermo,
        state: CompressorState | None = None,
    ) -> float:
        compressor_speed__rpm = (
            state.compressor_speed__rpm
            if state is not None
            else simulation_point.compressor_speed__rpm
        )
        sv_position__perc = (
            state.sv_position__perc
            if state is not None
            else simulation_point.sv_position__perc
        )
        swept_volume__m3_s = self.swept_vol__m3_rotation * compressor_speed__rpm / 60
        max_mass_flow__kg_s = swept_volume__m3_s * suction_state.density__kg_m3
        capacity__perc = self._capacity_model(sv_position__perc)
        vol_flow_frac = capacity__perc / 100
        return max_mass_flow__kg_s * vol_flow_frac

    def get_shaft_work__kw(
        self,
        mass_flow__kg_s: float,
        discharge_state: AmmoniaThermo,
        suction_state: AmmoniaThermo,
    ) -> float:
        return mass_flow__kg_s * (
            discharge_state.enthalpy__kj_kg - suction_state.enthalpy__kj_kg
        )

    def get_evap_heat_removal__kw(
        self,
        mass_flow__kg_s: float,
        condensate_state: AmmoniaThermo,
        suction_state: AmmoniaThermo,
    ) -> float:
        enthalpy_change__kj_kg = (
            suction_state.enthalpy__kj_kg - condensate_state.enthalpy__kj_kg
        )
        return mass_flow__kg_s * enthalpy_change__kj_kg

    def get_oil_heat_removal__kw(
        self,
        mass_flow__kg_s: float,
        discharge_state: AmmoniaThermo,
    ) -> float:
        discharge_saturation_state = AmmoniaThermo(
            pressure__pa=UnitConversions.psig_to_pascal(discharge_state.pressure__psig),
            quality=1.0,
        )
        return (
            mass_flow__kg_s
            * (
                discharge_state.enthalpy__kj_kg
                - discharge_saturation_state.enthalpy__kj_kg
            )
            * self.frac_superheat_oil_removed
        )

    def get_power_use__kw(self, shaft_work__kw: float) -> float:
        motor_size__kw = UnitConversions.hp_to_kw(self.motor_size__hp)
        normalized_shaft_work = shaft_work__kw / motor_size__kw
        motor__perc = (
            self.motor_eff_intercept * (normalized_shaft_work > 0.01)
            + normalized_shaft_work / self.motor_efficiency
        )
        return motor__perc * motor_size__kw

    def run_compression(
        self,
        simulation_point: CompressorTimePoint,
        suction_state: AmmoniaThermo,
        condensate_state: AmmoniaThermo,
        state: CompressorState | None = None,
    ) -> CompressionReturn:
        if state is not None and state.compressor_speed__rpm < 1.0:
            discharge_state = suction_state
        elif state is None and simulation_point.compressor_speed__rpm < 1.0:
            discharge_state = suction_state
        else:
            discharge_state = self.get_discharge_state(
                simulation_point, suction_state, condensate_state, state
            )

        mass_flow__kg_s = self.compute_mass_flow__kg_s(
            simulation_point, suction_state, discharge_state, state
        )
        shaft_work__kw = self.get_shaft_work__kw(
            mass_flow__kg_s, discharge_state, suction_state
        )
        evap_heat_removed__kw = self.get_evap_heat_removal__kw(
            mass_flow__kg_s, condensate_state, suction_state
        )
        oil_heat_removal__kw = self.get_oil_heat_removal__kw(
            mass_flow__kg_s, discharge_state
        )
        power_use__kw = self.get_power_use__kw(shaft_work__kw)

        return CompressionReturn(
            name=simulation_point.name,
            date_time=simulation_point.date_time,
            shaft_work__kw=shaft_work__kw,
            evap_heat_removed__kw=evap_heat_removed__kw,
            oil_heat_removal__kw=oil_heat_removal__kw,
            power_use__kw=power_use__kw,
            suction_mass_flow__kg_s=mass_flow__kg_s,
            compressor_speed__rpm=state.compressor_speed__rpm if state else None,
            sv_position__perc=state.sv_position__perc if state else None,
            compressor_current__amps=simulation_point.compressor_current__amps,
        )

    def get_state_derivative(
        self,
        state: CompressorState,
        simulation_point: CompressorTimePoint,
    ) -> CompressorStateDerivative:
        dsv_dt = self.first_order_dynamics(
            target=simulation_point.sv_position__perc,
            state=state.sv_position__perc,
            time_constant__s=self.sv_time_constant__s,
        )
        drpm_dt = self.first_order_dynamics(
            target=simulation_point.compressor_speed__rpm,
            state=state.compressor_speed__rpm,
            time_constant__s=self.rpm_time_constant__s,
        )
        return CompressorStateDerivative(
            name=state.name,
            date_time=simulation_point.date_time,
            compressor_speed__rpm=drpm_dt,
            sv_position__perc=dsv_dt,
        )

    def get_state_from_measurement(
        self,
        measurement: CompressionReturn,
        simulation_point: CompressorTimePoint,
    ) -> CompressorState:
        return CompressorState(
            name=simulation_point.name,
            date_time=simulation_point.date_time,
            compressor_speed__rpm=measurement.compressor_speed__rpm,
            sv_position__perc=measurement.sv_position__perc,
        )

    def initialize(
        self, date_time: datetime
    ) -> tuple[CompressorState, CompressionReturn, CompressorTimePoint]:
        compressor_measurement = CompressionReturn(
            name=self.name,
            date_time=date_time,
            shaft_work__kw=0,
            evap_heat_removed__kw=0,
            oil_heat_removal__kw=0,
            power_use__kw=0,
            actual_power_use__kw=0,
            suction_mass_flow__kg_s=0,
            compressor_speed__rpm=0,
            sv_position__perc=0.0,
        )
        compressor_actuator = CompressorTimePoint(
            name=self.name,
            date_time=date_time,
            compressor_speed__rpm=0,
            sv_position__perc=0.0,
        )
        return (
            self.get_state_from_measurement(
                compressor_measurement, compressor_actuator
            ),
            compressor_measurement,
            compressor_actuator,
        )
