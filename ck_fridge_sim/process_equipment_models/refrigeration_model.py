from datetime import datetime
from typing import Literal

import numpy as np

from ck_fridge_sim.base_model_classes import (
    MeasurementBase,
    ModelBase,
    StateBase,
    StateDerivativeBase,
)
from ck_fridge_sim.thermo_tools.humid_air_properties import HumidAirProperties
from ck_fridge_sim.thermo_tools.unit_conversions import UnitConversions


class RoomState(StateBase):
    kind: Literal["refrigerated_room"] = "refrigerated_room"
    air_temp__c: float
    external_product_temp__c: float
    internal_product_temp__c: float


class RoomStateDerivative(RoomState, StateDerivativeBase):
    pass


class RoomMeasurement(MeasurementBase):
    kind: Literal["refrigerated_room"] = "refrigerated_room"
    air_temp__f: float


class RefrigeratedRoom(ModelBase):
    name: str
    evaporators: list[str]
    length__m: float
    width__m: float
    height__m: float
    rh__perc: float  # Relative humidity of the room in %
    product_mass_holdup__kg: float
    external_product__frac: float = (
        0.1  # Fraction of the product that is being actively cooled to temperature
    )
    external_product_air_htc__kw_m2_k: float = 0.025  # Cooling htc
    internal_product_htc__kw_m2_k: float = 0.0025
    product_specific_surface_area__m2_kg: float = 0.024
    product_heat_capacity__kj_kg_k: float = 4.186
    ambient_air_pressure__pa: float = 101325.0  # Ambient air pressure in Pa
    htc__kw_m2_k: float = 0.0025
    max_sun_temp_adjustment__c: float = 5.0
    default_temp__f: float = -1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ha_props = HumidAirProperties(self.ambient_air_pressure__pa)

    @property
    def volume_of_air__m3(self) -> float:
        return self.length__m * self.width__m * self.height__m

    @property
    def external_product_mass__kg(self):
        return self.product_mass_holdup__kg * self.external_product__frac

    @property
    def internal_product_mass__kg(self):
        return self.product_mass_holdup__kg - self.external_product_mass__kg

    @property
    def thermal_mass_of_external_product__kj_k(self):
        return self.product_heat_capacity__kj_kg_k * self.external_product_mass__kg

    @property
    def thermal_mass_of_internal_product__kj_k(self):
        return self.product_heat_capacity__kj_kg_k * self.internal_product_mass__kg

    def thermal_mass_of_air__kj_k(self, temp_room__c: float) -> float:
        room_air_density__kg_m3 = self._ha_props.get_density(
            temp_room__c, self.rh__perc
        )
        room_air_heat_capacity__kj_kg_k = self._ha_props.get_heat_capacity(
            temp_room__c, self.rh__perc
        )
        return (
            self.volume_of_air__m3
            * room_air_density__kg_m3
            * room_air_heat_capacity__kj_kg_k
        )

    def get_heat_load__kw(
        self,
        state: RoomState,
        ambient_temp__c: float,
        ambient_rh__perc: float,
        sun_angle__deg: float | None,
        cloud_cover__perc: float | None,
        date_time: datetime,
    ) -> float:
        sun_temp_adjustment__c = 0
        if sun_angle__deg and cloud_cover__perc:
            sun_temp_adjustment__c = (
                self.max_sun_temp_adjustment__c
                * max(0, np.sin(sun_angle__deg / 180 * np.pi))
                * (100 - cloud_cover__perc)
                / 100
            )

        transmission_load__kw = (
            self.htc__kw_m2_k
            * self.length__m
            * self.width__m
            * (ambient_temp__c + sun_temp_adjustment__c - state.air_temp__c)
        )
        infiltration_load__kw = (
            (200 + 1.5 * ambient_rh__perc)
            if date_time.isoweekday() < 6
            and date_time.hour >= 6
            and date_time.hour <= 17
            else (75 + 0.5 * ambient_rh__perc)
        )

        miscellaneous_load__kw = 100
        return transmission_load__kw + infiltration_load__kw + miscellaneous_load__kw

    def get_state_derivative(
        self, state: RoomState, heat_load__kw: float, evap_q__kw: float
    ) -> RoomStateDerivative:
        external_product_heat_transfer__kw = (
            self.external_product_air_htc__kw_m2_k
            * self.product_specific_surface_area__m2_kg
            * self.product_mass_holdup__kg
            * (state.external_product_temp__c - state.air_temp__c)
        )

        internal_product_heat_transfer__kw = (
            self.internal_product_htc__kw_m2_k
            * self.product_specific_surface_area__m2_kg
            * self.product_mass_holdup__kg
            * (state.internal_product_temp__c - state.external_product_temp__c)
        )

        air_temp_deriv__c_s = (
            heat_load__kw - evap_q__kw + external_product_heat_transfer__kw
        ) / self.thermal_mass_of_air__kj_k(state.air_temp__c)

        external_temp_deriv__c_s = (
            internal_product_heat_transfer__kw - external_product_heat_transfer__kw
        ) / self.thermal_mass_of_external_product__kj_k

        internal_temp_deriv__c_s = (
            -internal_product_heat_transfer__kw
            / self.thermal_mass_of_internal_product__kj_k
        )

        return RoomStateDerivative(
            name=self.name,
            date_time=state.date_time,
            air_temp__c=air_temp_deriv__c_s,
            external_product_temp__c=external_temp_deriv__c_s,
            internal_product_temp__c=internal_temp_deriv__c_s,
        )

    def get_state_from_measurement(
        self,
        measurement: RoomMeasurement,
        simulation_point: None = None,
    ) -> RoomState:
        return RoomState(
            name=measurement.name,
            date_time=measurement.date_time,
            air_temp__c=UnitConversions.fahrenheit_to_celsius(measurement.air_temp__f),
            external_product_temp__c=UnitConversions.fahrenheit_to_celsius(
                measurement.air_temp__f
            ),
            internal_product_temp__c=UnitConversions.fahrenheit_to_celsius(
                measurement.air_temp__f
            ),
        )

    def initialize(self, date_time: datetime) -> tuple[RoomState, RoomMeasurement]:
        room_measurement = RoomMeasurement(
            name=self.name, date_time=date_time, air_temp__f=self.default_temp__f
        )
        return self.get_state_from_measurement(room_measurement), room_measurement

    @classmethod
    def default_room(cls, name: str, evaporator_list: list[str]) -> "RefrigeratedRoom":
        return cls(
            name=name,
            evaporators=evaporator_list,
            length__m=96.4,
            width__m=96.4,
            height__m=18.3,
            rh__perc=70,
            product_mass_holdup__kg=20000000,
        )
