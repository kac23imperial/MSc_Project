from datetime import datetime

from pydantic import BaseModel

from ck_fridge_sim.control.control_base import (
    Actuators,
    ControllerSetting,
    ControlSystemState,
)
from ck_fridge_sim.control.controllers import Controller
from ck_fridge_sim.equipment_models import Measurement


class ControlSystem(BaseModel):
    controllers: list[Controller]

    def action(
        self,
        control_states: ControlSystemState,
        measurements: list[Measurement],
        actuators: Actuators,
        time: datetime | None = None,
    ) -> None:
        """
        Actuators and control_states are updated in place with this method
        """
        for controller in self.controllers:
            controller.action(control_states, measurements, actuators, time)

    def update_settings(
        self, settings: list[ControllerSetting], time: datetime
    ) -> None:
        """
        settings are updated in place with this method
        """
        for setting in settings:
            controller = [
                controller
                for controller in self.controllers
                if controller.name == setting.controller_name
            ]
            if len(controller) == 0:
                raise ValueError(f"Controller {setting.controller_name} not found")
            elif len(controller) > 1:
                raise ValueError(
                    f"Multiple controllers with name {setting.controller_name}"
                )

            controller = controller[0]
            setting.update_value(time)
            controller.update_setting(setting)
