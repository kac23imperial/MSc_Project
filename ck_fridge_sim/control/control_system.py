from datetime import datetime

import numpy as np
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
            # print(f"Controller '{controller.name}': Setting '{setting.setting_name}' updated to value {setting.value}")
            controller.update_setting(setting)
            # Debug statement to print the updated setting value along with the controller name
            # print(f"[DEBUG] Updated setting for controller '{controller.name}': {setting.name} = {setting.value}")


    def update_settings_by_action( # ADDED
        self, settings: list[ControllerSetting], action: np.ndarray, time: datetime
    ) -> None:
        """
        Update settings by an action `a_t`, rather than their Source function at timestep `time`
        """
        # print(f"Number of settings to update: {len(settings)}")
        # print(f"Size of action array: {len(action)}")

        action_index = 0  # Initialize separate index for the action array

        for i, setting in enumerate(settings[:-1]):
            if i == 2:  # Skip the second setting (HPR controller)
                continue

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
            setting.value = action[action_index]
            # print(f"Controller '{controller.name}': Setting '{setting.setting_name}' updated to value {setting.value}")
            controller.update_setting(setting)

            action_index += 1  # Increment the action index only if a setting was updated


        # Separate pass for the HPR controller (second setting)
        hpr_setting = settings[2]  # The second element is the HPR controller
        controller = [
            controller
            for controller in self.controllers
            if controller.name == hpr_setting.controller_name
        ][0]
        # print(f"SIM: Using `update_settings` for HPR controller: {controller.name}")
        # Use time from the environment or simulation time
        self.update_settings([hpr_setting], time)

            # print(f"{i}:  SIM: controller: {controller}")
            # setting.value = action[i]
            # # Debugging
            # print(f"SIM: Setting value: {setting.value}")
            # controller.update_setting(setting)
            # print(f"{i}:  SIM: LAST: {controller.update_setting(setting)}")

            # Debugging: print the updated setting value along with the controller name
            # print(f"SIM: [DEBUG] Updated setting for controller '{controller.name}': {setting.name} = {setting.value}")

