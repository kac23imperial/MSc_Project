from datetime import datetime, timedelta
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from ck_fridge_sim.control.control_base import (
    ControllerSetting,
    Disturbance,
    Disturbances,
)
from ck_fridge_sim.simulator.example import (
    configure_example_simulation_from_path,
)
from ck_fridge_sim.sources.sources import (
    Constant,
    Sinusoid,
    Step,
    SunAngleAltitude,
)
from ck_fridge_sim.solver.solvers import RK12
from ck_fridge_sim.thermo_tools.unit_conversions import UnitConversions
from reinforcement_learning.normaliser import Normaliser
from reinforcement_learning.pricer import Pricer

class RefrigerationEnv(gym.Env):
    """
    Custom Gym environment for controlling a refrigeration system.
    """
    # Action space constants
    SUCTION_PRESSURE_LOW = -1
    DISCHARGE_PRESSURE_LOW = 110
    EVAPORATOR_TEMP_LOW = -10
    
    SUCTION_PRESSURE_HIGH = 6
    DISCHARGE_PRESSURE_HIGH = 160
    EVAPORATOR_TEMP_HIGH = 5

    # Observation space constants
    LATEST_ROOM_TEMP_LOW = UnitConversions.fahrenheit_to_celsius(-20)
    ELECTRICITY_PRICE_LOW = 0
    TOTAL_REFRIGERATION_POWER_LOW = 0
    LATEST_AMBIENT_TEMP_LOW = 8
    TIME_OF_DAY_LOW = 0

    LATEST_ROOM_TEMP_HIGH = UnitConversions.fahrenheit_to_celsius(45)
    ELECTRICITY_PRICE_HIGH = 100
    TOTAL_REFRIGERATION_POWER_HIGH = 1000
    LATEST_AMBIENT_TEMP_HIGH = 50
    TIME_OF_DAY_HIGH = 24
    
    def __init__(self, solver_time_step_s=15, rl_time_step_min=60, episode_duration_min=1440):
        """
        Initializes the Refrigeration environment.

        Args:
            solver_time_step_s (int): The time step in seconds for the solver.
            rl_time_step_min (int): The time step in minutes for RL agent actions.
            episode_duration_min (int): Total duration of the episode in minutes, usually 24 hours.
        """
        super(RefrigerationEnv, self).__init__()

        # Initialize essential components
        self.normaliser = Normaliser()
        self.pricer = Pricer()

        # Environment configuration
        self.num_evaporators = 7
        self.solver_time_step_s = solver_time_step_s
        self.time_step_min = rl_time_step_min
        self.episode_duration_min = episode_duration_min
        self.step_start_time = None

        # Initialize environment components
        self.__init_datetime__()
        self.__init_simulator__()
        self.__init_action_space__()
        self.__init_observation_space__()
        
        # Initialize internal variables
        self.df = []
        self.accumulated_dfs = []
        self.step_counts = []
        self.average_temps = []
        self.electricity_costs = []
        self.actions = []
        self.initial_state_loaded = False

    def reset(self, seed=None, **kwargs):
        """
        Resets the simulation environment to its initial state.

        This method resets various internal variables, including accumulated data frames,
        total electricity cost, energy and temperature penalties, and time stamps.
        It also loads the initial state for the simulation.

        Args:
            seed (int, optional): A seed for the random number generator, if needed.
            **kwargs: Additional arguments passed to the reset method.

        Returns:
            tuple: The initial observation and any additional information.
        """
        # Reset flags and data structures
        self.initial_state_loaded = False
        self.accumulated_dfs = []
        self.full_df = pd.DataFrame()
        self.total_electricity_cost = 0

        # Reset tracking variables
        self.time_stamps = []
        self.step_costs = []

        #  Reset penalty tracking
        self.total_energy_penalty = 0
        self.total_temp_penalty = 0

        super().reset(seed=seed)

        # Load the initial simulation state
        self._load_initial_state()

        # Return the initial observation for the simulation
        return self.get_observation()
    
    def get_observation(self):
        """
        Generates the current observation for the RL environment.

        The observation includes the latest air temperature, total refrigeration power,
        electricity price, ambient temperature, and time of day, all normalized between
        -1 and 1.

        Returns:
            tuple:
                - norm_observation (np.array): Normalized observation array.
                - info (dict): Additional information including refrigeration power, electricity price, and latest air temperature.
        """
        # Extract the latest air temperature from the simulator's internal state
        latest_air_temp = next(
            state.air_temp__c for state in self.simulator.process_states.equipment_state_list if state.name == "room_1"
        )

        # Get total refrigeration power over the time interval
        total_refrigeration_power = self.get_total_refrigeration_power_interval(start_time=self.step_start_time, end_time=self.simulator.end_datetime)

        # Get the current electricity price
        electricity_price = self.pricer.get_price(self.simulation_time)

        # Get the latest ambient temperature from disturbances
        latest_ambient_temp = self.simulator.disturbances.get_disturbance_by_name("ambient_temp__c").value(self.simulation_time)

        # Get the normalized time of day
        time_of_day = self.normaliser.get_normalised_time_of_day(self.simulation_time)

        # Create the observation array
        observation = np.array([
            latest_air_temp,
            electricity_price,
            total_refrigeration_power,
            latest_ambient_temp,
            time_of_day
        ])

        # Info dictionary containing additional data for analysis
        info = {
        'total_refrigeration_power': total_refrigeration_power,
        'electricity_price': electricity_price,
        'latest_air_temp': latest_air_temp,
        }

        # Normalize the observation array
        norm_observation = self.normaliser.normalise_observation(
            observation,
            self.original_observation_space_low,
            self.original_observation_space_high
        )

        return norm_observation, info
    
    def step(self, action):
        """
        Executes one step in the simulation, applying the provided action and processing the results.

        Args:
            action (np.array): The action to be applied in the simulation.

        Returns:
            tuple: Contains normalized observation, reward, done flag, truncation flag (False), and additional info.
        """
        # Prepare the simulation environment with the given action
        self.step_simulation_setup(action)

        try:
            # Run the simulation with the action and retrieve results
            solution, measurement_list, actuator_list = self.simulator.simulate(action=self.denormalised_action, verbose=False)
        except RuntimeError as e:
            # Handle simulation errors, if any
            return self._handle_simulation_error(e)
        
        # Append the action to the list of actions
        self.actions.append(self.denormalised_action)
        # Save the current state of the simulation
        self._dump_current_state()
        # Update the simulation time
        self.simulation_time = self.simulator.process_states.simulation_time

        print(f"\n-------------------------{self.simulation_time}-------------------------\n")

        # Attempt to process the results with a limited number of retries
        tries = 0
        max_tries = 3

        while tries < max_tries:
            try:
                self._process_results(solution, measurement_list, actuator_list)
                break  # Exit the loop if successful
            except KeyError as e:
                tries += 1
                print(f"Attempt {tries}: Error processing results: {e}")
                if tries == max_tries:
                    print(f"Max tries ({max_tries}) exceeded. Aborting.")
        
        # Retrieve the current observation and additional info
        norm_observation, info = self.get_observation()

        # Calculate penalties for energy and temperature deviations
        energy_penalty, temp_penalty = self._calculate_penalties(info)

        # Update total penalties
        self.total_energy_penalty += energy_penalty
        self.total_temp_penalty += temp_penalty

        # Track the current step count
        current_step = len(self.step_counts) + 1
        self.step_counts.append(current_step)
        
        # Calculate the average room air temperature (in Celsius) over a specified time interval
        mean_room_air_temp_f = self.calculate_average_value(
            df=self.full_df,
            device_name="room_1",
            point_name="air_temp__f",
            start_time=self.step_start_time,
            end_time=self.simulator.end_datetime
        )
        mean_room_air_temp_c = UnitConversions.fahrenheit_to_celsius(mean_room_air_temp_f)

        # Append the calculated values to the tracking lists
        self.average_temps.append(mean_room_air_temp_c)
        self.electricity_costs.append( self.total_electricity_cost)

        # Track the step cost and energy usage
        self.time_stamps.append(self.simulation_time)
        self.step_costs.append(energy_penalty)
        self.total_electricity_cost += energy_penalty

        reward = self._calculate_reward(energy_penalty, temp_penalty)

        # Determine if the episode is done (end of simulation)
        buffer_period = timedelta(minutes=10)
        done = self.simulation_time + buffer_period >= self.end_datetime_episode

        if done:
            # Calculate total refrigeration power used over the entire episode
            total_refrigeration_power = self.get_total_refrigeration_power_interval(start_time=self.start_datetime, end_time=self.end_datetime_episode)
            print(f"Total refrigeration used over 24 hours: {total_refrigeration_power}")

            # Calculate the mean room air temperature over the full episode
            mean_room_air_temp_f = self.calculate_average_value(
                df=self.full_df,
                device_name="room_1",
                point_name="air_temp__f",
                start_time=self.start_datetime,
                end_time=self.end_datetime_episode
            )
            mean_room_air_temp_c = UnitConversions.fahrenheit_to_celsius(mean_room_air_temp_f)
            print(f"Mean temp over 24 hours: {mean_room_air_temp_c}")

            # Print total energy and temperature penalties for the episode
            print(f"Total energy penalty cost over 24 hours: {self.total_energy_penalty}")
            print(f"Total temperature penalty cost over 24 hours: {self.total_temp_penalty}")

            # Plot the results
            self._plot_results()

        return norm_observation, reward, done, False, info

######################### Helper methods for initializing environment component #########################
    def __init_datetime__(self):
        """
        Initializes the simulation start and end datetimes for the episode.
        """
        self.start_datetime = self.pricer.get_start_datetime()
        self.end_datetime_episode = self.start_datetime + timedelta(minutes=self.episode_duration_min)

    def __init_simulator__(self):
        """
        Configures the simulator using the process and control configurations and initializes 
        disturbances and settings.

        The simulator is set up to handle the refrigeration process, including disturbances and controls.
        """
        self.simulator = configure_example_simulation_from_path(
            process_config_path=Path("simulation_payload/simple_flowsheet_config.json"),
            control_config_path=Path("simulation_payload/simple_control_config.yaml"),
            disturbances=self.create_disturbances(),
            settings=self.create_settings(),
            start_datetime=self.start_datetime,
            end_datetime=self.start_datetime + timedelta(minutes=self.time_step_min),
            solver=RK12(max_step__s=self.solver_time_step_s, relative_tolerance=0.01),
        )

    def __init_action_space__(self):
        """
        Initializes the action space for the RL agent.

        The action space includes control values for suction pressure, discharge pressure, 
        and evaporator temperature. These are normalized between -1 and 1.
        """
        # Create action space boundary arrays
        action_space_low_values = [
            self.SUCTION_PRESSURE_LOW,
            self.DISCHARGE_PRESSURE_LOW,
            self.EVAPORATOR_TEMP_LOW
        ]
        
        action_space_high_values = [
            self.SUCTION_PRESSURE_HIGH,
            self.DISCHARGE_PRESSURE_HIGH,
            self.EVAPORATOR_TEMP_HIGH
        ]
        
        # Store the original action space boundaries
        self.original_action_space_low = np.array(action_space_low_values)
        self.original_action_space_high = np.array(action_space_high_values)
        
        # Define the normalized action space used by the RL agent
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(action_space_low_values),),
            dtype=np.float64
        )

    def __init_observation_space__(self):
        """
        Initializes the observation space for the RL environment.

        The observation space is bounded by the predefined limits for room temperature,
        electricity price, refrigeration power, ambient temperature, and time of day.
        The space is normalized to be between -1 and 1 for each variable.
        """
        # Create observation space boundary arrays
        observation_space_low_values = [
            self.LATEST_ROOM_TEMP_LOW,
            self.ELECTRICITY_PRICE_LOW,
            self.TOTAL_REFRIGERATION_POWER_LOW,
            self.LATEST_AMBIENT_TEMP_LOW,
            self.TIME_OF_DAY_LOW
        ]
        
        observation_space_high_values = [
            self.LATEST_ROOM_TEMP_HIGH,
            self.ELECTRICITY_PRICE_HIGH,
            self.TOTAL_REFRIGERATION_POWER_HIGH,
            self.LATEST_AMBIENT_TEMP_HIGH,
            self.TIME_OF_DAY_HIGH
        ]
        
        # Store the original observation space boundaries
        self.original_observation_space_low = np.array(observation_space_low_values)
        self.original_observation_space_high = np.array(observation_space_high_values)
        
        # Define the normalized observation space used by the RL agent
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(observation_space_low_values),),
            dtype=np.float64
        )


######################### Control Settings & Disturbances #########################
    def create_disturbances(self):
        """
        Creates a list of disturbances.

        The disturbances include ambient temperature, ambient relative humidity,
        machine room temperature, sun angle altitude, and cloud cover - each
        modelled as either a sinusoidal or step function.

        Returns:
            Disturbances: A Disturbances object containing the list of defined disturbances.
        """
        disturbance_list = [
            Disturbance(
                name="ambient_temp__c",
                source=Sinusoid(
                    amplitude=8.0,
                    offset=16.0,
                    shift_time_str=self.start_datetime.replace(
                        hour=9, minute=0, second=0, microsecond=0
                    ).isoformat(),
                ),
            ),
            Disturbance(
                name="ambient_rh__perc",
                source=Sinusoid(
                    amplitude=10.0,
                    offset=50.0,
                    shift_time_str=self.start_datetime.replace(
                        hour=21, minute=0, second=0, microsecond=0
                    ).isoformat(),
                ),
            ),
            Disturbance(
                name="machine_room_temp__c",
                source=Step(
                    start_value=25.0,
                    end_value=15.0,
                    step_time_str=self.start_datetime.replace(
                        hour=12, minute=0, second=0, microsecond=0
                    ).isoformat(),
                ),
            ),
            Disturbance(
                name="sun_angle_altitude",
                source=SunAngleAltitude(
                    latitude__deg=32.5494799,
                    longitude__deg=-116.9827151,
                ),
            ),
            Disturbance(
                name="cloud_cover__perc",
                source=Step(
                    start_value=0.0,
                    end_value=50.0,
                    step_time_str=self.start_datetime.replace(
                        hour=14, minute=0, second=0, microsecond=0
                    ).isoformat(),
                ),
            ),
        ]
        return Disturbances(disturbance_list=disturbance_list)

    def create_settings(self):
        """
        Creates a list of controller settings.

        The settings include discharge and suction pressure setpoints, seven
        evaporator temperature setpoints, and HPR level. These settings are
        used to configure controllers in the refrigeration system.

        Returns:
            list: A list of ControllerSetting objects.
        """
        initial_discharge_pressure__psig = 115

        discharge_pressure_setpoint = ControllerSetting(
            name="EC-1|pressure_setpoint",
            controller_name="condenser_fan_controller",
            setting_name="setpoint",
            value=initial_discharge_pressure__psig,
        )

        suction_pressure_setpoint = ControllerSetting(
            name="LTR-1|pressure_setpoint",
            controller_name="lss_1",
            setting_name="pi_controller",
            nested_setting_names=["setpoint"],
            value=3,
        )

        evaporator_setpoints = [
            ControllerSetting(
                name=f"{evap}|temp_setpoint",
                controller_name=f"evaporator_{i+1}_controller",
                setting_name="setpoint",
                value=-17.0,
            )
            for i, evap in enumerate([f"Evap AU-{i+1}" for i in range(self.num_evaporators)])
        ]

        hpr_setpoint = ControllerSetting(
            name="HPR-1|level_setpoint",
            controller_name="hpr_level_control",
            setting_name="setpoint",
            value=33.0,
        )

        settings = [
            suction_pressure_setpoint,
            discharge_pressure_setpoint,
            hpr_setpoint,
        ] + evaporator_setpoints

        return settings

######################### Other Helper Methods #########################
    def get_total_refrigeration_power_interval(self, start_time: pd.Timestamp, end_time: pd.Timestamp):
        """
        Calculates the total refrigeration power consumed over a specific time interval.

        Args:
            start_time (pd.Timestamp): The start time of the interval.
            end_time (pd.Timestamp): The end time of the interval.

        Returns:
            float: Total energy consumed (in kWh) by the refrigeration system over the interval.
        """
        # Ensure that full_df has the correct datetime index and is sorted
        self.full_df.index = pd.to_datetime(self.full_df.index)
        self.full_df = self.full_df.sort_index()

        # Filter for the specific power column
        power_df = self.full_df.filter(like="dual_cool").filter(like="total_refrigeration_power__kw")

        # Check if the DataFrame is empty
        if power_df.empty:
            return 0
        
        # Slice the DataFrame for the given time interval
        interval_df = power_df.loc[start_time:end_time]
        interval_df = interval_df.copy()

        # Calculate the time difference between consecutive rows in hours
        interval_df['time_diff'] = interval_df.index.to_series().diff().dt.total_seconds() / 3600.0

        # Calculate energy consumed in each interval (kW * hours = kWh)
        interval_df.loc[:, 'energy_kwh'] = interval_df['dual_cool | total_refrigeration_power__kw'] * interval_df['time_diff']

        # Sum up the energy consumed over the interval
        total_refrigeration_power = interval_df['energy_kwh'].sum()

        return total_refrigeration_power

    def calculate_average_value(self, df: pd.DataFrame, device_name: str, point_name: str, start_time: pd.Timestamp, end_time: pd.Timestamp) -> float:
        """
        Calculates average value for a specific device and point name over a given time range.

        Args:
            df (pd.DataFrame): DataFrame containing the relevant data.
            device_name (str): The name of the device to filter.
            point_name (str): The name of the point to filter.
            start_time (pd.Timestamp): The start time of the time range.
            end_time (pd.Timestamp): The end time of the time range.

        Returns:
            float: The average value of the filtered data. Returns 0.0 if the data is empty.
        """
        # Filter the DataFrame to extract relevant data for the device and point
        relevant_data = df.filter(like=device_name).filter(like=point_name)
        relevant_data = relevant_data.loc[start_time:end_time]
        average_value = relevant_data.mean().values[0]
        return average_value if not relevant_data.empty else 0.0

    def calculate_temp_penalty(self, T_r, T_upper=UnitConversions.fahrenheit_to_celsius(0), lambda_=10):
        """
        Calculates temperature penalty based on the deviation from the upper temperature limit.
        The balancing parameter can be tweaked here.

        Args:
            T_r (float): The current temperature reading.
            T_upper (float, optional): The upper temperature limit. Defaults to 0°C (converted from Fahrenheit).
            lambda_ (float, optional): A scaling factor for the penalty calculation. Defaults to 10.

        Returns:
            float: The calculated temperature penalty.
        """
        if T_r > T_upper:
            penalty = 5 + lambda_ * (-np.exp(-(T_r - T_upper)) + 1)
            print(f"Final temperature penalty: {penalty}")
        else:
            penalty = 0
            print(f"Final temperature penalty: {penalty}")
        return penalty

    def calculate_energy_penalty(self, info, end_time: pd.Timestamp):
        """
        Calculates the energy penalty based on energy consumption and electricity price.

        Args:
            info (dict): A dictionary containing 'total_refrigeration_power'.
            end_time (pd.Timestamp): The time at which the energy penalty is calculated.

        Returns:
            float: The calculated energy penalty (cost of electricity).
        """
        time_step_hours = self.time_step_min / 60

        # Calculate total energy consumption in kWh (assuming total_refrigeration_power is in kW)
        total_energy_consumption_MWh = (info['total_refrigeration_power'] * time_step_hours)/1000

        # Get the current electricity price
        electricity_price = self.pricer.get_price(end_time)

        # Calculate the energy penalty (cost of electricity)
        energy_penalty = electricity_price * total_energy_consumption_MWh
        print(f"Energy Penalty: {energy_penalty}")

        return energy_penalty

    def _load_initial_state(self):
        """
        Loads the initial simulation state from JSON files and updates the simulation times.

        The initial process and control states are loaded, and the simulation time is set to the start_datetime.
        The current state is then saved to JSON.

        Returns:
            None
        """
        # Load initial process and control states from JSON
        self.simulator.load_state_from_json(
            Path("simulation_payload/initial_process_states.json"),
            Path("simulation_payload/initial_control_states.json")
        )

        # Update the datetime in each state in process_states and control_states
        self.simulator.process_states.simulation_time = self.start_datetime

        for state in self.simulator.process_states.equipment_state_list:
            state.date_time = self.start_datetime

        for control_state in self.simulator.control_states.control_states:
            control_state.date_time = self.start_datetime

        self._dump_current_state()

        self.simulation_time = self.start_datetime

        self.initial_state_loaded = True

    def _load_simulation_state(self):
        """
        Loads appropriate simulation state based on whether it is the initial state or
        a normal state after initial setup.

        If the initial state has been loaded, it loads the normal process and control states
        from JSON files for subsequent simulation steps. Otherwise, it loads the initial state.
        """
        if self.initial_state_loaded:
            # Load the normal state for subsequent steps
            self.simulator.load_state_from_json(
                Path("simulation_payload/process_states.json"),
                Path("simulation_payload/control_states.json")
            )
        else:
            # Load the initial state (already handled in reset)
            self._load_initial_state()

    def _dump_current_state(self):
        """
        Dumps the current process and control states of the simulation to JSON files.
        
        This is typically called after each step to ensure the simulation state is saved.
        """
        self.simulator.dump_state_to_json(
            Path("simulation_payload/process_states.json"),
            Path("simulation_payload/control_states.json")
        )

    def step_simulation_setup(self, action):
        """
        Prepares the simulation for the next step by denormalizing the action and setting up state variables.

        Args:
            action (np.array): The action to be denormalized and applied to the simulation.

        Returns:
            None
        """
        # Denormalize the action to its real-world values
        self.denormalised_action = self.normaliser.denormalise_action(action, self.original_action_space_low, self.original_action_space_high)

        # Set evaporator temperatures based on the base temperature (third action value)
        base_temp = self.denormalised_action[2]
        evaporator_temps = [UnitConversions.fahrenheit_to_celsius(base_temp + 0.5 * i) for i in range(self.num_evaporators)]

        # Update the action array with the evaporator temperatures
        self.denormalised_action = np.concatenate([self.denormalised_action[:2], evaporator_temps])
        self.last_action = self.denormalised_action

        # Load the simulation state for the next step and set the start time
        self._load_simulation_state()
        self.step_start_time = self.simulation_time
        self.simulator.end_datetime = self.simulation_time + timedelta(minutes=self.time_step_min)

    def _handle_simulation_error(self, error):
        """
        Handles errors during simulation by logging the error, applying a heavy penalty,
        and resetting the environment.

        Args:
            error (Exception): The error that occurred during the simulation.

        Returns:
            tuple: Contains the reset observation, reward, done flag, False (indicating no truncation), and info.
        """
        print(f"Simulation error occurred: {error}")

        # Apply a heavy penalty when an error occurs
        reward = -600
        print(f"Final reward: {reward}")
        done = True
        self._plot_results()
        obs, info = self.reset()
        return obs, reward, done, False, info

    def _process_results(self, solution, measurement_list, actuator_list):
        """
        Processes and accumulates the results from simulation by consolidating data
        from the provided solution, measurement, and actuator lists.

        Args:
            solution (Solution): The solution object from the simulator.
            measurement_list (list): List of measurement names.
            actuator_list (list): List of actuator names.

        Returns:
            None
        """
        df = self.simulator.process_results(solution, measurement_list, actuator_list)
        df = df.loc[:, ~df.columns.duplicated()]
        self.accumulated_dfs.append(df)
        self.full_df = pd.concat(self.accumulated_dfs, axis=0)

        if not isinstance(self.simulation_time, pd.Timestamp):
            self.simulation_time = pd.to_datetime(self.simulation_time)

        if self.simulation_time not in df.index:
            closest_time = df.index[df.index.get_indexer([self.simulation_time], method='nearest')][0]
            self.simulation_time = closest_time
    
    def _plot_results(self):
        """
        Plots simulation results and exports relevant data to CSV files.
        """
        fig1 = self.simulator.plot_results(
            self.full_df,
            device_names=["room_1"],
            point_names=[
                "air_temp__f",
            ],
        )

        fig1.show()

        fig2 = self.simulator.plot_results(
            self.full_df,
            device_names=["dual_cool"],
            point_names=[
                "total_refrigeration_power__kw",
            ],
        )

        fig2.show()

        # Filter relevant columns
        df_filtered = self.full_df[['dual_cool | total_refrigeration_power__kw']].resample('10min').mean()
        df_filtered_2 = self.full_df[['room_1 | air_temp__f']].resample('10min').mean()

        # Export the DataFrame to a CSV file
        df_filtered.to_csv('results/data_DDPG_6_hours/total_refrigeration_power.csv', index=True)
        df_filtered_2.to_csv('results/data_DDPG_6_hours/air_temp__f.csv', index=True)

    def _calculate_penalties(self, info):
        """
        Calculates energy and temperature penalties for reward function.
        
        Args:
            info (dict): Information containing latest air temperature and other data.
        
        Returns:
            tuple: A tuple containing energy and temperature penalties.
        """
        energy_penalty = self.calculate_energy_penalty(
            info = info,
            end_time=self.simulation_time,
        )

        temp_penalty = self.calculate_temp_penalty(info['latest_air_temp'])
        return energy_penalty, temp_penalty
    
    def _calculate_reward(self, energy_penalty, temp_penalty):
        """
        Calculates the reward based on energy and temperature penalties.

        Args:
            energy_penalty (float): The calculated energy penalty.
            temp_penalty (float): The calculated temperature penalty.
        
        Returns:
            float: The total reward value.
        """
        reward = -energy_penalty - temp_penalty
        print(f"Final reward: {reward}")
        return reward