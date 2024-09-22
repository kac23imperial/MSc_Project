import numpy as np

class Normaliser:
    """
    A class to handle normalization and denormalization of actions and observations for reinforcement learning environments.

    This class provides methods to normalize and denormalize actions and observations, 
    apply action constraints, and handle time of day normalization in a simulation environment.
    """

    def get_normalised_time_of_day(self, simulation_time):
        """
        Normalizes the time of day to a range of [-1, 1], where -1 corresponds to 00:00 and 1 corresponds to 23:59.

        Args:
            simulation_time (datetime): The current simulation time.

        Returns:
            float: The normalized time of day in the range [-1, 1].
        """
        return 2 * (simulation_time.hour / 24.0 + simulation_time.minute / 1440.0) - 1

    def normalise_action(self, action, action_space_low, action_space_high):
        """
        Normalizes an action to the range [-1, 1] based on the action space limits.

        Args:
            action (np.array): The action to be normalized.
            action_space_low (np.array): The lower bounds of the action space.
            action_space_high (np.array): The upper bounds of the action space.

        Returns:
            np.array: The normalized action in the range [-1, 1].
        """
        return (
            2 * (action - action_space_low) / (action_space_high - action_space_low) - 1
        )

    def denormalise_action(self, norm_action, action_space_low, action_space_high):
        """
        Denormalizes an action from the range [-1, 1] to its original scale based on the action space limits.

        Args:
            norm_action (np.array): The normalized action in the range [-1, 1].
            action_space_low (np.array): The lower bounds of the action space.
            action_space_high (np.array): The upper bounds of the action space.

        Returns:
            np.array: The denormalized action in the original action space.
        """
        return (norm_action + 1) * (
            action_space_high - action_space_low
        ) / 2 + action_space_low

    def apply_action_constraints(self, action, action_space):
        """
        Clamps the action values to be within the action space bounds.

        Args:
            action (np.array): The action to be constrained.
            action_space (gym.spaces.Box): The action space containing the valid range.

        Returns:
            np.array: The constrained action.
        """
        return np.clip(action, action_space.low, action_space.high)
    
    def normalise_observation(self, observation, original_observation_space_low, original_observation_space_high):
        """
        Normalizes an observation to the range [-1, 1], except for the time of day which is already normalized.

        Args:
            observation (np.array): The observation to be normalized.
            original_observation_space_low (np.array): The lower bounds of the observation space.
            original_observation_space_high (np.array): The upper bounds of the observation space.

        Returns:
            np.array: The normalized observation.
        """
        observation_except_time = observation[:-1]  # Exclude the last element (time of day)
        normalized_except_time = (
            2
            * (observation_except_time - original_observation_space_low[:-1])
            / (original_observation_space_high[:-1] - original_observation_space_low[:-1])
            - 1
        )

        normalized_time_of_day = observation[-1]  # Keep the time of day normalization the same
        
        return np.append(normalized_except_time, normalized_time_of_day)

    def denormalise_observation(self, norm_observation, observation_space_low, observation_space_high):
        """
        Denormalizes an observation from the range [-1, 1] back to its original scale.

        Args:
            norm_observation (np.array): The normalized observation.
            observation_space_low (np.array): The lower bounds of the observation space.
            observation_space_high (np.array): The upper bounds of the observation space.

        Returns:
            np.array: The denormalized observation in the original observation space.
        """
        return (norm_observation + 1) * (
            observation_space_high - observation_space_low
        ) / 2 + observation_space_low