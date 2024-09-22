import os
import cProfile
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from reinforcement_learning.refrigeration_env import RefrigerationEnv

# Profile the entire script
profiler = cProfile.Profile()
profiler.enable()

# Create and reset the environment
env = RefrigerationEnv()
env.reset()

# Custom callback to print progress after each step
class PrintCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super(PrintCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.global_step_count = 0  # Track steps across all episodes

    def _on_step(self):
        self.global_step_count += 1
        progress = (self.global_step_count / self.total_timesteps) * 100
        print(f"Step {self.global_step_count}/{self.total_timesteps} - ({progress:.2f}%)")
        return True

# Directory paths for saving models and logs
models_dir = "models/TD3"
logdir = "logs"

# Ensure directories exist
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)

# Set device for training (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the DDPG model
model = DDPG('MlpPolicy', env, verbose=1, batch_size=64, tensorboard_log=logdir)

# Simulation settings
TIMESTEPS_LOOP = 24
TIMESTEPS = 200
TOTAL_TIMESTEPS = TIMESTEPS_LOOP * TIMESTEPS  # Example: 4800 steps

# Create a print callback for tracking progress
print_callback = PrintCallback(total_timesteps=TOTAL_TIMESTEPS)
callback = CallbackList([print_callback])

# Training loop with model saving after each loop
for i in range(TIMESTEPS_LOOP):
    model.learn(
        total_timesteps=TIMESTEPS,
        reset_num_timesteps=False,
        tb_log_name="TD3_NORMAL",
        callback=callback
    )
    # Save the model at the end of each loop
    model.save(f"{models_dir}/{TIMESTEPS * (i+1)}")

# Disable profiler and print results
profiler.disable()
profiler.print_stats()
