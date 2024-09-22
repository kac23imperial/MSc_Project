import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from reinforcement_learning.refrigeration_env import RefrigerationEnv

# Initialize the environment
env = RefrigerationEnv()
env.reset()

# Custom Callback to print progress after each step
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
models_dir = "models/DDPG_6_hours"
logdir = "logs"

# Ensure directories exist
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)

# Set device for training (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained DDPG model
model_path = os.path.join(models_dir, "400.zip")
model = DDPG.load(model_path, env=env, device=device)

# Simulation settings
TIMESTEPS_LOOP = 24

# Create a print callback for tracking progress
print_callback = PrintCallback(total_timesteps=TIMESTEPS_LOOP)
callback = CallbackList([print_callback])

# Continue learning or simulating using the loaded model
model.learn(
    total_timesteps=TIMESTEPS_LOOP,
    reset_num_timesteps=False,
    tb_log_name="DDPG_6_hours",
    callback=callback
)