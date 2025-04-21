import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch.nn as nn
from stable_baselines3.common.monitor import Monitor
import time
import os

# Create environment
def make_env():
    return Monitor(gym.make("Ant-v5"))

env = DummyVecEnv([make_env])
env = VecNormalize(env, norm_obs=False, norm_reward=True)

# Create a custom policy with appropriate network size
policy_kwargs = {
    "net_arch": [
        {"pi": [256, 256, 256], "vf": [256, 256, 256]}
    ],
    "activation_fn": nn.ReLU  # ReLU activation
}

model = PPO(
    policy="MlpPolicy",
    policy_kwargs=policy_kwargs,
    env=env,
    learning_rate=3e-4,
    n_steps=2048,    
    batch_size=512,   
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1,
    clip_range_vf=0.2,   # Added value function clipping
    ent_coef=0.005,      # Added entropy coefficient for exploration
    max_grad_norm=0.5,   # Added gradient clipping
)

# Train the model
model.learn(total_timesteps=1_000_000)
env.training = False
env.norm_reward = False
print(evaluate_policy(model, env, n_eval_episodes=10))