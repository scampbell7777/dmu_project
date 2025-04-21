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
    return Monitor(gym.make("Hopper-v5"))

env = DummyVecEnv([make_env])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

# Create a custom policy with appropriate network size
policy_kwargs = {
    "net_arch": [
        {"pi": [128, 128], "vf": [128, 128]}  # Slightly smaller network for Hopper
    ],
    "activation_fn": nn.ReLU  # ReLU activation
}

model_path = "ppo_hopper"
vec_normalize_path = "vec_normalize_hopper.pkl"

if False: #os.path.exists(vec_normalize_path):
    print(f"Loading environment normalization stats from {vec_normalize_path}")
    env = VecNormalize.load(vec_normalize_path, env)
    # Keep training normalization statistics
    env.training = True
    env.norm_reward = True
else:
    print("No saved normalization stats found, creating new normalized environment")
    env = VecNormalize(env, norm_obs=False, norm_reward=False)

if False: # os.path.exists(model_path + ".zip"):  # Make sure to check with .zip extension
    print("Loading pretrained model...")
    model = PPO.load(model_path, env=env)
else:
    print("Creating new model...")
    model = PPO(
        policy="MlpPolicy",
        policy_kwargs=policy_kwargs,
        env=env,
        learning_rate=3e-4,
        n_steps=1024,     # Reduced from 2048 for Hopper
        batch_size=64,    # Reduced from 256 for Hopper
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1
    )

# Train the model
model.learn(total_timesteps=500_000)  # Reduced from 1M for Hopper as it's simpler
env.training = False
env.norm_reward = False
print(evaluate_policy(model, env, n_eval_episodes=10))
# Save the model and normalization stats
model.save("ppo_hopper")
env.save("vec_normalize_hopper.pkl")

# For evaluation and testing
# eval_env = DummyVecEnv([make_env])
# eval_env = VecNormalize.load("vec_normalize_hopper.pkl", eval_env)
# eval_env.training = False
# eval.norm_obs = True
# eval_env.norm_reward = False

# # Evaluate the trained agent
# mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)  # Reduced from 100
# print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Optional: Visualize agent's performance
# Uncomment below to view the trained agent
"""
obs = eval_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = eval_env.step(action)
    eval_env.render()
    if dones:
        obs = eval_env.reset()
"""