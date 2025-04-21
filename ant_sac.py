import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch.nn as nn
from stable_baselines3.common.monitor import Monitor
import os

# Create environment
def make_env():
    return Monitor(gym.make("Ant-v5"))

env = DummyVecEnv([make_env])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

# Custom policy network architecture
policy_kwargs = {
    "net_arch": {
        "pi": [256, 256, 256],  # Policy network
        "qf": [256, 256, 256]   # Q-function network
    },
    "activation_fn": nn.ReLU
}

# Create SAC model
model = SAC(
    policy="MlpPolicy",
    env=env,
    policy_kwargs=policy_kwargs,
    learning_rate=3e-4,
    buffer_size=1_000_000,    # Replay buffer size
    batch_size=256,           # Minibatch size
    tau=0.005,                # Soft update coefficient
    gamma=0.99,               # Discount factor
    train_freq=1,             # Update policy every 1 step
    gradient_steps=1,         # How many gradient steps to do after each rollout
    ent_coef="auto_0.1",      # Entropy coefficient (auto adjusted)
    target_update_interval=1, # Update target network every step
    verbose=1,
    device="auto"
)

# Train the model
timesteps = 1_000_000
model.learn(total_timesteps=timesteps, log_interval=10)

# Save the model
model.save("sac_ant")

# Evaluate the trained model
env.training = False
env.norm_reward = False

# Run evaluation
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")