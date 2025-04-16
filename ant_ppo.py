import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch.nn as nn
from stable_baselines3.common.monitor import Monitor
import time
import os

# # Create environment
# def make_env():
#     return Monitor(gym.make("Ant-v4"))

# env = DummyVecEnv([make_env])
# env = VecNormalize(env, norm_obs=True, norm_reward=True)

# # Create a custom policy with larger network
# policy_kwargs = {
#     "net_arch": [
#         {"pi": [256, 256], "vf": [256, 256]}  # Separate policy and value networks
#     ],
#     "activation_fn": nn.ReLU  # ReLU often works better than tanh for complex environments
# }

# model_path = "ppo_ant"
# vec_normalize_path = "vec_normalize_ant.pkl"

# if os.path.exists(vec_normalize_path):
#     print(f"Loading environment normalization stats from {vec_normalize_path}")
#     env = VecNormalize.load(vec_normalize_path, env)
#     # Keep training normalization statistics
#     env.training = True
#     env.norm_reward = True
# else:
#     print("No saved normalization stats found, creating new normalized environment")
#     env = VecNormalize(env, norm_obs=True, norm_reward=True)

# if os.path.exists(model_path + ".zip"):  # Make sure to check with .zip extension
#     print("Loading pretrained model...")
#     model = PPO.load(model_path, env=env)
# else:
#     print("Creating new model...")
#     model = PPO(
#         policy="MlpPolicy",
#         policy_kwargs=policy_kwargs,
#         env=env,
#         learning_rate=3e-4,
#         n_steps=2048,
#         batch_size=256,
#         n_epochs=10,
#         gamma=0.99,
#         gae_lambda=0.95,
#         clip_range=0.2,
#         verbose=1
#     )

# # Train the model
# model.learn(total_timesteps=1_000_000)

# # Save the model and normalization stats
# model.save("ppo_ant")
# env.save("vec_normalize_ant.pkl")

# # For evaluation and testing
# eval_env = DummyVecEnv([make_env])
# eval_env = VecNormalize.load("vec_normalize_ant.pkl", eval_env)
# eval_env.training = False
# eval_env.norm_reward = False

# # Evaluate the trained agent
# mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)
# print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")


#### eval the model ##################################################################################

# Load the saved model
# model = PPO.load("ppo_ant")

# # Define a function to create the environment (same as you used during training)
# def make_env():
#     return gym.make("Ant-v4")

# # For evaluation
# eval_env = DummyVecEnv([make_env])
# eval_env = VecNormalize.load("vec_normalize_ant.pkl", eval_env)
# eval_env.training = False
# eval_env.training = False
# eval_env.norm_reward = False
# # eval_env.norm_obs=False

# # Evaluate the trained agent
# mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
# print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# # For rendering - create a new environment with render_mode and wrap it properly
# render_env = gym.make("Ant-v4", render_mode="human")
# render_vec_env = DummyVecEnv([lambda: render_env])
# # We need to also normalize this environment, but with the same stats as the training one
# render_vec_env = VecNormalize.load("vec_normalize_ant.pkl", render_vec_env)
# render_vec_env.training = False
# render_vec_env.norm_reward = False

# # Test and visualize the trained agent
# while True:
#     obs = render_vec_env.reset()
#     for i in range(500):
#         action, _states = model.predict(obs, deterministic=True)
#         obs, rewards, dones, infos = render_vec_env.step(action)
#         time.sleep(.1)
#         if dones[0]:
#             obs = render_vec_env.reset()

# render_env.close()