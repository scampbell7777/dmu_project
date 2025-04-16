import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
import torch.nn as nn
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import time
import os

# # Create environment
# def make_env():
#     return Monitor(gym.make("Humanoid-v4"))

# if __name__ == "__main__":
#     # Use multiple parallel environments for faster training on Mac
#     num_envs = 8  # Adjust based on your CPU cores
#     env = SubprocVecEnv([make_env for _ in range(num_envs)])
#     env = VecNormalize(env, norm_obs=True, norm_reward=True)

#     # Create a custom policy with larger network for the more complex Humanoid environment
#     policy_kwargs = {
#         "net_arch": [
#             {"pi": [512, 512, 256], "vf": [512, 512, 256]}  # Deeper and wider networks for Humanoid
#         ],
#         "activation_fn": nn.ReLU
#     }

#     model_path = "ppo_humanoid"
#     vec_normalize_path = "vec_normalize_humanoid.pkl"

#     if os.path.exists(vec_normalize_path):
#         print(f"Loading environment normalization stats from {vec_normalize_path}")
#         env = VecNormalize.load(vec_normalize_path, env)
#         # Keep training normalization statistics
#         env.training = True
#         env.norm_reward = True
#     else:
#         print("No saved normalization stats found, using new normalized environment")
#         # Note: VecNormalize is already created above

#     # Setup checkpoint callback for saving intermediate models
#     os.makedirs("./checkpoints/", exist_ok=True)  # Ensure directory exists
#     checkpoint_callback = CheckpointCallback(
#         save_freq=100000,  # Save every 100k steps
#         save_path="./checkpoints/",
#         name_prefix="humanoid_model",
#         save_replay_buffer=False,
#         save_vecnormalize=True,
#     )

#     if os.path.exists(model_path + ".zip"):
#         print("Loading pretrained model...")
#         model = PPO.load(model_path, env=env)
#     else:
#         print("Creating new model...")
#         model = PPO(
#             policy="MlpPolicy",
#             policy_kwargs=policy_kwargs,
#             env=env,
#             learning_rate=3e-4,
#             n_steps=2048,
#             batch_size=64,  # Smaller batch size to avoid memory issues
#             n_epochs=10,
#             gamma=0.99,
#             gae_lambda=0.95,
#             clip_range=0.2,
#             ent_coef=0.01,  # Add some exploration incentive
#             verbose=1,
#             tensorboard_log="./humanoid_tensorboard/"
#         )

#     # Train the model with more timesteps due to higher complexity
#     print("Starting training...")
#     model.learn(total_timesteps=3_000_000, callback=checkpoint_callback)

#     # Save the final model and normalization stats
#     print("Saving model...")
#     model.save(model_path)
#     env.save(vec_normalize_path)

#     # For evaluation and testing
#     print("Evaluating model...")
#     eval_env = DummyVecEnv([make_env])
#     eval_env = VecNormalize.load(vec_normalize_path, eval_env)
#     eval_env.training = False
#     eval_env.norm_reward = False

#     # Evaluate the trained agent
#     mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=30)
#     print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

  
model = PPO.load("ppo_humanoid")

# For rendering - create a new environment with render_mode and wrap it properly
render_env = gym.make("Humanoid-v4", render_mode="human")
render_vec_env = DummyVecEnv([lambda: render_env])
# We need to also normalize this environment, but with the same stats as the training one
render_vec_env = VecNormalize.load("vec_normalize_humanoid.pkl", render_vec_env)
render_vec_env.training = False
render_vec_env.norm_reward = False

# Test and visualize the trained agent
while True:
    obs = render_vec_env.reset()
    for i in range(500):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = render_vec_env.step(action)
        time.sleep(.1)
        if dones[0]:
            obs = render_vec_env.reset()

render_env.close()
   