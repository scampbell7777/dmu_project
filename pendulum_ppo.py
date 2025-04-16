# pip install stable-baselines3[extra]
# pip install gymnasium[box2d]
# pip install "gymnasium[mujoco]"
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Create environment
env = gym.make("InvertedPendulum-v4")  # Specify render_mode here
eval_env = gym.make("InvertedPendulum-v4")

# Initialize the model
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1
)

# Train the model
model.learn(total_timesteps=100000)

# Save the model
# model.save("ppo_inverted_pendulum")

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

env = gym.make("InvertedPendulum-v4", render_mode="human")
# Test the trained agent
obs, _ = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    # No need to call env.render() as it happens automatically with render_mode="human"
    if terminated or truncated:
        obs, _ = env.reset()
        
env.close()