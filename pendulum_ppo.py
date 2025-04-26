import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

# Custom callback for plotting training metrics
class TrainingMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TrainingMetricsCallback, self).__init__(verbose)
        self.rewards = []
        self.timesteps = []
        self.episode_rewards = []
        self.num_episodes = 0
        self.episode_reward = 0

    def _on_step(self):
        # Track rewards
        self.episode_reward += self.locals["rewards"][0]
        if self.episode_reward  == 1000:
            return False
        # If episode ended
        if self.locals["dones"][0]:
            self.num_episodes += 1
            self.episode_rewards.append(self.episode_reward)
            self.timesteps.append(self.num_timesteps)
            self.episode_reward = 0
        return True
    
res = []
for i in range(10):
    env = gym.make("InvertedPendulum-v4")  # Specify render_mode here
    eval_env = gym.make("InvertedPendulum-v4")

    # Initialize the model
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1
    )

    # Create the callback
    callback = TrainingMetricsCallback()
    # Train the model with callback
    model.learn(total_timesteps=50000, callback=callback)
    print(callback.timesteps), print(callback.episode_rewards)
    res.append([callback.timesteps, callback.episode_rewards])
print(res)

# # Plot episode rewards
plt.subplot(1, 2, 1)
plt.plot(callback.timesteps, callback.episode_rewards)
plt.title('Rewards over Time')

plt.show()

