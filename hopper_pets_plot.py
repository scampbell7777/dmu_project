# import numpy as np
# import torch
# import torch.nn as nn
# import gymnasium as gym
# import matplotlib.pyplot as plt
# from torch.utils.data import DataLoader, TensorDataset
# import random
# import copy
# from gymnasium.wrappers import NormalizeObservation

# import gymnasium as gym
# from gymnasium import spaces
# import numpy as np


# class MinMaxNormalization(gym.Wrapper):
#     """
#     A wrapper that normalizes environment observations using min-max normalization
#     and keeps track of exact minimum and maximum values observed.
#     """
#     def __init__(self, env, clip_obs=10.0):
#         super(MinMaxNormalization, self).__init__(env)
#         self.clip_obs = clip_obs
        
#         # Initialize min and max to None (will be set on first observation)
#         self.obs_min = None
#         self.obs_max = None
        
#         # Set observation space to be in range [0, 1]
#         low = np.zeros(self.observation_space.shape, dtype=np.float32)
#         high = np.ones(self.observation_space.shape, dtype=np.float32)
#         self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
#     def reset(self, **kwargs):
#         obs, info = self.env.reset(**kwargs)
        
#         # Update min/max on reset
#         self._update_obs_stats(obs)
        
#         # Normalize observation
#         normalized_obs = self._normalize(obs)
        
#         return normalized_obs, info
    
#     def step(self, action):
#         obs, reward, terminated, truncated, info = self.env.step(action)
        
#         # Update min/max on each step
#         self._update_obs_stats(obs)
        
#         # Normalize observation
#         normalized_obs = self._normalize(obs)
        
#         return normalized_obs, reward, terminated, truncated, info
    
#     def _update_obs_stats(self, obs):
#         """Update running min/max statistics with new observation."""
#         # Handle first observation
#         if self.obs_min is None:
#             self.obs_min = np.full(self.env.observation_space.shape, np.inf, dtype=np.float32)
#             self.obs_max = np.full(self.env.observation_space.shape, -np.inf, dtype=np.float32)
        
#         # Clip observations to prevent extreme values if needed
#         clipped_obs = np.clip(obs, -self.clip_obs, self.clip_obs)
        
#         # Update min and max exactly
#         self.obs_min = np.minimum(self.obs_min, clipped_obs)
#         self.obs_max = np.maximum(self.obs_max, clipped_obs)
    
#     def _normalize(self, obs):
#         """Normalize observation using current min/max values."""
#         # Clip observations to prevent extreme values if needed
#         clipped_obs = np.clip(obs, -self.clip_obs, self.clip_obs)
        
#         # Avoid division by zero by adding a small epsilon where max = min
#         epsilon = 1e-8
#         denominator = self.obs_max - self.obs_min
#         # Where max equals min, set denominator to 1 to get 0 after normalization
#         denominator = np.where(denominator < epsilon, 1.0, denominator)
        
#         # Normalize to [0, 1] range
#         normalized_obs = (clipped_obs - self.obs_min) / denominator
        
#         return normalized_obs
    
#     def unnormalize(self, normalized_obs):
#         """
#         Convert a normalized observation back to the original scale.
#         Supports both NumPy arrays and PyTorch tensors.
#         """
#         if self.obs_min is None or self.obs_max is None:
#             raise ValueError("Cannot unnormalize before any observations have been seen")
        
#         # Check if input is a PyTorch tensor
#         is_torch_tensor = isinstance(normalized_obs, torch.Tensor)
        
#         if is_torch_tensor:
#             # Convert numpy arrays to torch tensors with the same device as input
#             device = normalized_obs.device
#             obs_min = torch.tensor(self.obs_min, dtype=normalized_obs.dtype, device=device)
#             obs_max = torch.tensor(self.obs_max, dtype=normalized_obs.dtype, device=device)
            
#             # Ensure normalized_obs is in [0, 1] range
#             normalized_obs = torch.clamp(normalized_obs, 0.0, 1.0)
            
#             # Unnormalize
#             original_obs = normalized_obs * (obs_max - obs_min) + obs_min
#         else:
#             # Handle numpy array case
#             # Convert to numpy if it's a list or other type
#             if not isinstance(normalized_obs, np.ndarray):
#                 normalized_obs = np.array(normalized_obs, dtype=np.float32)
            
#             # Ensure normalized_obs is in [0, 1] range
#             normalized_obs = np.clip(normalized_obs, 0.0, 1.0)
            
#             # Unnormalize
#             original_obs = normalized_obs * (self.obs_max - self.obs_min) + self.obs_min
        
#         return original_obs
    
#     def get_normalization_params(self):
#         """Return the current min and max values used for normalization."""
#         return {
#             "obs_min": self.obs_min.copy() if self.obs_min is not None else None,
#             "obs_max": self.obs_max.copy() if self.obs_max is not None else None
#         }




# env = gym.make('Hopper-v5')
# env = MinMaxNormalization(env, clip_obs=10.0)

# def collect_data(num_episodes=100, max_steps=200):
#     # env = gym.make('Hopper-v5')
#     # env = NormalizeObservation(env)
#     trajectories = []
    
#     while len(trajectories) < num_episodes:
#         state, _ = env.reset()
#         states, actions, next_states, rewards = [], [], [], []
        
#         for t in range(max_steps):
#             action = env.action_space.sample()
#             next_state, reward, terminated, truncated, info = env.step(action)
            
#             states.append(state)
#             actions.append(action)
#             next_states.append(next_state)
#             rewards.append(reward)
            
#             state = next_state
#             if terminated or truncated:
#                 break
        
#         trajectories.append({
#             'states': np.array(states),
#             'actions': np.array(actions),
#             'next_states': np.array(next_states),
#             'rewards': np.array(rewards)
#         })
#         print(f"Episode {len(trajectories)}: collected trajectory with {len(states)} steps and reward {sum(rewards)}")
    
#     env.close()
#     return trajectories

# def is_healthy(observation, exclude_current_positions_from_observation=True):
#     healthy_state_range = [-100.0, 100.0]
#     healthy_z_range = [0.7, float('inf')]
#     healthy_angle_range = [-0.2, 0.2]
    
#     if exclude_current_positions_from_observation:
#         state_healthy = all(healthy_state_range[0] <= x <= healthy_state_range[1] for x in observation[1:])
#         z_healthy = healthy_z_range[0] <= observation[0] <= healthy_z_range[1]
#         angle_healthy = healthy_angle_range[0] <= observation[1] <= healthy_angle_range[1]
#     else:
#         state_healthy = all(healthy_state_range[0] <= x <= healthy_state_range[1] for x in observation[2:])
#         z_healthy = healthy_z_range[0] <= observation[1] <= healthy_z_range[1]
#         angle_healthy = healthy_angle_range[0] <= observation[2] <= healthy_angle_range[1]
    
#     return state_healthy and z_healthy and angle_healthy

# class ModelMember(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_dim=200):
#         super(ModelMember, self).__init__()
#         self.state_dim = state_dim
#         self.action_dim = action_dim
        
#         self.net = nn.Sequential(
#             nn.Linear(state_dim + action_dim, hidden_dim),
#             nn.SiLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.SiLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.SiLU(),
#             nn.Linear(hidden_dim, state_dim)
#         )
        
#     def forward(self, state, action, n_euler_steps=1):
#         state_action = torch.cat([state, action], dim=-1)
#         next_state = state
#         step_size = 1.0 / n_euler_steps
#         for _ in range(n_euler_steps):
#             next_state = step_size * self.net(state_action) + next_state
#             state_action = torch.cat([next_state, action], dim=-1)
#         return next_state

# class StatePredictor(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_dim=128, ensemble_size=1):
#         super(StatePredictor, self).__init__()
#         self.ensemble_size = ensemble_size
#         self.models = nn.ModuleList([
#             ModelMember(state_dim, action_dim, hidden_dim) 
#             for _ in range(ensemble_size)
#         ])
        
#     def forward(self, state, action, n_euler_steps=1, model_idx=None):
#         if model_idx is not None:
#             return self.models[model_idx](state, action, n_euler_steps)
        
#         predictions = [model(state, action, n_euler_steps) for model in self.models]
#         return torch.stack(predictions, dim=0)
    
#     def predict_mean(self, state, action, n_euler_steps=1):
#         predictions = self.forward(state, action, n_euler_steps)
#         return torch.mean(predictions, dim=0)

# def prepare_training_data(trajectories, batch_size=32):
#     state_samples = []
#     action_samples = []
#     next_state_samples = []
#     reward_samples = []
    
#     for traj in trajectories:
#         states = traj['states']
#         actions = traj['actions']
#         next_states = traj['next_states']
#         rewards = traj['rewards']
        
#         for i in range(len(states)):
#             state_samples.append(states[i])
#             action_samples.append(actions[i])
#             next_state_samples.append(next_states[i])
#             reward_samples.append(rewards[i])
    
#     state_tensor = torch.tensor(np.array(state_samples), dtype=torch.float32)
#     action_tensor = torch.tensor(np.array(action_samples), dtype=torch.float32)
#     next_state_tensor = torch.tensor(np.array(next_state_samples), dtype=torch.float32)
#     reward_tensor = torch.tensor(np.array(reward_samples), dtype=torch.float32).unsqueeze(1)
    
#     dataset = TensorDataset(state_tensor, action_tensor, next_state_tensor, reward_tensor)
#     data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
#     return data_loader

# def train_model(data_loader, state_dim, action_dim, epochs=50, lr=5e-4, model=None):
#     if model is not None:
#         dynamics_model = model
#     else:
#         dynamics_model = StatePredictor(state_dim, action_dim, hidden_dim=64, ensemble_size=3)
    
#     ensemble_size = dynamics_model.ensemble_size
    
#     optimizers = [torch.optim.Adam(model.parameters(), lr=lr) for model in dynamics_model.models]
#     criterion = nn.MSELoss() #(reduction='none')
    
#     for epoch in range(epochs):
#         total_dynamics_losses = [0.0 for _ in range(ensemble_size)]
#         num_batches = 0
        
#         for states, actions, next_states, _ in data_loader:
#             batch_size = states.shape[0]
            
#             for model_idx, (model, optimizer) in enumerate(zip(dynamics_model.models, optimizers)):
#                 mask = torch.zeros(batch_size, dtype=torch.bool)
#                 indices = torch.randperm(batch_size)[:batch_size // 2]
#                 mask[indices] = True
                
#                 if not any(mask):
#                     continue
                
#                 optimizer.zero_grad()
#                 batch_states = states[mask]
#                 batch_actions = actions[mask]
#                 batch_next_states = next_states[mask]
                
#                 predicted_next_states = model(batch_states, batch_actions)
#                 # mse_loss = criterion(predicted_next_states, batch_next_states)
#                 # weights = torch.ones_like(mse_loss)
#                 # weights[:, 1] = 100.0
                
#                 # # Apply weights and compute mean
#                 # dynamics_loss = (mse_loss * weights).mean()

#                 dynamics_loss = criterion(predicted_next_states, batch_next_states)
#                 dynamics_loss.backward()
#                 optimizer.step()
                
#                 total_dynamics_losses[model_idx] += dynamics_loss.item()
            
#             num_batches += 1
        
#         avg_losses = [loss/num_batches for loss in total_dynamics_losses]
#         print(f"Epoch {epoch+1}/{epochs}, Dynamics Losses: {avg_losses}")
    
#     return dynamics_model

# def simulate_model(dynamics_model, reward_model, init_state, action_sequence):
#     init_state_tensor = torch.tensor(init_state, dtype=torch.float32).unsqueeze(0)
#     states = [init_state_tensor]
#     rewards = []
    
#     for action in action_sequence:
#         action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
#         next_state = dynamics_model.predict_mean(states[-1], action_tensor)
        
#         reward = reward_model(states[-1], action_tensor, next_state)
        
#         states.append(next_state)
#         rewards.append(reward)
    
#     return torch.stack(states).squeeze(1), torch.stack(rewards).squeeze(1)

# def check_termination(state_tensor, exclude_current_positions_from_observation=True):
#     healthy_state_range = [-100.0, 100.0]
#     healthy_z_range = [0.7, float('inf')]
#     healthy_angle_range = [-0.2, 0.2]
    
#     if exclude_current_positions_from_observation:
#         z_healthy = healthy_z_range[0] <= state_tensor[0].item() <= healthy_z_range[1]
#         angle_healthy = healthy_angle_range[0] <= state_tensor[1].item() <= healthy_angle_range[1]
#         state_healthy = all(healthy_state_range[0] <= x.item() <= healthy_state_range[1] 
#                             for x in state_tensor[1:])
#     else:
#         z_healthy = healthy_z_range[0] <= state_tensor[1].item() <= healthy_z_range[1]
#         angle_healthy = healthy_angle_range[0] <= state_tensor[2].item() <= healthy_angle_range[1]
#         state_healthy = all(healthy_state_range[0] <= x.item() <= healthy_state_range[1] 
#                             for x in state_tensor[2:])
    
#     return not (z_healthy and angle_healthy and state_healthy)

# def optimize_actions(dynamics_model, init_state, horizon=30, iterations=10, lr=1e-1, gamma=1.0):
#     init_state_tensor = torch.tensor(init_state, dtype=torch.float32).unsqueeze(0)
    
#     actions = nn.Parameter(torch.randn(horizon, 3) * 0.01)
#     optimizer = torch.optim.Adam([actions], lr=lr, weight_decay=1e-5)
    
#     for i in range(iterations):
#         optimizer.zero_grad()
        
#         total_loss = 0
#         for model_idx in range(dynamics_model.ensemble_size):
#             current_state = init_state_tensor.clone()
#             states = [current_state]
            
#             for t in range(horizon):
#                 action = torch.tanh(actions[t]).unsqueeze(0)
#                 next_state = dynamics_model(current_state, action, model_idx=model_idx)
#                 states.append(next_state)
#                 current_state = next_state
            
#             trajectory = torch.cat(states, dim=0)
#             trajectory = env.unnormalize(trajectory)
#             x_velocities = trajectory[:, 5]
#             angles = trajectory[:, 1]
#             z = trajectory[:, 0]
#             is_healthy = torch.tanh((z - 0.7) * 40) * torch.tanh((0.2 - torch.abs(angles)) * 40)

#             model_loss = -torch.mean(torch.tanh((z - 0.7) * 10)) -torch.mean(torch.tanh((0.2 - torch.abs(angles)) * 10)) - (x_velocities*is_healthy).mean()

#             total_loss += model_loss / dynamics_model.ensemble_size
#         total_loss.backward()
#         optimizer.step()
    
#     with torch.no_grad():
#         final_actions = torch.tanh(actions)
    
#     return final_actions

# def collect_optimized_trajectories(dynamics_model, num_episodes=10, horizon=30, iterations=50, lr=0.01):
#     # env = gym.make('Hopper-v5')
#     # env = NormalizeObservation(env)
#     trajectories = []
    
#     for episode in range(num_episodes):
#         state, _ = env.reset()
#         states, actions, next_states, rewards = [], [], [], []
        
#         # For single-step prediction plotting
#         true_z_values = []
#         true_angle_values = []
#         true_x_velocity_values = []
        
#         predicted_z_values = []
#         predicted_angle_values = []
#         predicted_x_velocity_values = []
        
#         done = False
#         steps = 0
#         total_reward = 0
        
#         # Store initial state for full trajectory prediction
#         initial_state = state.copy()
        
#         while not done and steps < 1000:
#             action_seq = optimize_actions(
#                 dynamics_model, 
#                 state, 
#                 horizon=horizon, 
#                 iterations=iterations,
#                 lr=lr
#             )
            
#             action = action_seq[0].detach().numpy()
#             next_state, reward, terminated, truncated, info = env.step(action)
#             total_reward += reward
            
#             # Store true values
#             true_z_values.append(state[0])
#             true_angle_values.append(state[1])
#             true_x_velocity_values.append(state[5])
            
#             # Get predicted next state
#             state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
#             action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
#             pred_next_state = dynamics_model.predict_mean(state_tensor, action_tensor).squeeze(0).detach().numpy()
            
#             # Store predicted values
#             predicted_z_values.append(pred_next_state[0])
#             predicted_angle_values.append(pred_next_state[1])
#             predicted_x_velocity_values.append(pred_next_state[5])
            
#             states.append(state)
#             actions.append(action)
#             next_states.append(next_state)
#             rewards.append(reward)
            
#             state = next_state
#             steps += 1
#             done = terminated or truncated
        
#         trajectories.append({
#             'states': np.array(states),
#             'actions': np.array(actions),
#             'next_states': np.array(next_states),
#             'rewards': np.array(rewards)
#         })
        
#         print(info)
#         print(f"Episode {episode+1}: collected trajectory with {len(states)} steps and reward {total_reward}")
        
#         # Plot single-step predictions vs. true values
#         steps_range = range(len(true_z_values))
        
#         fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        
#         # Z-position plot
#         axs[0].plot(steps_range, true_z_values, label='True z')
#         axs[0].plot(steps_range, predicted_z_values, label='Predicted z')
#         axs[0].set_title('Z Position (Single-step Prediction)')
#         axs[0].set_xlabel('Steps')
#         axs[0].set_ylabel('Z Position')
#         axs[0].legend()
        
#         # Angle plot
#         axs[1].plot(steps_range, true_angle_values, label='True angle')
#         axs[1].plot(steps_range, predicted_angle_values, label='Predicted angle')
#         axs[1].set_title('Angle (Single-step Prediction)')
#         axs[1].set_xlabel('Steps')
#         axs[1].set_ylabel('Angle')
#         axs[1].legend()
        
#         # X-velocity plot
#         axs[2].plot(steps_range, true_x_velocity_values, label='True x_velocity')
#         axs[2].plot(steps_range, predicted_x_velocity_values, label='Predicted x_velocity')
#         axs[2].set_title('X Velocity (Single-step Prediction)')
#         axs[2].set_xlabel('Steps')
#         axs[2].set_ylabel('X Velocity')
#         axs[2].legend()
        
#         plt.tight_layout()
#         plt.show()
#         plt.savefig(f'trajectory_plots_episode_{episode+1}_single_step.png')
#         plt.close()
        
#         # Generate full trajectory prediction using only the model
#         full_traj_pred_z = []
#         full_traj_pred_angle = []
#         full_traj_pred_x_velocity = []
        
#         # Start from initial state
#         curr_state = torch.tensor(initial_state, dtype=torch.float32).unsqueeze(0)
        
#         # Use the same actions as in the real trajectory
#         for action_step in actions:
#             action_tensor = torch.tensor(action_step, dtype=torch.float32).unsqueeze(0)
#             pred_next_state = dynamics_model.predict_mean(curr_state, action_tensor)
            
#             # Extract values for plotting
#             state_np = pred_next_state.squeeze(0).detach().numpy()
#             full_traj_pred_z.append(state_np[0])
#             full_traj_pred_angle.append(state_np[1])
#             full_traj_pred_x_velocity.append(state_np[5])
            
#             # Update current state for next prediction
#             curr_state = pred_next_state
        
#         # Plot full trajectory prediction vs. true values
#         fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        
#         # Z-position plot
#         axs[0].plot(steps_range, true_z_values, label='True z')
#         axs[0].plot(steps_range, full_traj_pred_z, label='Model-only predicted z')
#         axs[0].set_title('Z Position (Full Trajectory Prediction)')
#         axs[0].set_xlabel('Steps')
#         axs[0].set_ylabel('Z Position')
#         axs[0].legend()
        
#         # Angle plot
#         axs[1].plot(steps_range, true_angle_values, label='True angle')
#         axs[1].plot(steps_range, full_traj_pred_angle, label='Model-only predicted angle')
#         axs[1].set_title('Angle (Full Trajectory Prediction)')
#         axs[1].set_xlabel('Steps')
#         axs[1].set_ylabel('Angle')
#         axs[1].legend()
        
#         # X-velocity plot
#         axs[2].plot(steps_range, true_x_velocity_values, label='True x_velocity')
#         axs[2].plot(steps_range, full_traj_pred_x_velocity, label='Model-only predicted x_velocity')
#         axs[2].set_title('X Velocity (Full Trajectory Prediction)')
#         axs[2].set_xlabel('Steps')
#         axs[2].set_ylabel('X Velocity')
#         axs[2].legend()
        
#         plt.tight_layout()
#         plt.show()
#         plt.savefig(f'trajectory_plots_episode_{episode+1}_full_trajectory.png')
#         plt.close()
    
#     env.close()
#     return trajectories

# def eval_model(dynamics_model, n_evals=5):
#     # env = gym.make('Hopper-v5')
#     # env = NormalizeObservation(env)
#     avg_reward = 0
#     avg_steps = 0
    
#     for k in range(n_evals):
#         state, _ = env.reset()
#         total_reward = 0
#         episode_steps = 0
        
#         for step in range(1000):
#             action_seq = optimize_actions(
#                 dynamics_model, 
#                 state, 
#                 horizon=10, 
#                 iterations=50,
#                 lr=0.01
#             )
            
#             action = action_seq[0].detach().numpy()
#             next_state, reward, terminated, truncated, _ = env.step(action)
            
#             state = next_state
#             total_reward += reward
#             episode_steps += 1
            
#             if terminated or truncated:
#                 break
        
#         avg_reward += total_reward
#         avg_steps += episode_steps
#         print(f"Eval episode {k+1}: Steps = {episode_steps}, Reward = {total_reward:.2f}")
    
#     print(f"Avg steps: {avg_steps/n_evals:.1f}, Avg reward: {avg_reward/n_evals:.2f}")
#     env.close()
#     return avg_reward/n_evals

# def main():
#     random_trajectories = collect_data(num_episodes=100, max_steps=1000)
#     print(f"Collected {len(random_trajectories)} random trajectories")
    
#     traj_sample = random_trajectories[0]
#     state_dim = traj_sample['states'].shape[1]
#     action_dim = traj_sample['actions'].shape[1]
    
#     print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    
#     all_trajectories = random_trajectories.copy()
    
#     best_dynamics_model = None
#     best_performance = -float('inf')
    
#     num_iterations = 2000
#     for iteration in range(num_iterations):
#         print(f"\n===== ITERATION {iteration+1}/{num_iterations} =====")
        
#         data_loader = prepare_training_data(all_trajectories, batch_size=64)
        
#         dynamics_model = train_model(data_loader, state_dim, action_dim, epochs=100, lr=1e-3)
        
#         print(f"Iteration {iteration+1}: Model training complete")
#         horizon = 5 + int(iteration/5)
#         print(horizon)
#         print(f"Iteration {iteration+1}: Collecting optimized trajectories")
#         optimized_trajectories = collect_optimized_trajectories(
#             dynamics_model, 
#             num_episodes=1, 
#             horizon=horizon,
#             iterations=50,
#             lr=0.001
#         )
        
#         print(f"Iteration {iteration+1}: Collected {len(optimized_trajectories)} optimized trajectories")
        
#         all_trajectories.extend(optimized_trajectories)
#         print(f"Iteration {iteration+1}: Total trajectories: {len(all_trajectories)}")
    
#     print("\n===== FINAL EVALUATION =====")
#     print("Evaluating best model:")
#     final_performance = eval_model(best_dynamics_model, n_evals=5)
#     print(f"Best model performance: {final_performance:.2f}")

# if __name__ == "__main__":
#     main()


import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import random
import copy
from gymnasium.wrappers import NormalizeObservation

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class MinMaxNormalization(gym.Wrapper):
    """
    A wrapper that normalizes environment observations using min-max normalization
    and keeps track of exact minimum and maximum values observed.
    """
    def __init__(self, env, clip_obs=10.0):
        super(MinMaxNormalization, self).__init__(env)
        self.clip_obs = clip_obs
        
        # Initialize min and max to None (will be set on first observation)
        self.obs_min = None
        self.obs_max = None
        
        # Set observation space to be in range [0, 1]
        low = np.zeros(self.observation_space.shape, dtype=np.float32)
        high = np.ones(self.observation_space.shape, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # Update min/max on reset
        self._update_obs_stats(obs)
        
        # Normalize observation
        normalized_obs = self._normalize(obs)
        
        return normalized_obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update min/max on each step
        self._update_obs_stats(obs)
        
        # Normalize observation
        normalized_obs = self._normalize(obs)
        
        return normalized_obs, reward, terminated, truncated, info
    
    def _update_obs_stats(self, obs):
        """Update running min/max statistics with new observation."""
        # Handle first observation
        if self.obs_min is None:
            self.obs_min = np.full(self.env.observation_space.shape, np.inf, dtype=np.float32)
            self.obs_max = np.full(self.env.observation_space.shape, -np.inf, dtype=np.float32)
        
        # Clip observations to prevent extreme values if needed
        clipped_obs = np.clip(obs, -self.clip_obs, self.clip_obs)
        
        # Update min and max exactly
        self.obs_min = np.minimum(self.obs_min, clipped_obs)
        self.obs_max = np.maximum(self.obs_max, clipped_obs)
    
    def _normalize(self, obs):
        """Normalize observation using current min/max values."""
        # Clip observations to prevent extreme values if needed
        clipped_obs = np.clip(obs, -self.clip_obs, self.clip_obs)
        
        # Avoid division by zero by adding a small epsilon where max = min
        epsilon = 1e-8
        denominator = self.obs_max - self.obs_min
        # Where max equals min, set denominator to 1 to get 0 after normalization
        denominator = np.where(denominator < epsilon, 1.0, denominator)
        
        # Normalize to [0, 1] range
        normalized_obs = (clipped_obs - self.obs_min) / denominator
        
        return normalized_obs
    
    def unnormalize(self, normalized_obs):
        """
        Convert a normalized observation back to the original scale.
        Supports both NumPy arrays and PyTorch tensors.
        """
        if self.obs_min is None or self.obs_max is None:
            raise ValueError("Cannot unnormalize before any observations have been seen")
        
        # Check if input is a PyTorch tensor
        is_torch_tensor = isinstance(normalized_obs, torch.Tensor)
        
        if is_torch_tensor:
            # Convert numpy arrays to torch tensors with the same device as input
            device = normalized_obs.device
            obs_min = torch.tensor(self.obs_min, dtype=normalized_obs.dtype, device=device)
            obs_max = torch.tensor(self.obs_max, dtype=normalized_obs.dtype, device=device)
            
            # Ensure normalized_obs is in [0, 1] range
            normalized_obs = torch.clamp(normalized_obs, 0.0, 1.0)
            
            # Unnormalize
            original_obs = normalized_obs * (obs_max - obs_min) + obs_min
        else:
            # Handle numpy array case
            # Convert to numpy if it's a list or other type
            if not isinstance(normalized_obs, np.ndarray):
                normalized_obs = np.array(normalized_obs, dtype=np.float32)
            
            # Ensure normalized_obs is in [0, 1] range
            normalized_obs = np.clip(normalized_obs, 0.0, 1.0)
            
            # Unnormalize
            original_obs = normalized_obs * (self.obs_max - self.obs_min) + self.obs_min
        
        return original_obs
    
    def get_normalization_params(self):
        """Return the current min and max values used for normalization."""
        return {
            "obs_min": self.obs_min.copy() if self.obs_min is not None else None,
            "obs_max": self.obs_max.copy() if self.obs_max is not None else None
        }




env = gym.make('Hopper-v5')
env = MinMaxNormalization(env, clip_obs=10.0)

def collect_data(num_episodes=100, max_steps=200):
    # env = gym.make('Hopper-v5')
    # env = NormalizeObservation(env)
    trajectories = []
    
    while len(trajectories) < num_episodes:
        state, _ = env.reset()
        states, actions, next_states, rewards = [], [], [], []
        
        for t in range(max_steps):
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Store unnormalized states
            states.append(env.unnormalize(state))
            actions.append(action)
            next_states.append(env.unnormalize(next_state))
            rewards.append(reward)
            
            state = next_state
            if terminated or truncated:
                break
        
        trajectories.append({
            'states': np.array(states),
            'actions': np.array(actions),
            'next_states': np.array(next_states),
            'rewards': np.array(rewards)
        })
        print(f"Episode {len(trajectories)}: collected trajectory with {len(states)} steps and reward {sum(rewards)}")
    
    env.close()
    return trajectories

def is_healthy(observation, exclude_current_positions_from_observation=True):
    healthy_state_range = [-100.0, 100.0]
    healthy_z_range = [0.7, float('inf')]
    healthy_angle_range = [-0.2, 0.2]
    
    if exclude_current_positions_from_observation:
        state_healthy = all(healthy_state_range[0] <= x <= healthy_state_range[1] for x in observation[1:])
        z_healthy = healthy_z_range[0] <= observation[0] <= healthy_z_range[1]
        angle_healthy = healthy_angle_range[0] <= observation[1] <= healthy_angle_range[1]
    else:
        state_healthy = all(healthy_state_range[0] <= x <= healthy_state_range[1] for x in observation[2:])
        z_healthy = healthy_z_range[0] <= observation[1] <= healthy_z_range[1]
        angle_healthy = healthy_angle_range[0] <= observation[2] <= healthy_angle_range[1]
    
    return state_healthy and z_healthy and angle_healthy

class ModelMember(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=200):
        super(ModelMember, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
    def forward(self, state, action, n_euler_steps=1):
        state_action = torch.cat([state, action], dim=-1)
        next_state = state
        step_size = 1.0 / n_euler_steps
        for _ in range(n_euler_steps):
            next_state = step_size * self.net(state_action) + next_state
            state_action = torch.cat([next_state, action], dim=-1)
        return next_state

class StatePredictor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, ensemble_size=1):
        super(StatePredictor, self).__init__()
        self.ensemble_size = ensemble_size
        self.models = nn.ModuleList([
            ModelMember(state_dim, action_dim, hidden_dim) 
            for _ in range(ensemble_size)
        ])
        
    def forward(self, state, action, n_euler_steps=1, model_idx=None):
        if model_idx is not None:
            return self.models[model_idx](state, action, n_euler_steps)
        
        predictions = [model(state, action, n_euler_steps) for model in self.models]
        return torch.stack(predictions, dim=0)
    
    def predict_mean(self, state, action, n_euler_steps=1):
        predictions = self.forward(state, action, n_euler_steps)
        return torch.mean(predictions, dim=0)

def prepare_training_data(trajectories, batch_size=32):
    state_samples = []
    action_samples = []
    next_state_samples = []
    reward_samples = []
    
    for traj in trajectories:
        states = traj['states']
        actions = traj['actions']
        next_states = traj['next_states']
        rewards = traj['rewards']
        
        for i in range(len(states)):
            state_samples.append(states[i])
            action_samples.append(actions[i])
            next_state_samples.append(next_states[i])
            reward_samples.append(rewards[i])
    
    # Convert to tensors
    state_tensor = torch.tensor(np.array(state_samples), dtype=torch.float32)
    action_tensor = torch.tensor(np.array(action_samples), dtype=torch.float32)
    next_state_tensor = torch.tensor(np.array(next_state_samples), dtype=torch.float32)
    reward_tensor = torch.tensor(np.array(reward_samples), dtype=torch.float32).unsqueeze(1)
    
    # Normalize states for training
    state_min = env.obs_min
    state_max = env.obs_max
    state_range = state_max - state_min
    state_range = np.where(state_range < 1e-8, 1.0, state_range)
    
    normalized_state_tensor = (state_tensor - torch.tensor(state_min, dtype=torch.float32)) / torch.tensor(state_range, dtype=torch.float32)
    normalized_next_state_tensor = (next_state_tensor - torch.tensor(state_min, dtype=torch.float32)) / torch.tensor(state_range, dtype=torch.float32)
    
    dataset = TensorDataset(normalized_state_tensor, action_tensor, normalized_next_state_tensor, reward_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return data_loader

def train_model(data_loader, state_dim, action_dim, epochs=50, lr=5e-4, model=None):
    if model is not None:
        dynamics_model = model
    else:
        dynamics_model = StatePredictor(state_dim, action_dim, hidden_dim=200, ensemble_size=3)
    
    ensemble_size = dynamics_model.ensemble_size
    
    optimizers = [torch.optim.Adam(model.parameters(), lr=lr) for model in dynamics_model.models]
    criterion = nn.MSELoss() #(reduction='none')
    
    for epoch in range(epochs):
        total_dynamics_losses = [0.0 for _ in range(ensemble_size)]
        num_batches = 0
        
        for states, actions, next_states, _ in data_loader:
            batch_size = states.shape[0]
            
            for model_idx, (model, optimizer) in enumerate(zip(dynamics_model.models, optimizers)):
                mask = torch.zeros(batch_size, dtype=torch.bool)
                indices = torch.randperm(batch_size)[:batch_size // 2]
                mask[indices] = True
                
                if not any(mask):
                    continue
                
                optimizer.zero_grad()
                batch_states = states[mask]
                batch_actions = actions[mask]
                batch_next_states = next_states[mask]
                
                predicted_next_states = model(batch_states, batch_actions)
                # mse_loss = criterion(predicted_next_states, batch_next_states)
                # weights = torch.ones_like(mse_loss)
                # weights[:, 1] = 100.0
                
                # # Apply weights and compute mean
                # dynamics_loss = (mse_loss * weights).mean()

                dynamics_loss = criterion(predicted_next_states, batch_next_states)
                dynamics_loss.backward()
                optimizer.step()
                
                total_dynamics_losses[model_idx] += dynamics_loss.item()
            
            num_batches += 1
        
        avg_losses = [loss/num_batches for loss in total_dynamics_losses]
        print(f"Epoch {epoch+1}/{epochs}, Dynamics Losses: {avg_losses}")
    
    return dynamics_model

def simulate_model(dynamics_model, reward_model, init_state, action_sequence):
    init_state_tensor = torch.tensor(init_state, dtype=torch.float32).unsqueeze(0)
    states = [init_state_tensor]
    rewards = []
    
    for action in action_sequence:
        action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        next_state = dynamics_model.predict_mean(states[-1], action_tensor)
        
        reward = reward_model(states[-1], action_tensor, next_state)
        
        states.append(next_state)
        rewards.append(reward)
    
    return torch.stack(states).squeeze(1), torch.stack(rewards).squeeze(1)

def check_termination(state_tensor, exclude_current_positions_from_observation=True):
    healthy_state_range = [-100.0, 100.0]
    healthy_z_range = [0.7, float('inf')]
    healthy_angle_range = [-0.2, 0.2]
    
    if exclude_current_positions_from_observation:
        z_healthy = healthy_z_range[0] <= state_tensor[0].item() <= healthy_z_range[1]
        angle_healthy = healthy_angle_range[0] <= state_tensor[1].item() <= healthy_angle_range[1]
        state_healthy = all(healthy_state_range[0] <= x.item() <= healthy_state_range[1] 
                            for x in state_tensor[1:])
    else:
        z_healthy = healthy_z_range[0] <= state_tensor[1].item() <= healthy_z_range[1]
        angle_healthy = healthy_angle_range[0] <= state_tensor[2].item() <= healthy_angle_range[1]
        state_healthy = all(healthy_state_range[0] <= x.item() <= healthy_state_range[1] 
                            for x in state_tensor[2:])
    
    return not (z_healthy and angle_healthy and state_healthy)

def optimize_actions(dynamics_model, init_state, horizon=30, iterations=10, lr=1e-1, gamma=1.0):
    # Normalize the initial state
    normalized_init_state = env._normalize(init_state)
    init_state_tensor = torch.tensor(normalized_init_state, dtype=torch.float32).unsqueeze(0)
    
    actions = nn.Parameter(torch.randn(horizon, 3) * 0.01)
    optimizer = torch.optim.Adam([actions], lr=lr, weight_decay=1e-5)
    
    for i in range(iterations):
        optimizer.zero_grad()
        
        total_loss = 0
        for model_idx in range(dynamics_model.ensemble_size):
            current_state = init_state_tensor.clone()
            states = [current_state]
            
            for t in range(horizon):
                action = torch.tanh(actions[t]).unsqueeze(0)
                next_state = dynamics_model(current_state, action, model_idx=model_idx)
                states.append(next_state)
                current_state = next_state
            
            trajectory = torch.cat(states, dim=0)
            # Unnormalize the trajectory for loss calculation
            unnorm_trajectory = env.unnormalize(trajectory)
            x_velocities = unnorm_trajectory[:, 5]
            angles = unnorm_trajectory[:, 1]
            z = unnorm_trajectory[:, 0]
            is_healthy = torch.tanh((z - 0.7) * 40) * torch.tanh((0.2 - torch.abs(angles)) * 40)

            model_loss = -torch.mean(torch.tanh((z - 0.7) * 10)) -torch.mean(torch.tanh((0.2 - torch.abs(angles)) * 10)) - (x_velocities*is_healthy).mean()

            total_loss += model_loss / dynamics_model.ensemble_size
        total_loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        final_actions = torch.tanh(actions) #+ torch.randn(horizon, 3) * 0.1
    
    return final_actions

# def collect_optimized_trajectories(dynamics_model, num_episodes=10, horizon=30, iterations=50, lr=0.01):
#     # env = gym.make('Hopper-v5')
#     # env = NormalizeObservation(env)
#     trajectories = []
    
#     for episode in range(num_episodes):
#         state, _ = env.reset()
#         states, actions, next_states, rewards = [], [], [], []
        
#         # For single-step prediction plotting
#         true_z_values = []
#         true_angle_values = []
#         true_x_velocity_values = []
        
#         predicted_z_values = []
#         predicted_angle_values = []
#         predicted_x_velocity_values = []
        
#         done = False
#         steps = 0
#         total_reward = 0
        
#         # Store initial state for full trajectory prediction
#         initial_state = state.copy()
        
#         while not done and steps < 1000:
#             # Get unnormalized state for optimization
#             unnorm_state = env.unnormalize(state)
#             action_seq = optimize_actions(
#                 dynamics_model, 
#                 unnorm_state, 
#                 horizon=horizon, 
#                 iterations=iterations,
#                 lr=lr
#             )
            
#             action = action_seq[0].detach().numpy()
#             next_state, reward, terminated, truncated, info = env.step(action)
#             total_reward += reward
            
#             # Store true values (unnormalized for plotting)
#             unnorm_state = env.unnormalize(state)
#             unnorm_next_state = env.unnormalize(next_state)
#             true_z_values.append(unnorm_state[0])
#             true_angle_values.append(unnorm_state[1])
#             true_x_velocity_values.append(unnorm_state[5])
            
#             # Get predicted next state
#             state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
#             action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
#             pred_next_state = dynamics_model.predict_mean(state_tensor, action_tensor).squeeze(0).detach().numpy()
            
#             # Unnormalize predicted next state for plotting
#             pred_unnorm_next_state = env.unnormalize(pred_next_state)
            
#             # Store predicted values
#             predicted_z_values.append(pred_unnorm_next_state[0])
#             predicted_angle_values.append(pred_unnorm_next_state[1])
#             predicted_x_velocity_values.append(pred_unnorm_next_state[5])
            
#             # Store unnormalized states 
#             states.append(unnorm_state)
#             actions.append(action)
#             next_states.append(unnorm_next_state)
#             rewards.append(reward)
            
#             state = next_state
#             steps += 1
#             done = terminated or truncated
        
#         trajectories.append({
#             'states': np.array(states),
#             'actions': np.array(actions),
#             'next_states': np.array(next_states),
#             'rewards': np.array(rewards)
#         })
        
#         print(info)
#         print(f"Episode {episode+1}: collected trajectory with {len(states)} steps and reward {total_reward}")
        
#         # Plot single-step predictions vs. true values
#         steps_range = range(len(true_z_values))
        
#         fig, axs = plt.subplots(3, 1, figsize=(8, 8))
        
#         # Z-position plot
#         axs[0].plot(steps_range, true_z_values, label='True z')
#         axs[0].plot(steps_range, predicted_z_values, label='Predicted z')
#         axs[0].set_title('Z Position (Single-step Prediction)')
#         axs[0].set_xlabel('Steps')
#         axs[0].set_ylabel('Z Position')
#         axs[0].legend()
        
#         # Angle plot
#         axs[1].plot(steps_range, true_angle_values, label='True angle')
#         axs[1].plot(steps_range, predicted_angle_values, label='Predicted angle')
#         axs[1].set_title('Angle (Single-step Prediction)')
#         axs[1].set_xlabel('Steps')
#         axs[1].set_ylabel('Angle')
#         axs[1].legend()
        
#         # X-velocity plot
#         axs[2].plot(steps_range, true_x_velocity_values, label='True x_velocity')
#         axs[2].plot(steps_range, predicted_x_velocity_values, label='Predicted x_velocity')
#         axs[2].set_title('X Velocity (Single-step Prediction)')
#         axs[2].set_xlabel('Steps')
#         axs[2].set_ylabel('X Velocity')
#         axs[2].legend()
        
#         plt.tight_layout()
#         plt.show()
#         plt.savefig(f'trajectory_plots_episode_{episode+1}_single_step.png')
#         plt.close()
        
#         # # Generate full trajectory prediction using only the model
#         # full_traj_pred_z = []
#         # full_traj_pred_angle = []
#         # full_traj_pred_x_velocity = []
        
#         # # Start from initial state (normalized)
#         # curr_state = torch.tensor(initial_state, dtype=torch.float32).unsqueeze(0)
        
#         # # Use the same actions as in the real trajectory
#         # for action_step in actions:
#         #     action_tensor = torch.tensor(action_step, dtype=torch.float32).unsqueeze(0)
#         #     pred_next_state = dynamics_model.predict_mean(curr_state, action_tensor)
            
#         #     # Extract values for plotting (unnormalized)
#         #     state_np = pred_next_state.squeeze(0).detach().numpy()
#         #     unnorm_state_np = env.unnormalize(state_np)
#         #     full_traj_pred_z.append(unnorm_state_np[0])
#         #     full_traj_pred_angle.append(unnorm_state_np[1])
#         #     full_traj_pred_x_velocity.append(unnorm_state_np[5])
            
#         #     # Update current state for next prediction
#         #     curr_state = pred_next_state
        
#         # # Plot full trajectory prediction vs. true values
#         # fig, axs = plt.subplots(3, 1, figsize=(10, 10))
        
#         # # Z-position plot
#         # axs[0].plot(steps_range, true_z_values, label='True z')
#         # axs[0].plot(steps_range, full_traj_pred_z, label='Model-only predicted z')
#         # axs[0].set_title('Z Position (Full Trajectory Prediction)')
#         # axs[0].set_xlabel('Steps')
#         # axs[0].set_ylabel('Z Position')
#         # axs[0].legend()
        
#         # # Angle plot
#         # axs[1].plot(steps_range, true_angle_values, label='True angle')
#         # axs[1].plot(steps_range, full_traj_pred_angle, label='Model-only predicted angle')
#         # axs[1].set_title('Angle (Full Trajectory Prediction)')
#         # axs[1].set_xlabel('Steps')
#         # axs[1].set_ylabel('Angle')
#         # axs[1].legend()
        
#         # # X-velocity plot
#         # axs[2].plot(steps_range, true_x_velocity_values, label='True x_velocity')
#         # axs[2].plot(steps_range, full_traj_pred_x_velocity, label='Model-only predicted x_velocity')
#         # axs[2].set_title('X Velocity (Full Trajectory Prediction)')
#         # axs[2].set_xlabel('Steps')
#         # axs[2].set_ylabel('X Velocity')
#         # axs[2].legend()

#         # do midpoint plot.
#         # full_traj_pred_z = []
#         # full_traj_pred_angle = []
#         # full_traj_pred_x_velocity = []

#         # # Start from the middle of the trajectory (normalized)
#         # mid_point = len(states) // 2
#         # curr_state = torch.tensor(env._normalize(states[mid_point]), dtype=torch.float32).unsqueeze(0)

#         # # Use the actions from mid-point onwards
#         # for action_step in actions[mid_point:]:
#         #     action_tensor = torch.tensor(action_step, dtype=torch.float32).unsqueeze(0)
#         #     pred_next_state = dynamics_model.predict_mean(curr_state, action_tensor)
            
#         #     # Extract values for plotting (unnormalized)
#         #     state_np = pred_next_state.squeeze(0).detach().numpy()
#         #     unnorm_state_np = env.unnormalize(state_np)
#         #     full_traj_pred_z.append(unnorm_state_np[0])
#         #     full_traj_pred_angle.append(unnorm_state_np[1])
#         #     full_traj_pred_x_velocity.append(unnorm_state_np[5])
            
#         #     # Update current state for next prediction
#         #     curr_state = pred_next_state

#         # # Plot full trajectory prediction vs. true values
#         # fig, axs = plt.subplots(3, 1, figsize=(10, 10))

#         # # Adjust steps_range for the half trajectory
#         # half_steps_range = range(mid_point, len(true_z_values))

#         # # Z-position plot
#         # axs[0].plot(half_steps_range, true_z_values[mid_point:], label='True z')
#         # axs[0].plot(half_steps_range, full_traj_pred_z, label='Model-only predicted z')
#         # axs[0].set_title('Z Position (Full Trajectory Prediction from Mid-point)')
#         # axs[0].set_xlabel('Steps')
#         # axs[0].set_ylabel('Z Position')
#         # axs[0].legend()

#         # # Angle plot
#         # axs[1].plot(half_steps_range, true_angle_values[mid_point:], label='True angle')
#         # axs[1].plot(half_steps_range, full_traj_pred_angle, label='Model-only predicted angle')
#         # axs[1].set_title('Angle (Full Trajectory Prediction from Mid-point)')
#         # axs[1].set_xlabel('Steps')
#         # axs[1].set_ylabel('Angle')
#         # axs[1].legend()

#         # # X-velocity plot
#         # axs[2].plot(half_steps_range, true_x_velocity_values[mid_point:], label='True x_velocity')
#         # axs[2].plot(half_steps_range, full_traj_pred_x_velocity, label='Model-only predicted x_velocity')
#         # axs[2].set_title('X Velocity (Full Trajectory Prediction from Mid-point)')
#         # axs[2].set_xlabel('Steps')
#         # axs[2].set_ylabel('X Velocity')
#         # axs[2].legend()
        
#         # plt.tight_layout()
#         # plt.show()
#         # plt.savefig(f'trajectory_plots_episode_{episode+1}_full_trajectory.png')
#         # plt.close()

#     # For n-step prediction errors
#     n_steps = [1, 5, 10, 20]  # Different prediction horizons to evaluate
#     n_step_errors = {
#         'z': {n: [] for n in n_steps},
#         'angle': {n: [] for n in n_steps},
#         'x_velocity': {n: [] for n in n_steps}
#     }

#     # Calculate n-step prediction errors along the trajectory
#     for start_idx in range(len(states) - max(n_steps)):
#         # Start from a real state (normalized)
#         start_state = torch.tensor(env._normalize(states[start_idx]), dtype=torch.float32).unsqueeze(0)
#         curr_state = start_state.clone()
        
#         # Make n-step predictions for each horizon
#         for n in n_steps:
#             if start_idx + n >= len(states):
#                 continue
                
#             # Apply n actions in sequence
#             for i in range(n):
#                 action_tensor = torch.tensor(actions[start_idx + i], dtype=torch.float32).unsqueeze(0)
#                 pred_next_state = dynamics_model.predict_mean(curr_state, action_tensor)
#                 curr_state = pred_next_state
            
#             # Calculate error with ground truth
#             pred_state_np = curr_state.squeeze(0).detach().numpy()
#             pred_unnorm = env.unnormalize(pred_state_np)
#             true_unnorm = states[start_idx + n]
            
#             # Calculate errors for each metric
#             z_error = abs(pred_unnorm[0] - true_unnorm[0])
#             angle_error = abs(pred_unnorm[1] - true_unnorm[1])
#             x_velocity_error = abs(pred_unnorm[5] - true_unnorm[5])
            
#             # Store errors
#             n_step_errors['z'][n].append(z_error)
#             n_step_errors['angle'][n].append(angle_error)
#             n_step_errors['x_velocity'][n].append(x_velocity_error)
            
#             # Reset current state for next n value
#             curr_state = start_state.clone()

#     # Plot n-step prediction errors
#     metrics = ['z', 'angle', 'x_velocity']
#     titles = ['Z Position', 'Angle', 'X Velocity']
#     fig, axs = plt.subplots(3, 1, figsize=(8, 8))

#     for i, (metric, title) in enumerate(zip(metrics, titles)):
#         for n in n_steps:
#             # Get valid error values
#             errors = n_step_errors[metric][n]
#             steps_range = range(len(errors))
            
#             # Plot this n-step error
#             axs[i].plot(steps_range, errors, label=f'{n}-step error')
        
#         axs[i].set_title(f'{title} Prediction Error')
#         axs[i].set_xlabel('Trajectory Step')
#         axs[i].set_ylabel(f'Absolute Error in {title}')
#         axs[i].legend()

#     plt.tight_layout()
#     plt.show()
#     plt.savefig(f'n_step_prediction_errors_episode_{episode+1}.png')
#     plt.close()
    
#     env.close()
#     return trajectories

# def eval_model(dynamics_model, n_evals=5):
#     # env = gym.make('Hopper-v5')
#     # env = NormalizeObservation(env)
#     avg_reward = 0
#     avg_steps = 0
    
#     for k in range(n_evals):
#         state, _ = env.reset()
#         total_reward = 0
#         episode_steps = 0
        
#         for step in range(1000):
#             # Unnormalize state for optimization
#             unnorm_state = env.unnormalize(state)
#             action_seq = optimize_actions(
#                 dynamics_model, 
#                 unnorm_state, 
#                 horizon=10, 
#                 iterations=50,
#                 lr=0.01
#             )
            
#             action = action_seq[0].detach().numpy()
#             next_state, reward, terminated, truncated, _ = env.step(action)
            
#             state = next_state
#             total_reward += reward
#             episode_steps += 1
            
#             if terminated or truncated:
#                 break
        
#         avg_reward += total_reward
#         avg_steps += episode_steps
#         print(f"Eval episode {k+1}: Steps = {episode_steps}, Reward = {total_reward:.2f}")
    
#     print(f"Avg steps: {avg_steps/n_evals:.1f}, Avg reward: {avg_reward/n_evals:.2f}")
#     env.close()
#     return avg_reward/n_evals

def collect_optimized_trajectories(dynamics_model, num_episodes=10, horizon=30, iterations=50, lr=0.01):
    trajectories = []
    
    # For tracking n-step errors across episodes
    n_steps = [1, 5, 10, 20]
    all_episodes_errors = {
        'z': {n: [] for n in n_steps},
        'angle': {n: [] for n in n_steps},
        'x_velocity': {n: [] for n in n_steps}
    }
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        states, actions, next_states, rewards = [], [], [], []
        
        # For single-step prediction plotting
        true_z_values = []
        true_angle_values = []
        true_x_velocity_values = []
        
        predicted_z_values = []
        predicted_angle_values = []
        predicted_x_velocity_values = []
        
        done = False
        steps = 0
        total_reward = 0
        
        # Store initial state for full trajectory prediction
        initial_state = state.copy()
        
        while not done and steps < 1000:
            # Get unnormalized state for optimization
            unnorm_state = env.unnormalize(state)
            action_seq = optimize_actions(
                dynamics_model, 
                unnorm_state, 
                horizon=horizon, 
                iterations=iterations,
                lr=lr
            )
            
            action = action_seq[0].detach().numpy()
            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Store true values (unnormalized for plotting)
            unnorm_state = env.unnormalize(state)
            unnorm_next_state = env.unnormalize(next_state)
            true_z_values.append(unnorm_state[0])
            true_angle_values.append(unnorm_state[1])
            true_x_velocity_values.append(unnorm_state[5])
            
            # Get predicted next state
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
            pred_next_state = dynamics_model.predict_mean(state_tensor, action_tensor).squeeze(0).detach().numpy()
            
            # Unnormalize predicted next state for plotting
            pred_unnorm_next_state = env.unnormalize(pred_next_state)
            
            # Store predicted values
            predicted_z_values.append(pred_unnorm_next_state[0])
            predicted_angle_values.append(pred_unnorm_next_state[1])
            predicted_x_velocity_values.append(pred_unnorm_next_state[5])
            
            # Store unnormalized states 
            states.append(unnorm_state)
            actions.append(action)
            next_states.append(unnorm_next_state)
            rewards.append(reward)
            
            state = next_state
            steps += 1
            done = terminated or truncated
        
        trajectories.append({
            'states': np.array(states),
            'actions': np.array(actions),
            'next_states': np.array(next_states),
            'rewards': np.array(rewards)
        })
        
        print(info)
        print(f"Episode {episode+1}: collected trajectory with {len(states)} steps and reward {total_reward}")
        
        # Plot single-step predictions vs. true values
        steps_range = range(len(true_z_values))
        
        fig, axs = plt.subplots(3, 1, figsize=(8, 8))
        
        # Z-position plot
        axs[0].plot(steps_range, true_z_values, label='True z')
        axs[0].plot(steps_range, predicted_z_values, label='Predicted z')
        axs[0].set_title('Z Position (Single-step Prediction)')
        axs[0].set_xlabel('Steps')
        axs[0].set_ylabel('Z Position')
        axs[0].legend()
        
        # Angle plot
        axs[1].plot(steps_range, true_angle_values, label='True angle')
        axs[1].plot(steps_range, predicted_angle_values, label='Predicted angle')
        axs[1].set_title('Angle (Single-step Prediction)')
        axs[1].set_xlabel('Steps')
        axs[1].set_ylabel('Angle')
        axs[1].legend()
        
        # X-velocity plot
        axs[2].plot(steps_range, true_x_velocity_values, label='True x_velocity')
        axs[2].plot(steps_range, predicted_x_velocity_values, label='Predicted x_velocity')
        axs[2].set_title('X Velocity (Single-step Prediction)')
        axs[2].set_xlabel('Steps')
        axs[2].set_ylabel('X Velocity')
        axs[2].legend()
        
        plt.tight_layout()
        plt.show()
        plt.savefig(f'trajectory_plots_episode_{episode+1}_single_step.png')
        plt.close()
        
        # For n-step prediction errors
        episode_n_step_errors = {
            'z': {n: [] for n in n_steps},
            'angle': {n: [] for n in n_steps},
            'x_velocity': {n: [] for n in n_steps}
        }

        # Calculate n-step prediction errors along the trajectory
        for start_idx in range(len(states) - max(n_steps)):
            # Start from a real state (normalized)
            start_state = torch.tensor(env._normalize(states[start_idx]), dtype=torch.float32).unsqueeze(0)
            curr_state = start_state.clone()
            
            # Make n-step predictions for each horizon
            for n in n_steps:
                if start_idx + n >= len(states):
                    continue
                    
                # Apply n actions in sequence
                for i in range(n):
                    action_tensor = torch.tensor(actions[start_idx + i], dtype=torch.float32).unsqueeze(0)
                    pred_next_state = dynamics_model.predict_mean(curr_state, action_tensor)
                    curr_state = pred_next_state
                
                # Calculate error with ground truth
                pred_state_np = curr_state.squeeze(0).detach().numpy()
                pred_unnorm = env.unnormalize(pred_state_np)
                true_unnorm = states[start_idx + n]
                
                # Calculate percentage errors for each metric (avoid division by zero)
                # For z position
                if abs(true_unnorm[0]) > 1e-6:
                    z_error = 100 * abs(pred_unnorm[0] - true_unnorm[0]) / abs(true_unnorm[0])
                else:
                    z_error = 0 if abs(pred_unnorm[0] - true_unnorm[0]) < 1e-6 else 100
                
                # For angle
                if abs(true_unnorm[1]) > 1e-6:
                    angle_error = 100 * abs(pred_unnorm[1] - true_unnorm[1]) / abs(true_unnorm[1])
                else:
                    angle_error = 0 if abs(pred_unnorm[1] - true_unnorm[1]) < 1e-6 else 100
                
                # For x velocity
                if abs(true_unnorm[5]) > 1e-6:
                    x_velocity_error = 100 * abs(pred_unnorm[5] - true_unnorm[5]) / abs(true_unnorm[5])
                else:
                    x_velocity_error = 0 if abs(pred_unnorm[5] - true_unnorm[5]) < 1e-6 else 100
                
                # Cap errors at 100%
                z_error = min(z_error, 100)
                angle_error = min(angle_error, 100)
                x_velocity_error = min(x_velocity_error, 100)
                
                # Store errors
                episode_n_step_errors['z'][n].append(z_error)
                episode_n_step_errors['angle'][n].append(angle_error)
                episode_n_step_errors['x_velocity'][n].append(x_velocity_error)
                
                # Reset current state for next n value
                curr_state = start_state.clone()
        
        # Store this episode's errors for averaging later
        for metric in ['z', 'angle', 'x_velocity']:
            for n in n_steps:
                all_episodes_errors[metric][n].append(episode_n_step_errors[metric][n])

        # Plot n-step prediction errors for this episode
        metrics = ['z', 'angle', 'x_velocity']
        titles = ['Z Position', 'Angle', 'X Velocity']
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))

        for i, (metric, title) in enumerate(zip(metrics, titles)):
            for n in n_steps:
                # Get valid error values
                errors = episode_n_step_errors[metric][n]
                steps_range = range(len(errors))
                
                # Plot this n-step error
                axs[i].plot(steps_range, errors, label=f'{n}-step error')
            
            axs[i].set_title(f'{title} Percentage Error')
            axs[i].set_xlabel('Trajectory Step')
            axs[i].set_ylabel(f'Percentage Error in {title} (%)')
            # Set y-axis limit to 100%
            axs[i].set_ylim(0, 100)
            axs[i].legend()

        plt.tight_layout()
        plt.show()
        plt.savefig(f'n_step_prediction_percentage_errors_episode_{episode+1}.png')
        plt.close()
    
    # Plot average n-step prediction errors across all episodes
    metrics = ['z', 'angle', 'x_velocity']
    titles = ['Z Position', 'Angle', 'X Velocity']
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        for n in n_steps:
            # Calculate average errors across episodes at each timestep
            # First, find the minimum length across all episodes
            min_length = min(len(errors) for errors in all_episodes_errors[metric][n])
            
            # Truncate all error arrays to this minimum length
            truncated_errors = [errors[:min_length] for errors in all_episodes_errors[metric][n]]
            
            # Calculate average at each timestep
            avg_errors = np.mean(truncated_errors, axis=0)
            steps_range = range(len(avg_errors))
            
            # Plot this average n-step error
            axs[i].plot(steps_range, avg_errors, label=f'{n}-step error')
        
        axs[i].set_title(f'Average {title} Percentage Error Across {num_episodes} Episodes')
        axs[i].set_xlabel('Trajectory Step')
        axs[i].set_ylabel(f'Average Percentage Error in {title} (%)')
        # Set y-axis limit to 100%
        axs[i].set_ylim(0, 100)
        axs[i].legend()

    plt.tight_layout()
    plt.show()
    plt.savefig(f'average_n_step_prediction_percentage_errors_{num_episodes}_episodes.png')
    plt.close()
    
    env.close()
    return trajectories

def main():
    random_trajectories = collect_data(num_episodes=1000, max_steps=1000)
    print(f"Collected {len(random_trajectories)} random trajectories")
    
    traj_sample = random_trajectories[0]
    state_dim = traj_sample['states'].shape[1]
    action_dim = traj_sample['actions'].shape[1]
    
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    
    all_trajectories = random_trajectories.copy()
    
    best_dynamics_model = None
    best_performance = -float('inf')
    
    num_iterations = 2000
    for iteration in range(num_iterations):
        print(f"\n===== ITERATION {iteration+1}/{num_iterations} =====")
        
        data_loader = prepare_training_data(all_trajectories, batch_size=64)
        
        dynamics_model = train_model(data_loader, state_dim, action_dim, epochs=20, lr=1e-3)
        
        print(f"Iteration {iteration+1}: Model training complete")
        horizon = 1
        print(horizon)
        print(f"Iteration {iteration+1}: Collecting optimized trajectories")
        optimized_trajectories = collect_optimized_trajectories(
            dynamics_model, 
            num_episodes=10, 
            horizon=horizon,
            iterations=50,
            lr=0.001
        )
        
        print(f"Iteration {iteration+1}: Collected {len(optimized_trajectories)} optimized trajectories")
        
        all_trajectories.extend(optimized_trajectories)
        print(f"Iteration {iteration+1}: Total trajectories: {len(all_trajectories)}")
    
    print("\n===== FINAL EVALUATION =====")
    print("Evaluating best model:")
    final_performance = eval_model(best_dynamics_model, n_evals=5)
    print(f"Best model performance: {final_performance:.2f}")

if __name__ == "__main__":
    main()