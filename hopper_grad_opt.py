import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import random
import copy

def collect_data(num_episodes=100, max_steps=200):
    # Switch to Hopper environment
    env = gym.make('Hopper-v5')
    trajectories = []
    
    while len(trajectories) < num_episodes:
        state, _ = env.reset()
        states, actions, next_states, rewards = [], [], [], []
        
        for t in range(max_steps):
            # Sample from Hopper's action space (3D continuous actions)
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(reward)
            
            state = next_state
            if terminated or truncated:
                break
        
        if len(states) > 8:  # Need at least a few transitions for a trajectory
            trajectories.append({
                'states': np.array(states),
                'actions': np.array(actions),
                'next_states': np.array(next_states),
                'rewards': np.array(rewards)
            })
    
    env.close()
    return trajectories

# Check if Hopper state is healthy based on the criteria
def is_healthy(observation, exclude_current_positions_from_observation=True):
    # Default parameters
    healthy_state_range = [-100.0, 100.0]
    healthy_z_range = [0.7, float('inf')]
    healthy_angle_range = [-0.2, 0.2]
    
    # Adjust indices based on exclude_current_positions_from_observation
    if exclude_current_positions_from_observation:
        # 1. Check if elements observation[1:] are within healthy_state_range
        state_healthy = all(healthy_state_range[0] <= x <= healthy_state_range[1] for x in observation[1:])
        
        # 2. Check if height (observation[0]) is within healthy_z_range
        z_healthy = healthy_z_range[0] <= observation[0] <= healthy_z_range[1]
        
        # 3. Check if angle (observation[1]) is within healthy_angle_range
        angle_healthy = healthy_angle_range[0] <= observation[1] <= healthy_angle_range[1]
    else:
        # If we include positions, indices are shifted
        state_healthy = all(healthy_state_range[0] <= x <= healthy_state_range[1] for x in observation[2:])
        z_healthy = healthy_z_range[0] <= observation[1] <= healthy_z_range[1]
        angle_healthy = healthy_angle_range[0] <= observation[2] <= healthy_angle_range[1]
    
    return state_healthy and z_healthy and angle_healthy

# Network to learn dynamics i.e. the model
class StatePredictor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(StatePredictor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Increased network capacity for more complex dynamics
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)
        )
        
    def forward(self, state, action, n_euler_steps=16):
        # Concatenate state and action
        state_action = torch.cat([state, action], dim=-1)
        # If n_steps is greater than 1, this is basically Euler's method
        # for integrating a neural ODE
        next_state = state
        step_size = 1.0 / n_euler_steps
        for _ in range(n_euler_steps):
            next_state = step_size * self.net(state_action) + next_state
            state_action = torch.cat([next_state, action], dim=-1)
        return next_state


# Process trajectories into batched training data for the dynamics model.
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
    
    state_tensor = torch.tensor(np.array(state_samples), dtype=torch.float32)
    action_tensor = torch.tensor(np.array(action_samples), dtype=torch.float32)
    next_state_tensor = torch.tensor(np.array(next_state_samples), dtype=torch.float32)
    reward_tensor = torch.tensor(np.array(reward_samples), dtype=torch.float32).unsqueeze(1)
    
    dataset = TensorDataset(state_tensor, action_tensor, next_state_tensor, reward_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return data_loader

def train_model(data_loader, state_dim, action_dim, epochs=50, lr=5e-4):
    dynamics_model = StatePredictor(state_dim, action_dim)
    
    dynamics_optimizer = torch.optim.Adam(dynamics_model.parameters(), lr=lr)
    dynamics_criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        total_dynamics_loss = 0.0
        num_batches = 0
        
        for states, actions, next_states, _ in data_loader:
            # Train dynamics model
            dynamics_optimizer.zero_grad()
            predicted_next_states = dynamics_model(states, actions)
            dynamics_loss = dynamics_criterion(predicted_next_states, next_states)
            dynamics_loss.backward()
            dynamics_optimizer.step()
            
            total_dynamics_loss += dynamics_loss.item()
            num_batches += 1
        
        print(f"Epoch {epoch+1}/{epochs}, Dynamics Loss: {total_dynamics_loss/num_batches:.6f}")
    
    return dynamics_model

def simulate_model(dynamics_model, reward_model, init_state, action_sequence):
    init_state_tensor = torch.tensor(init_state, dtype=torch.float32).unsqueeze(0)
    states = [init_state_tensor]
    rewards = []
    
    for action in action_sequence:
        action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        next_state = dynamics_model(states[-1], action_tensor)
        
        # Predict reward
        reward = reward_model(states[-1], action_tensor, next_state)
        
        states.append(next_state)
        rewards.append(reward)
    
    return torch.stack(states).squeeze(1), torch.stack(rewards).squeeze(1)

# Check if the Hopper has fallen or is unhealthy based on our model predictions
def check_termination(state_tensor, exclude_current_positions_from_observation=True):
    healthy_state_range = [-100.0, 100.0]
    healthy_z_range = [0.7, float('inf')]
    healthy_angle_range = [-0.2, 0.2]
    
    if exclude_current_positions_from_observation:
        # Check height (z)
        z_healthy = healthy_z_range[0] <= state_tensor[0].item() <= healthy_z_range[1]
        
        # Check angle
        angle_healthy = healthy_angle_range[0] <= state_tensor[1].item() <= healthy_angle_range[1]
        
        # Check other state components
        state_healthy = all(healthy_state_range[0] <= x.item() <= healthy_state_range[1] 
                            for x in state_tensor[1:])
    else:
        z_healthy = healthy_z_range[0] <= state_tensor[1].item() <= healthy_z_range[1]
        angle_healthy = healthy_angle_range[0] <= state_tensor[2].item() <= healthy_angle_range[1]
        state_healthy = all(healthy_state_range[0] <= x.item() <= healthy_state_range[1] 
                            for x in state_tensor[2:])
    
    return not (z_healthy and angle_healthy and state_healthy)

def optimize_actions(dynamics_model, init_state, horizon=30, iterations=100, lr=1e-2):
    dynamics_model.eval()
    init_state_tensor = torch.tensor(init_state, dtype=torch.float32).unsqueeze(0)
    
    # Initialize action sequence for Hopper (3 dimensions per action)
    actions = nn.Parameter(torch.randn(horizon, 3) * 0.1)
    
    # Only actions are optimized. Not the dynamics model.
    optimizer = torch.optim.Adam([actions], lr=lr)
    
    for i in range(iterations):
        optimizer.zero_grad()
        
        current_state = init_state_tensor.clone()
        states = [current_state]
        terminated = False
        
        # Simulate forward using our dynamics model
        for t in range(horizon):
            if terminated:
                break
                
            # Clamp actions in a differentiable way
            action = torch.tanh(actions[t]).unsqueeze(0)  # Add batch dimension
            next_state = dynamics_model(current_state, action)
            states.append(next_state)
            
            # Check for termination
            terminated = check_termination(next_state.squeeze(0))
            
            current_state = next_state
        
        # Calculate our custom loss based on the trajectory
        trajectory = torch.cat(states, dim=0)
        
        # Calculate the x-velocity component (should be in observation space)
        # For Hopper, velocity components typically start after position components
        # With exclude_current_positions_from_observation=True, velocities start at index 5
        x_velocities = trajectory[:, 5] 
        angles = trajectory[:, 1] 
        velocity_loss = -torch.mean(x_velocities)
        
        angle_loss = torch.mean(angles ** 2)
        
        # 3. Add termination penalty
        # termination_penalty = 0.0
        # if terminated:
        #     termination_step = len(states) - 1
        #     termination_penalty = 100.0 * (horizon - termination_step)
        
        # Combine losses with appropriate weights
        # total_loss = velocity_loss + 100.0 * angle_loss #+ termination_penalty
        total_loss = 10 *  angle_loss
        total_loss.backward()
        optimizer.step()
        
        if (i+1) % 20 == 0:
            with torch.no_grad():
                print(f"Iteration {i+1}/{iterations}, Loss: {total_loss:.4f}, " 
                      f"Vel: {-velocity_loss:.4f}, Angle: {angle_loss:.4f}")
    
    with torch.no_grad():
        final_actions = torch.tanh(actions)  # Scale to [-1, 1]
    
    return final_actions

def eval_gradient_actions(dynamics_model, horizon=30, iterations=100, lr=0.01, n_evals=10):
    print(f"Evaluating gradient-based action optimization: horizon={horizon}, iterations={iterations}")
    env = gym.make('Hopper-v5')
    avg_reward = 0
    
    for k in range(n_evals):
        state, _ = env.reset()
        total_reward = 0
        episode_steps = 0
        
        for step in range(1000):  # Max episode length
            action_seq = optimize_actions(
                dynamics_model, 
                state, 
                horizon=horizon, 
                iterations=iterations,
                lr=lr
            )
            
            # Take the first action from the sequence
            action = action_seq[0].detach().numpy()
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            state = next_state
            total_reward += reward
            episode_steps += 1
            
            if terminated or truncated:
                break
        
        avg_reward += total_reward
        print(f"Episode {k+1}: Steps = {episode_steps}, Reward = {total_reward:.2f}")
    
    print(f"Average reward over {n_evals} episodes: {avg_reward / n_evals:.2f}")
    env.close()

def main():
    # 1) Collect initial data from random actions
    trajectories = collect_data(num_episodes=100, max_steps=500)
    print(f"Collected {len(trajectories)} random trajectories")
    
    traj_sample = trajectories[0]
    state_dim = traj_sample['states'].shape[1]  # Should be 11 for Hopper
    action_dim = traj_sample['actions'].shape[1]  # Should be 3 for Hopper
    
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    
    data_loader = prepare_training_data(trajectories, batch_size=64)
    dynamics_model = train_model(data_loader, state_dim, action_dim, epochs=100, lr=1e-3)
    print("Initial model training complete")
    
    print("Gradient-based action optimization evaluation:")
    eval_gradient_actions(dynamics_model, horizon=10, iterations=50, lr=0.01, n_evals=3)

if __name__ == "__main__":
    main()