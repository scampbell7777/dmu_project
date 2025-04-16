import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import random
import copy

def collect_data(num_episodes=100, max_steps=200):
    env = gym.make('Hopper-v5')
    trajectories = []
    
    while len(trajectories) < num_episodes:
        state, _ = env.reset()
        states, actions, next_states, rewards = [], [], [], []
        
        for t in range(max_steps):
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(reward)
            
            state = next_state
            if terminated or truncated:
                break
        
        if len(states) > 8:
            trajectories.append({
                'states': np.array(states),
                'actions': np.array(actions),
                'next_states': np.array(next_states),
                'rewards': np.array(rewards)
            })
    
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

class StatePredictor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(StatePredictor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
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
        state_action = torch.cat([state, action], dim=-1)
        next_state = state
        step_size = 1.0 / n_euler_steps
        for _ in range(n_euler_steps):
            next_state = step_size * self.net(state_action) + next_state
            state_action = torch.cat([next_state, action], dim=-1)
        return next_state

class RewardPredictor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(RewardPredictor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim + state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state, action, next_state):
        inputs = torch.cat([state, action, next_state], dim=-1)
        return self.net(inputs)

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

def optimize_actions(dynamics_model, init_state, horizon=30, iterations=100, lr=1e-2):
    init_state_tensor = torch.tensor(init_state, dtype=torch.float32).unsqueeze(0)
    
    actions = nn.Parameter(torch.randn(horizon, 3) * 0.1)
    
    optimizer = torch.optim.Adam([actions], lr=lr)
    
    for i in range(iterations):
        optimizer.zero_grad()
        
        current_state = init_state_tensor.clone()
        states = [current_state]
        terminated = False
        
        for t in range(horizon):
            if terminated:
                break
                
            action = torch.tanh(actions[t]).unsqueeze(0)
            next_state = dynamics_model(current_state, action)
            states.append(next_state)
            
            terminated = check_termination(next_state.squeeze(0))
            
            current_state = next_state
        
        trajectory = torch.cat(states, dim=0)
        
        x_velocities = trajectory[:, 5] 
        angles = trajectory[:, 1] 
        velocity_loss = -torch.mean(x_velocities)
        
        angle_loss = torch.mean(angles ** 2)
        
        total_loss = 10 * angle_loss
        total_loss.backward()
        optimizer.step()
        
        if (i+1) % 20 == 0:
            with torch.no_grad():
                print(f"Iteration {i+1}/{iterations}, Loss: {total_loss:.4f}, " 
                      f"Vel: {-velocity_loss:.4f}, Angle: {angle_loss:.4f}")
    
    with torch.no_grad():
        final_actions = torch.tanh(actions)
    
    return final_actions

def collect_optimized_trajectories(dynamics_model, num_episodes=10, horizon=30, iterations=50, lr=0.01):
    env = gym.make('Hopper-v5')
    trajectories = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        states, actions, next_states, rewards = [], [], [], []
        
        done = False
        steps = 0
        
        while not done and steps < 500:
            action_seq = optimize_actions(
                dynamics_model, 
                state, 
                horizon=horizon, 
                iterations=iterations,
                lr=lr
            )
            
            action = action_seq[0].detach().numpy()
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(reward)
            
            state = next_state
            steps += 1
            done = terminated or truncated
        
        if len(states) > 8:
            trajectories.append({
                'states': np.array(states),
                'actions': np.array(actions),
                'next_states': np.array(next_states),
                'rewards': np.array(rewards)
            })
            print(f"Episode {episode+1}: collected trajectory with {len(states)} steps")
    
    env.close()
    return trajectories

def eval_model(dynamics_model, n_evals=5):
    env = gym.make('Hopper-v5')
    avg_reward = 0
    avg_steps = 0
    
    for k in range(n_evals):
        state, _ = env.reset()
        total_reward = 0
        episode_steps = 0
        
        for step in range(1000):
            action_seq = optimize_actions(
                dynamics_model, 
                state, 
                horizon=10, 
                iterations=50,
                lr=0.01
            )
            
            action = action_seq[0].detach().numpy()
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            state = next_state
            total_reward += reward
            episode_steps += 1
            
            if terminated or truncated:
                break
        
        avg_reward += total_reward
        avg_steps += episode_steps
        print(f"Eval episode {k+1}: Steps = {episode_steps}, Reward = {total_reward:.2f}")
    
    print(f"Avg steps: {avg_steps/n_evals:.1f}, Avg reward: {avg_reward/n_evals:.2f}")
    env.close()
    return avg_reward/n_evals

def main():
    random_trajectories = collect_data(num_episodes=100, max_steps=1000)
    print(f"Collected {len(random_trajectories)} random trajectories")
    
    traj_sample = random_trajectories[0]
    state_dim = traj_sample['states'].shape[1]
    action_dim = traj_sample['actions'].shape[1]
    
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    
    all_trajectories = random_trajectories.copy()
    
    best_dynamics_model = None
    best_performance = -float('inf')
    
    num_iterations = 5
    for iteration in range(num_iterations):
        print(f"\n===== ITERATION {iteration+1}/{num_iterations} =====")
        
        data_loader = prepare_training_data(all_trajectories, batch_size=64)
        dynamics_model = train_model(data_loader, state_dim, action_dim, epochs=100, lr=1e-3)
        
        print(f"Iteration {iteration+1}: Model training complete")
        
        print(f"Iteration {iteration+1}: Collecting optimized trajectories")
        optimized_trajectories = collect_optimized_trajectories(
            dynamics_model, 
            num_episodes=20, 
            horizon=10, 
            iterations=50, 
            lr=0.01
        )
        
        print(f"Iteration {iteration+1}: Collected {len(optimized_trajectories)} optimized trajectories")
        
        print(f"Iteration {iteration+1}: Evaluating current model")
        current_performance = eval_model(dynamics_model, n_evals=3)
        
        if current_performance > best_performance:
            best_performance = current_performance
            best_dynamics_model = copy.deepcopy(dynamics_model)
            print(f"Iteration {iteration+1}: New best model! Performance: {best_performance:.2f}")
        
        all_trajectories.extend(optimized_trajectories)
        print(f"Iteration {iteration+1}: Total trajectories: {len(all_trajectories)}")
    
    print("\n===== FINAL EVALUATION =====")
    print("Evaluating best model:")
    final_performance = eval_model(best_dynamics_model, n_evals=5)
    print(f"Best model performance: {final_performance:.2f}")

if __name__ == "__main__":
    main()