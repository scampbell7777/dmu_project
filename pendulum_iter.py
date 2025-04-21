import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import random
import copy

def collect_data(num_episodes=100, max_steps=200):
    env = gym.make('InvertedPendulum-v4')
    trajectories = []
    total_steps = 0
    
    while len(trajectories) < num_episodes:
        state, _ = env.reset()
        states, actions, next_states = [], [], []
        episode_steps = 0
        
        for t in range(max_steps):
            action = env.action_space.sample()
            next_state, _, terminated, truncated, _ = env.step(action)
            
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            
            state = next_state
            episode_steps += 1
            if terminated or truncated:
                break
        
        total_steps += episode_steps
        trajectories.append({
            'states': np.array(states),
            'actions': np.array(actions),
            'next_states': np.array(next_states)})
    
    env.close()
    return trajectories, total_steps

class StatePredictor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(StatePredictor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
    def forward(self, state, action, n_euler_steps=4):
        state_action = torch.cat([state, action], dim=-1)
        next_state = state
        step_size = 1.0 / n_euler_steps
        for _ in range(n_euler_steps):
            next_state = step_size * self.net(state_action) + next_state
            state_action = torch.cat([next_state, action], dim=-1)
        return next_state

def prepare_training_data(trajectories, batch_size=32):
    state_samples = []
    action_samples = []
    next_state_samples = []
    
    for traj in trajectories:
        states = traj['states']
        actions = traj['actions']
        next_states = traj['next_states']
        
        for i in range(len(states)):
            state_samples.append(states[i])
            action_samples.append(actions[i])
            next_state_samples.append(next_states[i])
    
    state_tensor = torch.tensor(np.array(state_samples), dtype=torch.float32)
    action_tensor = torch.tensor(np.array(action_samples), dtype=torch.float32)
    next_state_tensor = torch.tensor(np.array(next_state_samples), dtype=torch.float32)
    
    dataset = TensorDataset(state_tensor, action_tensor, next_state_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return data_loader

def train_model(data_loader, state_dim, action_dim, epochs=50, lr=5e-4):
    model = StatePredictor(state_dim, action_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        for states, actions, next_states in data_loader:
            optimizer.zero_grad()
            predicted_next_states = model(states, actions)
            loss = criterion(predicted_next_states, next_states)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {total_loss/num_batches:.6f}")
    
    return model

def optimize_actions(model, init_state, horizon=10, iterations=100, lr=1e-4):
    init_state_tensor = torch.tensor(init_state, dtype=torch.float32).unsqueeze(0)
    actions = nn.Parameter(torch.randn(horizon, 1) * 0.1)
    optimizer = torch.optim.Adam([actions], lr=lr)
    
    for i in range(iterations):
        optimizer.zero_grad()
        
        current_state = init_state_tensor.clone()
        total_angle_cost = 0.0
        discount = 1.0
        for t in range(horizon):
            action = torch.tanh(actions[t]) * 3.0
            action = action.unsqueeze(0)
            next_state = model(current_state, action)
            angle = next_state[0, 1]
            angle_cost = angle ** 2
            total_angle_cost *= discount
            total_angle_cost += angle_cost
            current_state = next_state.detach().clone()
        
        total_angle_cost.backward()
        optimizer.step()
    
    with torch.no_grad():
        final_actions = torch.tanh(actions) * 3.0
    
    return final_actions

def eval_model(model, horizon=10, iterations=100, lr=0.1, n_evals=10, max_steps=1000):
    env = gym.make('InvertedPendulum-v4')
    rewards = []
    trajectories = []
    total_steps = 0
    
    for k in range(n_evals):
        state, _ = env.reset()
        total_reward = 0
        states, actions, next_states = [], [], []
        episode_steps = 0
        
        for step in range(max_steps):
            action_seq = optimize_actions(
                model, 
                state, 
                horizon=horizon, 
                iterations=iterations,
                lr=lr
            )
            
            action = action_seq[0].detach().numpy()
            next_state, reward, terminated, truncated, _ = env.step([action.item()])
            
            states.append(state)
            actions.append([action.item()])
            next_states.append(next_state)
            
            state = next_state
            total_reward += reward
            episode_steps += 1
            
            if terminated or truncated:
                break
        
        total_steps += episode_steps
        rewards.append(total_reward)
        trajectories.append({
            'states': np.array(states),
            'actions': np.array(actions),
            'next_states': np.array(next_states)})
        
        print(f"Episode {k+1}: Reward = {total_reward:.2f}, Steps = {episode_steps}")
    
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    print(f"Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
    
    env.close()
    return avg_reward, std_reward, trajectories, total_steps

def main():
    max_iterations = 20
    all_trajectories = []
    all_data_points = []
    total_env_steps = 0
    
    initial_trajectories, steps = collect_data(num_episodes=10)
    all_trajectories.extend(initial_trajectories)
    total_env_steps += steps
    print(f"Collected {len(initial_trajectories)} random trajectories, Total steps: {total_env_steps}")
    
    traj_sample = all_trajectories[0]
    state_dim = traj_sample['states'].shape[1]
    action_dim = traj_sample['actions'].shape[1]
    
    for iteration in range(max_iterations):
        print(f"\nIteration {iteration+1}/{max_iterations}")
        
        data_loader = prepare_training_data(all_trajectories, batch_size=64)
        model = train_model(data_loader, state_dim, action_dim, epochs=10, lr=1e-3)
        
        avg_reward, std_reward, new_trajectories, eval_steps = eval_model(
            model, horizon=10, iterations=50, lr=1e-2, n_evals=1
        )
        # avg_reward, std_reward, _, _ = eval_model(
        #     model, horizon=10, iterations=50, lr=1e-2, n_evals=1
        # )
        
        all_trajectories.extend(new_trajectories)
        
        all_data_points.append({
            'total_steps': total_env_steps,
            'avg_reward': avg_reward,
            'std_reward': std_reward
        })
        total_env_steps += eval_steps
        if avg_reward == 1000:
            break
        
        print(f"Total environment steps: {total_env_steps}")
    
    plot_results(all_data_points)

def plot_results(data_points):
    steps = [point['total_steps'] for point in data_points]
    rewards = [point['avg_reward'] for point in data_points]
    stds = [point['std_reward'] for point in data_points]
    print(steps, rewards)
    plt.figure(figsize=(10, 6))
    plt.errorbar(steps, rewards, yerr=stds, marker='o', linestyle='-', capsize=5)
    plt.xlabel('Total Environment Steps')
    plt.ylabel('Average Reward')
    plt.title('Learning Progress: Reward vs. Environment Steps')
    plt.grid(True)
    plt.savefig('learning_progress.png')
    plt.show()

if __name__ == "__main__":
    main()