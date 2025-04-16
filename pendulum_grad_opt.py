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
    
    while len(trajectories) < num_episodes:
        state, _ = env.reset()
        states, actions, next_states = [], [], []
        
        for t in range(max_steps):
            action = env.action_space.sample()
            next_state, _, terminated, truncated, _ = env.step(action)
            
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            
            state = next_state
            if terminated or truncated:
                break
        
        if len(states) > 8:  # Need at least a few transitions for a trajectory
            trajectories.append({
                'states': np.array(states),
                'actions': np.array(actions),
                'next_states': np.array(next_states)
            })
    
    env.close()
    return trajectories

# Network to learn dynamics i.e. the model
class StatePredictor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(StatePredictor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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

def simulate_model(model, init_state, action_sequence):
    init_state_tensor = torch.tensor(init_state, dtype=torch.float32).unsqueeze(0)
    states = [init_state_tensor]
    
    for action in action_sequence:
        action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        next_state = model(states[-1], action_tensor)
        states.append(next_state)
    
    return torch.stack(states).squeeze(1)

def optimize_actions(model, init_state, horizon=10, iterations=100, lr=1e-4):
    init_state_tensor = torch.tensor(init_state, dtype=torch.float32).unsqueeze(0)
    actions = nn.Parameter(torch.randn(horizon, 1) * 0.1)
    # Only actions are optimized. Not the dynamics model.
    optimizer = torch.optim.Adam([actions], lr=lr)
    
    for i in range(iterations):
        optimizer.zero_grad()
        
        current_state = init_state_tensor.clone()
        total_angle_cost = 0.0
        # Discount future states because our dynamics model will accrue larger errors in later timesteps.
        discount = 1.0
        for t in range(horizon):
            # Clamp actions in a differentiable way
            action = torch.tanh(actions[t]) * 3.0
            action = action.unsqueeze(0)  # Add batch dimension
            next_state = model(current_state, action, n_euler_steps=16)
            angle = next_state[0, 1]
            angle_cost = angle ** 2
            total_angle_cost*=discount
            total_angle_cost += angle_cost
            current_state = next_state.detach().clone()
        
        total_angle_cost.backward()
        optimizer.step()
        
        if (i+1) % 5 == 0:
            print(f"Iteration {i+1}/{iterations}, Cost: {total_angle_cost.item():.6f}")
    
    with torch.no_grad():
        final_actions = torch.tanh(actions) * 3.0
    
    return final_actions

def eval_gradient_actions(model, horizon=10, iterations=100, lr=0.1, n_evals=100):
    print(f"Evaluating gradient-based action optimization: horizon={horizon}, iterations={iterations}")
    env = gym.make('InvertedPendulum-v4')
    avg_reward = 0
    
    for k in range(n_evals):
        state, _ = env.reset()
        total_reward = 0
        
        for step in range(1000):
            action_seq = optimize_actions(
                model, 
                state, 
                horizon=horizon, 
                iterations=iterations,
                lr=lr
            )
            
            # Take the first action from the sequence
            action = action_seq[0].detach().numpy()
            next_state, reward, terminated, truncated, _ = env.step([action.item()])
            state = next_state
            total_reward += reward
            print(total_reward)
            if terminated or truncated:
                break
        
        avg_reward += total_reward
        print(f"Episode {k+1}: Reward = {total_reward:.2f}")
    
    print(f"Average reward over {n_evals} episodes: {avg_reward / n_evals:.2f}")
    env.close()

def main():
    # 1) Collect initial data from random actions
    trajectories = collect_data(num_episodes=1000)
    print(f"Collected {len(trajectories)} random trajectories")
    
    traj_sample = trajectories[0]
    state_dim = traj_sample['states'].shape[1]
    action_dim = traj_sample['actions'].shape[1]
    
    # Train initial model
    data_loader = prepare_training_data(trajectories, batch_size=64)
    model = train_model(data_loader, state_dim, action_dim, epochs=100, lr=1e-3)
    print("Initial model training complete")
    
    # Evaluate gradient-based action optimization
    print("Gradient-based action optimization evaluation:")
    eval_gradient_actions(model, horizon=10, iterations=50, lr=1e-2, n_evals=10)

if __name__ == "__main__":
    main()