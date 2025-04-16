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
        
    def forward(self, state, action):
        # Concatenate state and action
        state_action = torch.cat([state, action], dim=-1)
        # If n_steps is greater than 1, this is basically Euler's method
        # for integrating a neural ODE
        next_state = state
        n_steps = 1
        step_size = 1.0 / n_steps
        for _ in range(n_steps):
            next_state = step_size * self.net(state_action) + next_state
            state_action = torch.cat([next_state, action], dim=-1)
        return next_state

# Network to do imitation learning. Predicts actions given states.
class ActionPredictor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActionPredictor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Output between -1 and 1, will be scaled later
        )
        
    def forward(self, state):
        # Output actions in range -1 to 1
        actions = self.net(state)
        # Scale to action range for InvertedPendulum (-3 to 3)
        return 3.0 * actions

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

# Given trajectories from the dynamics model, batch them up so the imitation model can be
# trained.
def prepare_action_prediction_data(trajectories, batch_size=32, fraction_to_select=0.5):
    # Sort trajectories by length (number of states)
    sorted_trajectories = sorted(trajectories, key=lambda x: len(x['states']), reverse=True)
    
    # Select top X% longest trajectories
    num_top_trajectories = max(1, int(len(sorted_trajectories) * fraction_to_select))
    top_trajectories = sorted_trajectories[:num_top_trajectories]
    
    state_samples = []
    action_samples = []
    
    for traj in top_trajectories:
        states = traj['states']
        actions = traj['actions']
        
        for i in range(len(states)):
            state_samples.append(states[i])
            action_samples.append(actions[i])
    
    state_tensor = torch.tensor(np.array(state_samples), dtype=torch.float32)
    action_tensor = torch.tensor(np.array(action_samples), dtype=torch.float32)
    dataset = TensorDataset(state_tensor, action_tensor)
    
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

def train_action_predictor(data_loader, state_dim, action_dim, epochs=50, lr=5e-4):
    action_model = ActionPredictor(state_dim, action_dim)
    optimizer = torch.optim.Adam(action_model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        for states, target_actions in data_loader:
            optimizer.zero_grad()
            predicted_actions = action_model(states)
            loss = criterion(predicted_actions, target_actions)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        print(f"Epoch {epoch+1}/{epochs}, Action Predictor Avg Loss: {total_loss/num_batches:.6f}")
    
    return action_model

def simulate_model(model, init_state, action_sequence):
    init_state_tensor = torch.tensor(init_state, dtype=torch.float32).unsqueeze(0)
    states = [init_state_tensor]
    
    for action in action_sequence:
        action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        next_state = model(states[-1], action_tensor)
        states.append(next_state)
    
    return torch.stack(states).squeeze(1)

# Figure out if/when the pendulum fell over
def find_first_threshold_crossing(tensor, threshold=0.2, dim_index=1):
    abs_values = torch.abs(tensor[:, dim_index])
    indices = torch.where(abs_values > threshold)[0]
    
    # Return the first index if any exist, otherwise return tensor length
    if indices.numel() > 0:
        return indices[0].item()
    else:
        return tensor.shape[0]

def eval_action_predictor(dynamics_model, action_model, n_evals=100):
    print(f"Evaluating action predictor model")
    env = gym.make('InvertedPendulum-v4')
    avg_reward = 0
    
    for k in range(n_evals):
        state, _ = env.reset()
        total_reward = 0
        for step in range(1000):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = action_model(state_tensor).squeeze(0).numpy()
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            total_reward += reward
            
            if terminated or truncated:
                break
        
        avg_reward += total_reward
        print(f"Episode {k+1}: Reward = {total_reward:.2f}")
    
    print(f"Average reward over {n_evals} episodes: {avg_reward / n_evals:.2f}")
    env.close()

def cross_entropy_method(model, init_state, horizon=30, num_samples=100, iterations=5, elite_frac=0.2):
    init_state_tensor = torch.tensor(init_state, dtype=torch.float32)
    action_dim = 1
    action_mean = torch.zeros(horizon, action_dim)
    action_std = torch.ones(horizon, action_dim) * 3.0
    best_cost = float('inf')
    best_action_seq = None
    num_elite = max(1, int(num_samples * elite_frac))
    
    for iteration in range(iterations):
        action_seqs = torch.normal(
            action_mean.unsqueeze(0).repeat(num_samples, 1, 1),
            action_std.unsqueeze(0).repeat(num_samples, 1, 1)
        )
        
        action_seqs = torch.clamp(action_seqs, -3, 3)
        costs = []
        for i in range(num_samples):
            action_seq = action_seqs[i]
            current_state = init_state_tensor.unsqueeze(0)
            states = [current_state]
            
            for t in range(horizon):
                action = action_seq[t].unsqueeze(0)
                next_state = model(current_state, action)
                states.append(next_state)
                current_state = next_state
            
            trajectory = torch.stack(states).squeeze(1)
            # If angle is greater than 0.2, the episode is consider failed.
            fall_idx = find_first_threshold_crossing(trajectory, threshold=0.2, dim_index=1)
            if fall_idx < horizon:
                failure_cost = horizon - fall_idx
            else:
                failure_cost = 0.0
            
            costs.append(failure_cost)
            
            if failure_cost < best_cost:
                best_cost = failure_cost
                best_action_seq = action_seq.clone()
        
        costs = torch.tensor(costs)
        elite_indices = torch.argsort(costs)[:num_elite]
        elite_actions = action_seqs[elite_indices]
        action_mean = elite_actions.mean(dim=0)
        action_std = elite_actions.std(dim=0) + 1e-5
        
        action_std = action_std * (1.0 - 0.1 * iteration / iterations)
    
    return best_action_seq, best_cost

def eval_ce(model, horizon, num_samples, iterations=5, elite_frac=0.2, n_evals=100):
    print(f"Evaluating model with CEM: horizon={horizon}, samples={num_samples}, iterations={iterations}")
    env = gym.make('InvertedPendulum-v4')
    avg_reward = 0
    for k in range(n_evals):
        state, _ = env.reset()
        total_reward = 0.0
        for step in range(1000):
            action_seq, cost = cross_entropy_method(
                model, 
                state, 
                horizon=horizon, 
                num_samples=num_samples,
                iterations=iterations,
                elite_frac=elite_frac
            )
            
            # Take the first action from the sequence
            action = action_seq[0].detach().numpy()
            next_state, reward, terminated, truncated, _ = env.step([action.item()])
            state = next_state
            total_reward += reward
            
            if terminated or truncated:
                break
        
        avg_reward += total_reward
        print(f"Episode {k+1}: Reward = {total_reward:.2f}")
    
    print(f"Average reward over 100 episodes: {avg_reward / n_evals:.2f}")
    env.close()

def main():
    # 1) Collect initial data from random actions
    trajectories = collect_data(num_episodes=1000)
    print(f"Collected {len(trajectories)} random trajectories")
    
    traj_sample = trajectories[0]
    state_dim = traj_sample['states'].shape[1]
    action_dim = traj_sample['actions'].shape[1]
    
    all_trajectories = trajectories.copy()
    
    # Train initial model
    data_loader = prepare_training_data(all_trajectories, batch_size=64)
    model = train_model(data_loader, state_dim, action_dim, epochs=100, lr=1e-3)
    print("Initial model training complete")
    
    # Collect improved trajectories
    improved_trajectories = collect_improved_data(model, num_episodes=20, max_steps=500)
    print(f"Collected {len(improved_trajectories)} improved trajectories")
    
    # Train action prediction model using top X % longest trajectories
    action_data_loader = prepare_action_prediction_data(improved_trajectories, batch_size=64, fraction_to_select=0.1)
    action_model = train_action_predictor(action_data_loader, state_dim, action_dim, epochs=200, lr=1e-4)
    print("Action prediction model training complete")
    # Evaluate action prediction model
    print("Action predictor model evaluation:")
    eval_action_predictor(model, action_model, n_evals=100)
    
    # For comparison, eval the dynamics model
    # print("Final model evaluation:")
    # eval_ce(model, horizon=10, num_samples=40, iterations=5, elite_frac=0.2)
    
def collect_improved_data(model, num_episodes=1000, max_steps=500):
    env = gym.make('InvertedPendulum-v4')
    trajectories = []
    
    while len(trajectories) < num_episodes:
        state, _ = env.reset()
        states, actions, next_states = [], [], []
        
        for t in range(max_steps):
            action_seq, _ = cross_entropy_method(
                model, 
                state, 
                horizon=10,
                num_samples=40,
                iterations=5,
                elite_frac=0.2
            )
            
            action = action_seq[0].detach().numpy()
            next_state, _, terminated, truncated, _ = env.step([action.item()])
            
            states.append(state)
            actions.append([action.item()])
            next_states.append(next_state)
            
            state = next_state
            if terminated or truncated:
                break
        
        if len(states) > 8:  # Need at least a few transitions for a trajectory
            print(len(states))
            trajectories.append({
                'states': np.array(states),
                'actions': np.array(actions),
                'next_states': np.array(next_states)
            })
    
    env.close()
    return trajectories

if __name__ == "__main__":
    main()