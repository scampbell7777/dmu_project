

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
            #action = np.zeros_like(action)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
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

class ModelMember(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ModelMember, self).__init__()
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
        
    def forward(self, state, action, n_euler_steps=1):
        state_action = torch.cat([state, action], dim=-1)
        next_state = state
        step_size = 1.0 / n_euler_steps
        for _ in range(n_euler_steps):
            next_state = step_size * self.net(state_action) + next_state
            state_action = torch.cat([next_state, action], dim=-1)
        return next_state

class StatePredictor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, ensemble_size=3):
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
    
    state_tensor = torch.tensor(np.array(state_samples), dtype=torch.float32)
    action_tensor = torch.tensor(np.array(action_samples), dtype=torch.float32)
    next_state_tensor = torch.tensor(np.array(next_state_samples), dtype=torch.float32)
    reward_tensor = torch.tensor(np.array(reward_samples), dtype=torch.float32).unsqueeze(1)
    
    dataset = TensorDataset(state_tensor, action_tensor, next_state_tensor, reward_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return data_loader

def train_model(data_loader, state_dim, action_dim, epochs=50, lr=5e-4, model=None):
    if model is not None:
        dynamics_model = model
    else:
        dynamics_model = StatePredictor(state_dim, action_dim)
    
    ensemble_size = dynamics_model.ensemble_size
    
    optimizers = [torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4) for model in dynamics_model.models]
    criterion = nn.MSELoss()
    
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
                dynamics_loss = criterion(predicted_next_states, batch_next_states)
                dynamics_loss.backward()
                optimizer.step()
                
                total_dynamics_losses[model_idx] += dynamics_loss.item()
            
            num_batches += 1
        
        avg_losses = [loss/num_batches for loss in total_dynamics_losses]
        print(f"Epoch {epoch+1}/{epochs}, Dynamics Losses: {avg_losses}")
    
    return dynamics_model

def optimize_actions(dynamics_model, init_state, horizon=30, iterations=10, lr=1e-1, gamma=1.0):
    init_state_tensor = torch.tensor(init_state, dtype=torch.float32).unsqueeze(0)
    
    actions = nn.Parameter(torch.randn(horizon, 3) * 0.01)
    # actions = nn.Parameter(torch.rand(horizon, 3) * 2 - 1)
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
            x_velocities = trajectory[:, 5]
            angles = trajectory[:, 1]
            z = trajectory[:, 0]
            is_healthy = torch.tanh((z - 0.7) * 40) * torch.tanh((0.2 - torch.abs(angles)) * 40)

            model_loss = -is_healthy.mean() - (x_velocities*is_healthy).mean()
            total_loss += model_loss / dynamics_model.ensemble_size
        total_loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        final_actions = torch.tanh(actions)
    
    return final_actions

def optimize_actions_cem(dynamics_model, init_state, horizon=30, iterations=5, 
                        population_size=500, elite_fraction=0.1, alpha=0.1):
    init_state_tensor = torch.tensor(init_state, dtype=torch.float32).unsqueeze(0)
    num_elites = int(population_size * elite_fraction)
    
    # Initialize mean and std for sampling
    mu = torch.zeros(horizon, 3)
    sigma = torch.ones(horizon, 3)
    
    best_actions = None
    best_reward = float('-inf')
    
    for i in range(iterations):
        # Sample actions from current distribution
        action_samples = torch.randn(population_size, horizon, 3) * sigma.unsqueeze(0) + mu.unsqueeze(0)
        action_samples = torch.clamp(action_samples, -1, 1)  # Constrain to [-1, 1]
        rewards = torch.zeros(population_size)
        
        # Evaluate all sampled action sequences
        for s in range(population_size):
            total_reward = 0
            
            for model_idx in range(dynamics_model.ensemble_size):
                current_state = init_state_tensor.clone()
                reward_sum = 0
                valid_trajectory = True
                
                for t in range(horizon):
                    action = action_samples[s, t].unsqueeze(0)
                    next_state = dynamics_model(current_state, action, model_idx=model_idx)
                    
                    angle = next_state[0, 1]
                    z = next_state[0, 0]
                    x_velocity = next_state[0, 5]
                    
                    # if abs(angle) < 0.2 or z < 0.7:
                    #     valid_trajectory = False
                    #     break
                    
                    step_reward = 1 + 0.08 * x_velocity
                    reward_sum += step_reward
                    current_state = next_state
                
                if not valid_trajectory:
                    total_reward += -1 / dynamics_model.ensemble_size
                else:
                    total_reward += reward_sum / dynamics_model.ensemble_size
            
            rewards[s] = total_reward
            
            # Track best solution found across all iterations
            if rewards[s] > best_reward:
                best_reward = rewards[s]
                best_actions = action_samples[s].clone()
        
        # Select elite samples
        elite_idxs = torch.topk(rewards, num_elites).indices
        elite_samples = action_samples[elite_idxs]
        
        # Update distribution parameters
        new_mu = elite_samples.mean(dim=0)
        new_sigma = elite_samples.std(dim=0)
        
        # Smooth update
        mu = alpha * new_mu + (1 - alpha) * mu
        sigma = alpha * new_sigma + (1 - alpha) * sigma
    
    return best_actions

def optimize_actions_random_shooting(dynamics_model, init_state, horizon=30, num_samples=1000):
    init_state_tensor = torch.tensor(init_state, dtype=torch.float32).unsqueeze(0)
    action_samples = torch.rand(num_samples, horizon, 3) * 2 - 1  # Uniform between -1 and 1
    rewards = torch.zeros(num_samples)
    
    for s in range(num_samples):
        total_reward = 0
        valid_trajectory = True
        
        for model_idx in range(dynamics_model.ensemble_size):
            current_state = init_state_tensor.clone()
            reward_sum = 0
            
            for t in range(horizon):
                action = action_samples[s, t].unsqueeze(0)
                next_state = dynamics_model(current_state, action, model_idx=model_idx)
                
                angle = next_state[0, 1]
                z = next_state[0, 0]
                x_velocity = next_state[0, 5]
                
                if abs(angle) < 0.2 or z < 0.7:
                    valid_trajectory = False
                    break
                
                step_reward = 1 + 0.008 * x_velocity
                reward_sum += step_reward
                
                current_state = next_state
            
            if not valid_trajectory:
                total_reward += -1 / dynamics_model.ensemble_size
            else:
                total_reward += reward_sum / dynamics_model.ensemble_size
        
        rewards[s] = total_reward
    
    best_idx = torch.argmax(rewards)
    return action_samples[best_idx]

def collect_optimized_trajectories(dynamics_model, num_episodes=10, horizon=30, iterations=50, lr=0.01, method='gradient'):
    env = gym.make('Hopper-v5')
    trajectories = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        states, actions, next_states, rewards = [], [], [], []
        
        done = False
        steps = 0
        total_reward = 0
        while not done and steps < 1000:
            if method == 'gradient':
                action_seq = optimize_actions(
                    dynamics_model, 
                    state, 
                    horizon=horizon, 
                    iterations=iterations,
                    lr=lr
                )
            elif method == 'random_shooting':
                action_seq = optimize_actions_random_shooting(
                    dynamics_model, 
                    state, 
                    horizon=horizon, 
                    num_samples=iterations
                )
            elif method == 'cem':
                action_seq = optimize_actions_cem(
                    dynamics_model, 
                    state, 
                    horizon=5, iterations=5, 
                    population_size=1000, elite_fraction=0.1, alpha=0.0)
            
            action = action_seq[0].detach().numpy()
            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward+=reward
            
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
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
            next_state, reward, terminated, truncated, info = env.step(action)
            
            state = next_state
            total_reward += reward
            episode_steps += 1
            
            if terminated or truncated:
                break
        print(info)
        avg_reward += total_reward
        avg_steps += episode_steps
        print(f"Eval episode {k+1}: Steps = {episode_steps}, Reward = {total_reward:.2f}")
    
    print(f"Avg steps: {avg_steps/n_evals:.1f}, Avg reward: {avg_reward/n_evals:.2f}")
    env.close()
    return avg_reward/n_evals

def main():
    random_trajectories = collect_data(num_episodes=20, max_steps=1000)
    print(f"Collected {len(random_trajectories)} random trajectories")
    
    traj_sample = random_trajectories[0]
    state_dim = traj_sample['states'].shape[1]
    action_dim = traj_sample['actions'].shape[1]
    
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    
    all_trajectories = random_trajectories.copy()
    
    num_iterations = 20
    # dynamics_model = None
    total_steps = 0
    steps_history = []
    rewards_history = []
    
    for iteration in range(num_iterations):
        print(f"\n===== ITERATION {iteration+1}/{num_iterations} (Total Steps: {total_steps}) =====")
        
        data_loader = prepare_training_data(all_trajectories, batch_size=64)
    
        
        dynamics_model = train_model(data_loader, state_dim, action_dim, epochs=40, lr=1e-3)
        
        print(f"Iteration {iteration+1}: Model training complete")
        horizon = 10
        print(horizon)
        print(f"Iteration {iteration+1}: Collecting optimized trajectories")
        optimized_trajectories = collect_optimized_trajectories(
            dynamics_model, 
            num_episodes=1, 
            horizon=horizon, #20
            iterations=50,
            lr=0.001,
            method='gradient'
        )
        print("cem")
        collect_optimized_trajectories(
            dynamics_model, 
            num_episodes=1, 
            horizon=1   , #20
            iterations=3000,
            lr=0.001,
            method='cem'
        )
        
        # Update total steps and record history
        iteration_reward = 0
        for traj in optimized_trajectories:
            total_steps += len(traj['states'])
            iteration_reward += np.sum(traj['rewards'])
        
        avg_reward = iteration_reward / len(optimized_trajectories) if optimized_trajectories else 0
        steps_history.append(total_steps)
        rewards_history.append(avg_reward)
        
        print(f"Iteration {iteration+1}: Collected {len(optimized_trajectories)} optimized trajectories")
        
        all_trajectories.extend(optimized_trajectories)
        print(f"Iteration {iteration+1}: Total trajectories: {len(all_trajectories)}")
    plt.figure(figsize=(10, 6))
    plt.plot(steps_history, rewards_history, 'b-o', linewidth=2, markersize=4)
    plt.xlabel('Total Steps')
    plt.ylabel('Average Reward')
    plt.title('Learning Curve: Steps vs Average Reward')
    plt.grid(True)
    plt.savefig('learning_curve.png')
    plt.show()
    print([steps_history, rewards_history])

if __name__ == "__main__":
    main()

