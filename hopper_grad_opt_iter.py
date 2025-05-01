import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import random
import copy
import pickle


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
        
        trajectories.append({
            'states': np.array(states),
            'actions': np.array(actions),
            'next_states': np.array(next_states),
            'rewards': np.array(rewards)
        })
    
    env.close()
    return trajectories

class StatePredictor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128): # 128
        super(StatePredictor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.SiLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
    def forward(self, state, action, n_euler_steps=4): # 4
        state_action = torch.cat([state, action], dim=-1)
        next_state = state
        step_size = 1.0 / n_euler_steps
        for _ in range(n_euler_steps):
            next_state = step_size * self.net(state_action) + next_state
            state_action = torch.cat([next_state, action], dim=-1)
        return next_state

# class StatePredictor(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_dim=128, ensemble_size=5):
#         super(StatePredictor, self).__init__()
#         self.models = nn.ModuleList([
#             StatePredictorMini(state_dim, action_dim, hidden_dim) 
#             for _ in range(ensemble_size)
#         ])
        
#     def forward(self, state, action, n_euler_steps=4):
#         predictions = [model(state, action, n_euler_steps) for model in self.models]
#         return torch.mean(torch.stack(predictions), dim=0)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# # Jump ODE
# class StatePredictor(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_dim=128):
#         super(StatePredictor, self).__init__()
#         self.state_dim = state_dim
#         self.action_dim = action_dim
        
#         # Continuous dynamics network (similar to original)
#         self.dynamics_net = nn.Sequential(
#             nn.Linear(state_dim + action_dim, hidden_dim),
#             nn.Tanh(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.Tanh(),
#             nn.Linear(hidden_dim, state_dim)
#         )
        
#         # Jump probability network
#         self.jump_prob_net = nn.Sequential(
#             nn.Linear(state_dim + action_dim, hidden_dim),
#             nn.Tanh(),
#             nn.Linear(hidden_dim, 1),
#             nn.Sigmoid()  # Output probability of jumping
#         )
        
#         # Jump target network (predicts where to jump to)
#         self.jump_target_net = nn.Sequential(
#             nn.Linear(state_dim + action_dim, hidden_dim),
#             nn.Tanh(),
#             nn.Linear(hidden_dim, state_dim)
#         )
        
#     def continuous_dynamics(self, state, action):
#         """Compute the continuous state derivative"""
#         state_action = torch.cat([state, action], dim=-1)
#         return self.dynamics_net(state_action)
    
#     def jump_probability(self, state, action):
#         """Compute the probability of jumping at this state"""
#         state_action = torch.cat([state, action], dim=-1)
#         return self.jump_prob_net(state_action)
    
#     def jump_target(self, state, action):
#         """Compute the target state if a jump occurs"""
#         state_action = torch.cat([state, action], dim=-1)
#         return self.jump_target_net(state_action)
    
#     def forward(self, state, action, n_euler_steps=4, jump_threshold=0.5):
#         next_state = state
#         step_size = 1.0 / n_euler_steps
        
#         for step in range(n_euler_steps):
#             jump_prob = self.jump_probability(next_state, action)
#             should_jump = jump_prob > jump_threshold
#             if should_jump.any():
#                 jump_target = self.jump_target(next_state, action)
#                 mask = should_jump.float() #.unsqueeze(-1)
#                 dynamics = self.continuous_dynamics(next_state, action)
#                 next_state = mask * jump_target + (1 - mask) * (next_state + step_size * dynamics)
#             else:
#                 # If no jumps, apply standard Euler step
#                 dynamics = self.continuous_dynamics(next_state, action)
#                 next_state = next_state + step_size * dynamics

#         return next_state

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
    
    dynamics_optimizer = torch.optim.Adam(dynamics_model.parameters(), lr=lr, weight_decay=1e-4)
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

def optimize_actions(dynamics_model, init_state, horizon=30, iterations=10, lr=1e-1, gamma=1.0):
    init_state_tensor = torch.tensor(init_state, dtype=torch.float32).unsqueeze(0)
    
    actions = nn.Parameter(torch.randn(horizon, 3) * 0.1)
    
    optimizer = torch.optim.Adam([actions], lr=lr)
    
    for i in range(iterations):
        optimizer.zero_grad()
        
        current_state = init_state_tensor.clone()
        states = [current_state]
        terminated = False
        for t in range(horizon):
            action = torch.tanh(actions[t]).unsqueeze(0)
            next_state = dynamics_model(current_state, action)
            states.append(next_state)
            
            current_state = next_state
        trajectory = torch.cat(states, dim=0)
        x_velocities = trajectory[:, 5]
        angles = trajectory[:, 1]
        z = trajectory[:, 0]
        total_loss = -torch.mean(torch.tanh((z - 0.7) * 10)) + -torch.mean(torch.tanh((0.2 - torch.abs(angles)) * 10)) -x_velocities.mean()



        total_loss.backward()
        optimizer.step()
        
        # if (i+1) % 1 == 0:
        #     with torch.no_grad():
        #         print(f"Iteration {i+1}/{iterations}, Loss: {total_loss:.4f}, " 
        #               f"Vel: {velocity_loss:.4f}, Action: {action_loss}, health: {health_loss}")
    
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
        total_reward = 0
        while not done and steps < 1000:
            action_seq = optimize_actions(
                dynamics_model, 
                state, 
                horizon=horizon, 
                iterations=iterations,
                lr=lr
            )
            
            action = action_seq[0].detach().numpy()
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward+=reward
            
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

def load_trajectories(load_path="trained_trajectories.pkl"):
    with open(load_path, 'rb') as f:
        trajectories = pickle.load(f)
    return trajectories

def main():
    random_trajectories = collect_data(num_episodes=100, max_steps=1000)
    print(random_trajectories[0]["states"].shape)
    random_trajectories = load_trajectories("../../../transformers/trained_trajectories.pkl")
    print(random_trajectories[0]["states"].shape)
    print(random_trajectories[0]["states"].shape)
    print(random_trajectories[0]["actions"].shape)
    # print(random_trajectories)
    print(f"Collected {len(random_trajectories)} random trajectories")
    
    traj_sample = random_trajectories[0]
    state_dim = traj_sample['states'].shape[1]
    action_dim = traj_sample['actions'].shape[1]
    
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    
    all_trajectories = random_trajectories.copy()
    
    best_dynamics_model = None
    best_performance = -float('inf')
    
    num_iterations = 5
    dynamics_model = None
    for iteration in range(num_iterations):
        print(f"\n===== ITERATION {iteration+1}/{num_iterations} =====")
        
        data_loader = prepare_training_data(all_trajectories, batch_size=64)
        dynamics_model = train_model(data_loader, state_dim, action_dim, epochs=5, lr=1e-3, model=None)
        
        print(f"Iteration {iteration+1}: Model training complete")
        
        print(f"Iteration {iteration+1}: Collecting optimized trajectories")
        optimized_trajectories = collect_optimized_trajectories(
            dynamics_model, 
            num_episodes=1, 
            horizon=30, # 30
            iterations=50, # 50
            lr=0.001 #0.01
        )
        
        print(f"Iteration {iteration+1}: Collected {len(optimized_trajectories)} optimized trajectories")
        
        # print(f"Iteration {iteration+1}: Evaluating current model")
        # current_performance = eval_model(dynamics_model, n_evals=3)
        
        # if current_performance > best_performance:
        #     best_performance = current_performance
        #     best_dynamics_model = copy.deepcopy(dynamics_model)
        #     print(f"Iteration {iteration+1}: New best model! Performance: {best_performance:.2f}")
        
        all_trajectories.extend(optimized_trajectories)
        print(f"Iteration {iteration+1}: Total trajectories: {len(all_trajectories)}")
    
    print("\n===== FINAL EVALUATION =====")
    print("Evaluating best model:")
    final_performance = eval_model(best_dynamics_model, n_evals=5)
    print(f"Best model performance: {final_performance:.2f}")

if __name__ == "__main__":
    main()