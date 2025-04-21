import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import random
import copy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import pickle
import os

def make_env():
    return Monitor(gym.make("Hopper-v5"))

def train_ppo_agent(total_timesteps=500_000):
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    policy_kwargs = {
        "net_arch": [
            {"pi": [128, 128], "vf": [128, 128]}
        ],
        "activation_fn": nn.ReLU
    }
    
    model = PPO(
        policy="MlpPolicy",
        policy_kwargs=policy_kwargs,
        env=env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1
    )
    
    model.learn(total_timesteps=total_timesteps)
    env.training = False
    env.norm_reward = False
    
    print("Evaluating trained PPO agent:")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    model.save("ppo_hopper")
    env.save("vec_normalize_hopper.pkl")
    
    return model, env

def collect_ppo_trajectories(model, env, num_episodes=100, max_steps=1000):
    trajectories = []
    #env = gym.make('Hopper-v5')
    avg_reward = []
    for episode in range(num_episodes):
        obs = env.reset()[0]
        states, actions, next_states, rewards = [], [], [], []
        total_reward = 0
        for t in range(max_steps):
            action, _ = model.predict(obs, deterministic=False)
            # next_obs, reward, terminated, truncated, info = env.step(action)
            next_obs, reward, terminated, truncated = env.step([action])
            total_reward += reward
            states.append(obs.reshape(-1))
            actions.append(action.reshape(-1))
            next_states.append(next_obs.reshape(-1))
            rewards.append(reward)
            
            obs = next_obs
            
            if terminated or truncated:
                break
        
        avg_reward.append(total_reward)
        trajectories.append({
            'states': np.array(states),
            'actions': np.array(actions),
            'next_states': np.array(next_states),
            'rewards': np.array(rewards)
        })
            
        if (episode + 1) % 10 == 0:
            print(f"Collected {episode + 1}/{num_episodes} trajectories")
    print("Average reward per episode:", np.mean(avg_reward))
    return trajectories

def save_trajectories(trajectories, filename="ppo_trajectories.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump(trajectories, f)
    
    print(f"Saved {len(trajectories)} trajectories to {filename}")
    
    # Save some stats about the trajectories
    total_steps = sum(len(traj['states']) for traj in trajectories)
    avg_length = total_steps / len(trajectories)
    avg_reward = np.mean([np.sum(traj['rewards']) for traj in trajectories])
    
    print(f"Total steps: {total_steps}")
    print(f"Average trajectory length: {avg_length:.2f} steps")
    print(f"Average trajectory return: {avg_reward:.2f}")

def load_trajectories(filename="ppo_trajectories.pkl"):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Trajectory file {filename} not found")
    
    with open(filename, 'rb') as f:
        trajectories = pickle.load(f)
    
    print(f"Loaded {len(trajectories)} trajectories from {filename}")
    
    # Print some stats about the loaded trajectories
    total_steps = sum(len(traj['states']) for traj in trajectories)
    avg_length = total_steps / len(trajectories)
    avg_reward = np.mean([np.sum(traj['rewards']) for traj in trajectories])
    
    print(f"Total steps: {total_steps}")
    print(f"Average trajectory length: {avg_length:.2f} steps")
    print(f"Average trajectory return: {avg_reward:.2f}")
    
    return trajectories

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
        
    def forward(self, state, action, n_euler_steps=1):
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

# def optimize_actions(dynamics_model, init_state, horizon=30, iterations=10, lr=1e-1, gamma=1.0):
#     init_state_tensor = torch.tensor(init_state, dtype=torch.float32).unsqueeze(0)
    
#     actions = nn.Parameter(torch.randn(horizon, 3) * 0.1)
#     optimizer = torch.optim.Adam([actions], lr=lr)
    
#     for i in range(iterations):
#         optimizer.zero_grad()
        
#         current_state = init_state_tensor.clone()
#         states = [current_state]
#         for t in range(horizon):
#             action = torch.tanh(actions[t]).unsqueeze(0)
#             next_state = dynamics_model(current_state, action)
#             states.append(next_state)
#             current_state = next_state
            
#         trajectory = torch.cat(states, dim=0)
#         x_velocities = trajectory[:, 5]
#         angles = trajectory[:, 1]
        
#         velocity_losses = -x_velocities
#         angle_losses = 500 * angles**2
#         discount_factors = torch.tensor([gamma**t for t in range(len(velocity_losses))], 
#                                         dtype=torch.float32)
#         discounted_velocity_loss = torch.mean(velocity_losses * discount_factors)
#         discounted_angle_loss = torch.mean(angle_losses * discount_factors)
#         total_loss = discounted_velocity_loss/10 + discounted_angle_loss
        
#         # z = trajectory[:, 0]
#         # action_loss = 0.001 * torch.mean(actions**2)
#         # angle_loss = torch.sigmoid(10 * (torch.abs(angles) - 0.2))
#         # z_loss = torch.sigmoid(10 * (z - 0.7))
#         # health_loss = -torch.mean(angle_loss * z_loss)
#         # total_loss = health_loss + -torch.mean(x_velocities) + action_loss

#         total_loss.backward()
#         optimizer.step()
    
#     with torch.no_grad():
#         final_actions = torch.tanh(actions)
    
#     return final_actions

def optimize_actions2(dynamics_model, init_state, horizon=30, iterations=10, lr=1e-1, gamma=1.0):
    init_state_tensor = torch.tensor(init_state, dtype=torch.float32).unsqueeze(0)
    actions = nn.Parameter(torch.randn(horizon, 3) * 0.1)
    optimizer = torch.optim.Adam([actions], lr=lr)
    
    for i in range(iterations):
        optimizer.zero_grad()
        
        current_state = init_state_tensor.clone()
        states = [current_state]
        for t in range(horizon):
            action = torch.tanh(actions[t]).unsqueeze(0)
            next_state = dynamics_model(current_state, action)
            states.append(next_state)
            current_state = next_state
            
        # trajectory = torch.cat(states, dim=0)
        # z = trajectory[:, 0]
        # angles = trajectory[:, 1]
        # x_velocities = trajectory[:, 5]
        
        # z_constraint = (z > 0.7).float()
        # angle_constraint = (torch.abs(angles) > 0.2).float()
        # constraints_met = z_constraint * angle_constraint
        
        # rewards = constraints_met * (1.0 + x_velocities)
        # discount_factors = torch.tensor([gamma**t for t in range(len(rewards))], dtype=torch.float32)
        # total_reward = torch.sum(rewards * discount_factors)
        # total_loss = -total_reward
        trajectory = torch.cat(states, dim=0)
        x_velocities = trajectory[:, 5]
        angles = trajectory[:, 1]
        velocity_losses = -x_velocities
        angle_losses = 200 * angles**2
        discount_factors = torch.tensor([gamma**t for t in range(len(velocity_losses))], 
                                        dtype=torch.float32)
        discounted_velocity_loss = torch.mean(velocity_losses * discount_factors)
        discounted_angle_loss = torch.mean(angle_losses * discount_factors)
        total_loss = discounted_velocity_loss/10 + discounted_angle_loss
        #total_loss = discounted_angle_loss
        total_loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        final_actions = torch.tanh(actions)
        
        current_state = init_state_tensor.clone()
        states = [current_state]
        total_reward = 0
        for t in range(horizon):
            action = final_actions[t].unsqueeze(0)
            next_state = dynamics_model(current_state, action)
            states.append(next_state)
            current_state = next_state
            
            z_ok = current_state[0, 0] > 0.7
            angle_ok = torch.abs(current_state[0, 1]) < 0.2
            # print(action, " ", current_state[0, 1])
            # print(z_ok, " ", angle_ok)
            if z_ok and angle_ok:
                total_reward += 1.0 #+ current_state[0, 5].item() - 0.001*torch.sum(action**2).item()
            else:
                break
    
    return final_actions, total_reward

def optimize_actions(dynamics_model, init_state, n_trials=2, **kwargs):
    best_reward = -float('inf')
    best_actions = None
    #print("Optimizing actions...")
    for _ in range(n_trials):
        actions, reward = optimize_actions2(dynamics_model, init_state, **kwargs)
        #print(reward)
        if reward > best_reward:
            best_reward = reward
            best_actions = actions
    
    return best_actions #, best_reward

def get_true_reward(state, action, z0):
    # env = gym.make('Hopper-v5')
    # env.reset()
    # qpos_new = np.zeros(6)
    # qvel_new = np.zeros(6)
    # qpos_new[0] = z0
    # qpos_new[1:] = state[0][:5]
    # qvel_new = state[0][5:]
    # env.unwrapped.set_state(qpos_new, qvel_new)
    
    # next_state, reward, terminated, truncated, info = env.step(action)
    # print(info)
    # print(sum(action**2) * 0.001)
    # print(state[0][5])
    healthy = int(state[0][0] > 0.7 and abs(state[0][0]) > 0.2)
    reward = sum(action**2) * 0.001 + state[0][5] + healthy
    #print(healthy)
    # print(total_reward)
    # print(reward)
    # env.close()
    terminated = healthy == 0
    return reward, terminated

def random_shooting(dynamics_model, init_state, num_samples=500, horizon=40, gamma=1.0):
    init_state_tensor = torch.tensor(init_state, dtype=torch.float32).unsqueeze(0)
    best_actions = None
    best_score = float('-inf')
    
    for _ in range(num_samples):
        actions = 2*torch.rand(horizon, 3, requires_grad=False) - 1
        
        current_state = init_state_tensor.clone()
        states = [current_state]
        total_reward = 0
        for t in range(horizon):
            a = copy.deepcopy(actions[t]).detach().numpy()
            c = current_state.clone().detach().numpy()
            # print(get_true_reward(c, a, 0.8))
            # print(get_true_reward(c, a, 0.5))
            # print("sadfas")
            true_reward, terminated = get_true_reward(c, a, 0.8)
            total_reward += true_reward
            if terminated:
                # print("terminated")
                break
            action = actions[t].unsqueeze(0)
            next_state = dynamics_model(current_state, action)
            states.append(next_state)
            current_state = next_state
        
        trajectory = torch.cat(states, dim=0)
        x_velocities = trajectory[:, 5]
        angles = trajectory[:, 1]
        
        
        # discount_factors = torch.tensor([gamma**t for t in range(len(x_velocities))], dtype=torch.float32)
        # discounted_velocity_reward = torch.mean(x_velocities * discount_factors)
        # discounted_angle_penalty = torch.mean((angles**2) * discount_factors)
        
        # score = -discounted_velocity_reward/10 - 200 * discounted_angle_penalty
        score = total_reward
        if score > best_score:
            best_score = score
            best_actions = actions.clone()
    
    return best_actions

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
                horizon=20, 
                iterations=50,
                lr=0.01
            )
            print(step)
            # action_seq = random_shooting(
            #     dynamics_model, 
            #     state, 
            #     num_samples=500, 
            #     horizon=40
            # )
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

def load_dynamics_model(model_path, state_dim, action_dim):
    model = StatePredictor(state_dim, action_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode
    
    print(f"Loaded dynamics model from {model_path}")
    return model

def main():
    print("Step 1: Training PPO agent")
    ppo_model, ppo_env = train_ppo_agent(total_timesteps=1_000_000)
    
    print("Step 2: Collecting trajectories using the trained PPO agent")
    ppo_trajectories = collect_ppo_trajectories(ppo_model, ppo_env, num_episodes=1000, max_steps=1000)
    print(f"Collected {len(ppo_trajectories)} trajectories from PPO agent")
    save_trajectories(ppo_trajectories, filename="ppo_trajectories.pkl")
    ppo_trajectories = load_trajectories(filename="ppo_trajectories.pkl")
    traj_sample = ppo_trajectories[0]
    state_dim = traj_sample['states'].shape[1]
    action_dim = traj_sample['actions'].shape[1]
    
    dynamics_model = load_dynamics_model("dynamics_model.pt", state_dim, action_dim)
    
    # print("Step 3: Training dynamics model on PPO trajectories")
    # data_loader = prepare_training_data(ppo_trajectories, batch_size=64)
    # dynamics_model = train_model(data_loader, state_dim, action_dim, epochs=50, lr=1e-3)
    
    print("Step 4: Evaluating dynamics model with gradient-based control")
    performance = eval_model(dynamics_model, n_evals=5)
    print(f"Final dynamics model performance: {performance:.2f}")
    
    torch.save(dynamics_model.state_dict(), "dynamics_model.pt")
    print("Dynamics model saved to 'dynamics_model.pt'")

if __name__ == "__main__":
    main()