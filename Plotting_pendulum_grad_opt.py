def eval_gradient_actions(model, horizon=10, iterations=100, lr=0.1, n_evals=100):
    print(f"Evaluating gradient-based action optimization: horizon={horizon}, iterations={iterations}")
    env = gym.make('InvertedPendulum-v4')
    avg_reward = 0
    all_step_rewards = []

    for k in range(n_evals):
        state, _ = env.reset()
        total_reward = 0
        step_rewards = []

        for step in range(1000):
            action_seq = optimize_actions(
                model, 
                state, 
                horizon=horizon, 
                iterations=iterations,
                lr=lr
            )
            
            action = action_seq[0].detach().numpy()
            next_state, reward, terminated, truncated, _ = env.step([action.item()])
            state = next_state
            total_reward += reward
            step_rewards.append(total_reward)  # Store cumulative reward

            if terminated or truncated:
                break
        
        all_step_rewards.append(step_rewards)
        print(f"Episode {k+1}: Reward = {total_reward:.2f}")
        avg_reward += total_reward

    env.close()

    # Pad shorter episodes with their last reward value for uniform length
    max_len = max(len(r) for r in all_step_rewards)
    for r in all_step_rewards:
        while len(r) < max_len:
            r.append(r[-1])

    mean_rewards = np.mean(all_step_rewards, axis=0)

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(mean_rewards, label="Average Reward")
    plt.xlabel("Steps")
    plt.ylabel("Cumulative Reward")
    plt.title("Rewards vs Number of Steps")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Average reward over {n_evals} episodes: {avg_reward / n_evals:.2f}")
