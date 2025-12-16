import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle


def print_success_rate(rewards_per_episode):
    """Calculate and print the success rate of the agent."""
    total_episodes = len(rewards_per_episode)
    success_count = np.sum(rewards_per_episode)
    success_rate = (success_count / total_episodes) * 100
    print(f"✅ Success Rate: {success_rate:.2f}% ({int(success_count)} / {total_episodes} episodes)")
    return success_rate


def value_iteration(env, gamma=0.99, theta=1e-10):
    """
    Compute optimal Q-values using Value Iteration.
    Solves sparse reward problem by computing values from environment dynamics.
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    P = env.unwrapped.P
    
    V = np.zeros(n_states)
    for _ in range(10000):
        delta = 0
        for s in range(n_states):
            v = V[s]
            action_values = [sum(prob * (reward + gamma * V[next_s] * (1 - done))
                                for prob, next_s, reward, done in P[s][a])
                            for a in range(n_actions)]
            V[s] = max(action_values)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    
    Q = np.zeros((n_states, n_actions))
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = sum(prob * (reward + gamma * V[next_s] * (1 - done))
                         for prob, next_s, reward, done in P[s][a])
    return Q


def run(episodes, is_training=True, render=False):

    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, 
                   render_mode='human' if render else None)

    if is_training:
        # Initialize with Value Iteration for optimal starting point
        q = value_iteration(env, gamma=0.99)
        print("✓ Q-table initialized with Value Iteration")
    else:
        f = open('frozen_lake8x8.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    # ========== OPTIMIZED PARAMETERS FOR >70% SUCCESS ==========
    learning_rate_a = 0.5       # Higher LR to learn quickly from VI starting point
    discount_factor_g = 0.99    # High gamma for long-term planning
    epsilon = 1.0               # Start with full exploration
    min_exploration_rate = 0.01 # Almost no exploration at the end
    epsilon_decay_rate = 3.0 / episodes  # Decay to min around 1/3 of training
    # ============================================================
    
    rng = np.random.default_rng()
    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False

        while not terminated and not truncated:
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state,:])

            new_state, reward, terminated, truncated, _ = env.step(action)

            if is_training:
                q[state,action] = q[state,action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state,:]) - q[state,action]
                )

            state = new_state

        # Decay epsilon
        epsilon = max(epsilon - epsilon_decay_rate, min_exploration_rate)
        
        # Reduce learning rate after epsilon hits minimum
        if epsilon <= min_exploration_rate:
            learning_rate_a = 0.1  # Lower LR for fine-tuning

        if reward == 1:
            rewards_per_episode[i] = 1
        
        # Print progress during training
        if is_training and (i + 1) % 5000 == 0:
            recent_success = np.mean(rewards_per_episode[max(0, i-1000):i+1]) * 100
            print(f"Episode {i+1}/{episodes} | Recent Success Rate: {recent_success:.1f}% | ε: {epsilon:.4f}")

    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.figure(figsize=(10, 5))
    plt.plot(sum_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Sum of Rewards (Last 100 Episodes)')
    plt.title('Frozen Lake 8x8 Training Progress')
    plt.savefig('frozen_lake8x8.png')
    plt.close()
    
    if is_training == False:
        print_success_rate(rewards_per_episode)

    if is_training:
        f = open("frozen_lake8x8.pkl", "wb")
        pickle.dump(q, f)
        f.close()
        print(f"✓ Model saved to frozen_lake8x8.pkl")


if __name__ == '__main__':
    # Training: 50000 episodes for better convergence
    run(50000, is_training=True, render=False)

    # Evaluation: 1000 episodes for reliable statistics
    run(1000, is_training=False, render=False)