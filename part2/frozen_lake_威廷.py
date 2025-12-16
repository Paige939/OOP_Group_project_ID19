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
    else:
        f = open('frozen_lake8x8.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.01  # Small LR to not destroy VI values
    discount_factor_g = 0.99
    epsilon = 1
    
    # ========== TUNED PARAMETERS ==========
    min_exploration_rate = 0.05
    epsilon_decay_rate = 0.0001
    # ======================================
    
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

        epsilon = max(epsilon - epsilon_decay_rate, min_exploration_rate)

        if reward == 1:
            rewards_per_episode[i] = 1

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


if __name__ == '__main__':
    # Training: 15000 episodes
    run(15000, is_training=True, render=False)

    # Evaluation: 10 episodes (same as original sample code)
    run(10, is_training=False, render=False)




# 1. 不可以增加 num_episodes 和 max_steps_per_episode，但程式碼的其他部分可以修改。 2. 你可以使用高階演算法，但必須理解其內部原理，並能回答你所使用方法的相關問題。 3. 地圖大小（map size）必須至少為 8x8。 4. 助教在評分時會從 training → testing 全部執行一次你的程式，以獲得你的最終表現結果。

# 可以使用Q learning 以外的演算法，目標就是要確認最後的成功率一定要穩定大於0.7(70%)

# Run the Frozen Lake:
# • Goal:
# • Revise the sample code to achieve a **consistent success rate > 0.70** on **without 
# changing**: `num_episodes`and `max_steps_per_episode`
# • You may **only tune**:  ‘min_exploration_rate (currently is 0)’ and ‘epsilon_decay_rate’
# • You should demonstrate the agent’s performance.
# • 1. **Train** with your tuned exploration settings (no change to episodes/steps).
# • 2. **Evaluate** success rate over a **fixed evaluation run** (e.g., 500–1000 test episodes at ε≈0).
# • 3. **Report**:
# •   - Final **success rate** (wins/episodes)
# •   - (Optional) a short **moving-average curve** over training episodes
# • **Success definition:** An episode counts as success if it reaches the goal (Gymnasium returns 
# reward `1.0` at termination).
# • Keys:
# • env = gym.make("FrozenLake-v1", render_mode="ansi")
# • print(env.render())