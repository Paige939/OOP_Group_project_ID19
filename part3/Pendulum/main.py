from manage import PendulumEnvWrapper, Experiment
from Agents import RandomAgent, CEM_Agent, EnergyControlAgent, LQRAgent, ELAgent
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    episode_len = 200
    n_episodes = 5

    render_or_not = False

    # --- Random Agent ---
    #create environment for Random
    rand_env_wrapper = PendulumEnvWrapper(render_mode=render_or_not) 
    random_agent = RandomAgent(action=rand_env_wrapper.action, max_action=rand_env_wrapper.max_action)
    exper_rand = Experiment(rand_env_wrapper, random_agent, episode_len)
    rand_rewards = []
    print("\n=== Random Agent ===")
    for ep in range(n_episodes):
        reward = exper_rand.run_episode(render=False)
        rand_rewards.append(reward)
        print(f"Episode {ep+1} total reward: {reward:.2f}")
        

    # --- CEM Agent ---
    # create environmemt for CEM
    cem_env_wrapper = PendulumEnvWrapper(render_mode=render_or_not) 
    cem_agent = CEM_Agent(action=cem_env_wrapper.action, max_action=cem_env_wrapper.max_action,
                          num_samples=300, elite_frac=0.2, save_path="CEM_weights.npy")
    exper_cem = Experiment(cem_env_wrapper, cem_agent, episode_len)
    cem_rewards = []
    print("=== CEM Agent ===")
    for ep in range(n_episodes):
        reward = exper_cem.run_episode(render=False) #The render mode can be changed from render_or_not, "rgb_array", None
        cem_rewards.append(reward)
        print(f"Episode {ep+1} total reward: {reward:.2f}")

    
    # --- Energy Agent ---
    #create environment for EnergyControl
    energy_env = PendulumEnvWrapper(render_mode=render_or_not)
    energy_agent = EnergyControlAgent(action_dim=energy_env.action, max_action=energy_env.max_action)
    exper_energy = Experiment(energy_env, energy_agent, episode_len)
    energy_rewards = []
    print("\n=== Energy Control Agent ===")
    for ep in range(n_episodes):
        reward = exper_energy.run_episode(render=False)
        energy_rewards.append(reward)
        print(f"Episode {ep+1} total reward: {reward:.2f}")
    
    energy_env.close()

    # --- LQR Agent ---
    #create environment for LQR
    lqr_env = PendulumEnvWrapper(render_mode=render_or_not)
    lqr_agent = LQRAgent(action_dim=lqr_env.action, max_action=lqr_env.max_action)
    exper_lqr = Experiment(lqr_env, lqr_agent, episode_len)
    lqr_rewards = []
    print("\n=== LQR Agent (Expected to fail if starts at bottom) ===")
    for ep in range(n_episodes):
        reward = exper_lqr.run_episode(render=False)
        lqr_rewards.append(reward)
        print(f"Episode {ep+1} total reward: {reward:.2f}")
    lqr_env.close()

    # --- Energy + LQR Agent ---
    #create environment for EL
    el_env = PendulumEnvWrapper(render_mode=render_or_not)
    el_agent = ELAgent(action_dim=el_env.action, max_action=el_env.max_action)
    exper_el = Experiment(el_env, el_agent, episode_len)
    el_rewards = []
    print("\n=== EL Agent (Hybrid) ===")
    for ep in range(n_episodes):
        reward = exper_el.run_episode(render=False)
        el_rewards.append(reward)
        print(f"Episode {ep+1} total reward: {reward:.2f}")
    el_env.close()



    cem_env_wrapper.close()
    rand_env_wrapper.close() # close all environments

    # --- Plot comparison (Bar Chart with Mean ± Std) ---
    import numpy as np
    
    agents = ['Random', 'CEM', 'Energy', 'LQR', 'EL (Hybrid)']
    all_rewards = [rand_rewards, cem_rewards, energy_rewards, lqr_rewards, el_rewards]
    means = [np.mean(r) for r in all_rewards]
    stds = [np.std(r) for r in all_rewards]
    
    plt.figure(figsize=(10, 6))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    bars = plt.bar(agents, means, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 20, 
                 f'{mean:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xlabel("Agent", fontsize=12)
    plt.ylabel("Total Reward", fontsize=12)
    plt.title("Agent Comparison on Pendulum-v1 (Mean ± Std)", fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("Comparison.png", dpi=150)
    plt.show()
    
