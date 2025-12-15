from manage import PendulumEnvWrapper, Experiment
from Agents import RandomAgent, CEM_Agent, EnergyControlAgent, LQRAgent, ELAgent
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    episode_len = 200
    n_episodes = 5


    # --- Random Agent ---
    #create environment for Random
    rand_env_wrapper = PendulumEnvWrapper(render_mode=None) 
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
    cem_env_wrapper = PendulumEnvWrapper(render_mode=None) 
    cem_agent = CEM_Agent(action=cem_env_wrapper.action, max_action=cem_env_wrapper.max_action,
                          num_samples=300, elite_frac=0.2, save_path="CEM_weights.npy")
    exper_cem = Experiment(cem_env_wrapper, cem_agent, episode_len)
    cem_rewards = []
    print("=== CEM Agent ===")
    for ep in range(n_episodes):
        reward = exper_cem.run_episode(render=False) #The render mode can be changed from "human", "rgb_array", None
        cem_rewards.append(reward)
        print(f"Episode {ep+1} total reward: {reward:.2f}")

    
    # --- Energy Agent ---
    #create environment for EnergyControl
    energy_env = PendulumEnvWrapper(render_mode=None)
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
    lqr_env = PendulumEnvWrapper(render_mode=None)
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
    el_env = PendulumEnvWrapper(render_mode=None)
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

    # --- Plot comparison ---
    #plt.figure(figsize=(8,5))
    plt.figure(figsize=(10,6))
    plt.plot(range(1,n_episodes+1), cem_rewards, 'o-', label='CEM Agent')
    plt.plot(range(1,n_episodes+1), rand_rewards, 's-', label='Random Agent')
    plt.plot(range(1,n_episodes+1), energy_rewards, 'x--', label='Energy Only')
    plt.plot(range(1,n_episodes+1), lqr_rewards, 'v:', label='LQR Only')
    plt.plot(range(1,n_episodes+1), el_rewards, '^-', label='EL Agent', linewidth=2.5)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(" Agent on Pendulum-v1")
    plt.legend()
    plt.grid(True)
    plt.savefig("Comparison.png")
    plt.show()
    
