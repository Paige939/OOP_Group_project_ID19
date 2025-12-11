from manage import PendulumEnvWrapper
from manage import Experiment
from Agents.random_agent import RandomAgent

if __name__=="__main__":
    env_wrapper=PendulumEnvWrapper(render_mode="human")
    agent=RandomAgent(action=env_wrapper.action, max_action=env_wrapper.max_action)
    exper=Experiment(env=env_wrapper, agent=agent, episode_len=10)
    for ep in range(5):
        total_reward=exper.run_episode(render=True)
        print(f"Episode {ep+1} total reward: {total_reward:.2f}")

    exper.close()