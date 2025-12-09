import gymnasium as gym
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple
from base_agent import Agent



#------Environment wrapper: Encapsulation-----
class PendulumEnvWrapper:
    """
    Encapsulate the Gymnasium Pendulum environment 
    """
    def __init__(self, render_mode: None):
        #you can put in the render mode you want(human/rgb_array)
        self.env=gym.make("Pendulum-v1", render_mode=render_mode)
        #get the observation space (cos_theta, sin_theta, theta_dot)
        self.state=self.env.observation_space.shape[0] 
        #get the action space (torque ∈ [-2, +2] )
        self.action=self.env.action_space.shape[0]
        #get the max value of action
        self.max_action=float(self.env.action_space.high[0])
        #to store the current observation value
        self.current_observation=None

    def reset(self)->np.ndarray:
        """
        Reset the environmrnt and return the initial observation
        """
        obs,_=self.env.reset()
        self.current_observation=obs
        return obs
    
    def step(self, action: np.ndarray)->Tuple[np.ndarray, float, bool, dict]:
        #ensure action to be numpy array and not than out of the range
        action=np.array(action, dtype=np.float32)
        #-max_action~max_action
        action=np.clip(action, -self.max_action, self.max_action)
        #gym step execute a timestep to get the next observation, reward, terminated, truncated, info
        next_obs, reward, terminated, truncated, info=self.env.step(action)
        #the 2 gym fininsh conditions
        done=bool(terminated or truncated)
        self.current_observation=next_obs
        return next_obs, float(reward), done, info

    def render(self):
        """
        Render screen(if use human render mode, gymnasium will render automaically)
        """
        return None
    
    def close(self):
        """
        Close the environment
        """
        self.env.close()

    def get_state(self)->np.ndarray:
        """
        Return the current observation values
        """
        return self.current_observation


#------Environment Management+ Agent Management-----
class Experiment:
    """
    Response for:
    * Manage environment (PendulumEnvWrapper)
    * Manage agent
    * Tarining 
    """
    def __init__(self, env: PendulumEnvWrapper, agent: Agent, episode_len: int):
        self.env=env
        self.agent=agent
        self.episode_len=episode_len

    def run_episode(self, render: bool=False)->float:
        """
        Run an episode and return total reward
        * render: if True and is supported by env-> show on the screen
        """
        #Initialization
        self.agent.reset()
        obs=self.env.reset()
        total_reward=0.0
        #Execute step by step until reach the max steps or done=True
        for i in range(self.episode_len):
            action=self.agent.act(obs)
            next_obs, reward, done, info=self.env.step(action)
            total_reward+=reward
            obs=next_obs
            if render:
                self.env.render()
            if done:
                break
        return total_reward

    def close(self):
        self.env.close()






