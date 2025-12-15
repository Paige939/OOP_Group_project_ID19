import gymnasium as gym
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional
from base_agent import Agent



#------Environment wrapper: Encapsulation-----
class PendulumEnvWrapper:
    """
    Encapsulate the Gymnasium Pendulum environment 
    """
    def __init__(self, render_mode: Optional[str] = None):
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
        Reset the environment and return the initial observation
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
        #the 2 gym finish conditions
        done=bool(terminated or truncated)
        self.current_observation=next_obs
        return next_obs, float(reward), done, info

    def render(self):
        """
        Render screen(if use human render mode, gymnasium will render automatically)
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


#------Environment Management + Agent Management-----
class Experiment:
    """
    Responsible for:
    * Manage environment (PendulumEnvWrapper)
    * Manage agent
    * Training / Testing
    """
    def __init__(self, env: PendulumEnvWrapper, agent: Agent, episode_len: int = 200):
        self.env = env
        self.agent = agent
        self.episode_len = episode_len

    def run_episode(self, render: bool = False) -> float:
        """
        Run an episode and return total reward (用於測試/推論)
        * render: if True and is supported by env-> show on the screen
        """
        # Call the pre-episode for some agent (ex: CEM) - 如果有的話
        if hasattr(self.agent, 'pre_episode'):
            self.agent.pre_episode(self.env, self.episode_len)
        
        # Initialization
        self.agent.reset()
        obs = self.env.reset()
        total_reward = 0.0
        
        # Execute step by step until reach the max steps or done=True
        for i in range(self.episode_len):
            # 判斷 agent.act 是否支援 add_noise 參數
            if hasattr(self.agent, 'training_mode'):
                action = self.agent.act(obs, add_noise=False)  # DDPG/TD3: 測試時不加噪聲
            else:
                action = self.agent.act(obs)  # CEM/Random: 不支援 add_noise
            
            next_obs, reward, done, info = self.env.step(action)
            total_reward += reward
            obs = next_obs
            
            if render:
                self.env.render()
            if done:
                break
        
        return total_reward

    def train_episode(self) -> Tuple[float, float, float]:
        """
        執行一個訓練 episode（用於訓練 DDPG/TD3 Agent）
        
        Returns:
            total_reward: 總獎勵
            avg_critic_loss: 平均 Critic 損失
            avg_actor_loss: 平均 Actor 損失
        """
        # 檢查 Agent 是否有訓練方法
        if not hasattr(self.agent, 'train_step') or not hasattr(self.agent, 'store_transition'):
            raise AttributeError(
                f"{type(self.agent).__name__} does not support training. "
                "Please use DDPG_Agent or TD3_Agent."
            )
        
        # 設定為訓練模式
        if hasattr(self.agent, 'set_training_mode'):
            self.agent.set_training_mode(True)
        
        # Initialization
        self.agent.reset()
        obs = self.env.reset()
        total_reward = 0.0
        critic_losses = []
        actor_losses = []
        
        # Execute step by step
        for step in range(self.episode_len):
            # 選擇動作（訓練時加入探索噪聲）
            action = self.agent.act(obs, add_noise=True)
            
            # 執行動作
            next_obs, reward, done, info = self.env.step(action)
            total_reward += reward
            
            # 儲存經驗
            self.agent.store_transition(obs, action, reward, next_obs, done)
            
            # 訓練更新
            critic_loss, actor_loss = self.agent.train_step()
            if critic_loss > 0:  # 只記錄有效的損失
                critic_losses.append(critic_loss)
                if actor_loss > 0:
                    actor_losses.append(actor_loss)
            
            obs = next_obs
            
            if done:
                break
        
        # 計算平均損失
        avg_critic_loss = np.mean(critic_losses) if critic_losses else 0.0
        avg_actor_loss = np.mean(actor_losses) if actor_losses else 0.0
        
        return total_reward, avg_critic_loss, avg_actor_loss

    def close(self):
        """關閉環境"""
        self.env.close()
