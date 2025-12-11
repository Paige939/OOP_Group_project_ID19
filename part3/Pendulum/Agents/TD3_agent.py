import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple
import copy

from base_agent import Agent
from replay_buffer import ReplayBuffer
from neural_networks import ActorNetwork, TwinCriticNetwork


class TD3_Agent(Agent):
    """
    TD3 (Twin Delayed Deep Deterministic Policy Gradient) Agent
    * 繼承自 Agent 基類 (Inheritance + Polymorphism)
    * DDPG 的改進版本，解決 Q 值過度估計問題
    
    三大核心改進：
    1. Clipped Double Q-Learning：使用兩個 Critic，取較小的 Q 值
    2. Delayed Policy Updates：延遲更新 Actor（每 d 次 Critic 更新才更新一次）
    3. Target Policy Smoothing：目標動作加入裁剪噪聲
    
    公式：
    - 目標 Q 值：y = r + γ * min(Q1'(s', ã'), Q2'(s', ã'))
    - 其中 ã' = clip(μ'(s') + clip(ε, -c, c), -max_action, max_action)
    - ε ~ N(0, σ)
    """
    
    def __init__(self, 
                 action: int,
                 max_action: float,
                 state_dim: int = 3,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 actor_lr: float = 1e-3,
                 critic_lr: float = 1e-3,
                 buffer_size: int = 50000,
                 batch_size: int = 256,
                 exploration_noise: float = 0.1,
                 policy_noise: float = 0.2,
                 noise_clip: float = 0.5,
                 policy_delay: int = 2,
                 device: str = None):
        """
        Args:
            action: 動作空間維度
            max_action: 動作最大值
            state_dim: 狀態空間維度（Pendulum 預設 3）
            gamma: 折扣因子
            tau: 軟更新係數
            actor_lr: Actor 學習率
            critic_lr: Critic 學習率
            buffer_size: 經驗回放緩衝區大小
            batch_size: 批次大小
            exploration_noise: 探索噪聲標準差
            policy_noise: 目標策略平滑噪聲標準差 (σ)
            noise_clip: 噪聲裁剪範圍 (c)
            policy_delay: 策略延遲更新頻率 (d)
            device: 計算裝置
        """
        super().__init__(action, max_action)
        
        # 設定裝置
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.state_dim = state_dim
        self.action_dim = action
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.exploration_noise = exploration_noise
        
        # TD3 特有參數
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.total_iterations = 0  # 追蹤總更新次數
        
        # 建立 Actor 網路和目標網路
        self.actor = ActorNetwork(state_dim, action, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # 建立 Twin Critic 網路和目標網路
        self.critic = TwinCriticNetwork(state_dim, action).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # 經驗回放緩衝區
        self.replay_buffer = ReplayBuffer(buffer_size, state_dim, action)
        
        # 訓練模式標記
        self.training_mode = False
        
        print(f"[TD3] Initialized on device: {self.device}")
        print(f"[TD3] Actor parameters: {sum(p.numel() for p in self.actor.parameters())}")
        print(f"[TD3] Twin Critic parameters: {sum(p.numel() for p in self.critic.parameters())}")
        print(f"[TD3] Policy delay: {policy_delay}, Policy noise: {policy_noise}, Noise clip: {noise_clip}")
    
    def act(self, observation: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        根據觀察選擇動作 (Polymorphism - override 父類方法)
        
        Args:
            observation: 當前狀態
            add_noise: 是否添加探索噪聲
        
        Returns:
            action: 選擇的動作
        """
        state = torch.FloatTensor(observation.reshape(1, -1)).to(self.device)
        
        # 使用 Actor 網路輸出動作
        with torch.no_grad():
            action = self.actor(state).cpu().numpy().flatten()
        
        # 訓練時添加噪聲以探索
        if add_noise and self.training_mode:
            noise = np.random.normal(0, self.exploration_noise * self.max_action, 
                                    size=self.action_dim)
            action = action + noise
            action = np.clip(action, -self.max_action, self.max_action)
        
        return action
    
    def reset(self):
        """
        重置智能體狀態 (Polymorphism - override 父類方法)
        """
        # TD3 不需要特別的重置操作
        pass
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, 
                        reward: float, next_state: np.ndarray, done: bool):
        """
        儲存經驗到回放緩衝區
        
        Args:
            state: 當前狀態
            action: 執行的動作
            reward: 獲得的獎勵
            next_state: 下一個狀態
            done: 是否結束
        """
        self.replay_buffer.store(state, action, reward, next_state, done)
    
    def train_step(self) -> Tuple[float, float]:
        """
        執行一步訓練更新（TD3 的核心演算法）
        
        Returns:
            critic_loss, actor_loss: Critic 和 Actor 的損失值
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0
        
        self.total_iterations += 1
        
        # 從回放緩衝區取樣
        batch = self.replay_buffer.sample(self.batch_size)
        state = torch.FloatTensor(batch['states']).to(self.device)
        action = torch.FloatTensor(batch['actions']).to(self.device)
        reward = torch.FloatTensor(batch['rewards']).to(self.device)
        next_state = torch.FloatTensor(batch['next_states']).to(self.device)
        done = torch.FloatTensor(batch['dones']).to(self.device)
        
        # ==================== 更新 Critic (Twin Q-Networks) ====================
        with torch.no_grad():
            # ===== Target Policy Smoothing =====
            # 在目標動作上加入裁剪噪聲：ã' = clip(μ'(s') + clip(ε, -c, c), -max, max)
            noise = (torch.randn_like(action) * self.policy_noise * self.max_action).clamp(
                -self.noise_clip * self.max_action, 
                self.noise_clip * self.max_action
            )
            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, 
                self.max_action
            )
            
            # ===== Clipped Double Q-Learning =====
            # 計算兩個目標 Q 值，取較小值：y = r + γ * min(Q1', Q2')
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * self.gamma * target_q
        
        # 當前兩個 Q 值
        current_q1, current_q2 = self.critic(state, action)
        
        # Critic 損失：MSE 對兩個 Q 網路分別計算
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)
        
        # 更新 Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ==================== Delayed Policy Updates ====================
        # 每 policy_delay 次才更新 Actor 和目標網路
        actor_loss = 0.0
        if self.total_iterations % self.policy_delay == 0:
            # Actor 損失：-Q1(s, μ(s))（最大化 Q 值）
            # 注意：TD3 只使用 Q1 來更新 Actor
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            
            # 更新 Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # 軟更新目標網路
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
            
            actor_loss = actor_loss.item()
        
        return critic_loss.item(), actor_loss
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """
        軟更新目標網路：θ' ← τθ + (1-τ)θ'
        
        Args:
            source: 源網路（當前網路）
            target: 目標網路
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
    
    def set_training_mode(self, mode: bool):
        """設定訓練/測試模式"""
        self.training_mode = mode
        if mode:
            self.actor.train()
            self.critic.train()
        else:
            self.actor.eval()
            self.critic.eval()
    
    def save(self, filepath: str):
        """
        儲存模型
        
        Args:
            filepath: 儲存路徑
        """
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'total_iterations': self.total_iterations,
        }, filepath)
        print(f"[TD3] Model saved to {filepath}")
    
    def load(self, filepath: str):
        """
        載入模型
        
        Args:
            filepath: 載入路徑
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.total_iterations = checkpoint.get('total_iterations', 0)
        print(f"[TD3] Model loaded from {filepath}")
