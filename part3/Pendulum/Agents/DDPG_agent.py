import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple
import copy

from base_agent import Agent
from replay_buffer import ReplayBuffer
from neural_networks import ActorNetwork, CriticNetwork


class DDPG_Agent(Agent):
    """
    DDPG (Deep Deterministic Policy Gradient) Agent
    * 繼承自 Agent 基類 (Inheritance + Polymorphism)
    * 使用 Actor-Critic 架構處理連續動作空間
    * 包含經驗回放和目標網路機制
    
    演算法核心：
    1. Actor 輸出確定性動作：a = μ(s)
    2. Critic 評估 Q(s,a)
    3. 使用經驗回放穩定訓練
    4. 使用目標網路減少更新振盪
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
            device: 計算裝置 ('cuda' 或 'cpu')
        """
        super().__init__(action, max_action)
        
        # 設定裝置
        if device is None:  # 自動選擇裝置
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        else:   # 使用指定裝置
            self.device = torch.device(device)
        
        """ 這裡就是把變數初始化 """
        self.state_dim = state_dim
        self.action_dim = action
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.exploration_noise = exploration_noise
        
        # 建立 Actor 網路和目標網路
        self.actor = ActorNetwork(state_dim, action, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # 建立 Critic 網路和目標網路
        self.critic = CriticNetwork(state_dim, action).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # 經驗回放緩衝區
        self.replay_buffer = ReplayBuffer(buffer_size, state_dim, action)
        
        # 訓練模式標記
        self.training_mode = False
        
        print(f"[DDPG] Initialized on device: {self.device}")
        print(f"[DDPG] Actor parameters: {sum(p.numel() for p in self.actor.parameters())}")
        print(f"[DDPG] Critic parameters: {sum(p.numel() for p in self.critic.parameters())}")
    
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
        # DDPG 不需要特別的重置操作
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
        執行一步訓練更新
        
        Returns:
            critic_loss, actor_loss: Critic 和 Actor 的損失值
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0
        
        # 從回放緩衝區取樣
        batch = self.replay_buffer.sample(self.batch_size)
        state = torch.FloatTensor(batch['states']).to(self.device)
        action = torch.FloatTensor(batch['actions']).to(self.device)
        reward = torch.FloatTensor(batch['rewards']).to(self.device)
        next_state = torch.FloatTensor(batch['next_states']).to(self.device)
        done = torch.FloatTensor(batch['dones']).to(self.device)
        
        # ==================== 更新 Critic ====================
        with torch.no_grad():
            # 計算目標 Q 值：y = r + γ * Q'(s', μ'(s'))
            next_action = self.actor_target(next_state)
            target_q = self.critic_target(next_state, next_action)
            target_q = reward + (1 - done) * self.gamma * target_q
        
        # 當前 Q 值
        current_q = self.critic(state, action)
        
        # Critic 損失：MSE(Q(s,a), y)
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        # 更新 Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ==================== 更新 Actor ====================
        # Actor 損失：-Q(s, μ(s))（最大化 Q 值）
        actor_loss = -self.critic(state, self.actor(state)).mean()
        
        # 更新 Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ==================== 軟更新目標網路 ====================
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        
        return critic_loss.item(), actor_loss.item()
    
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
        }, filepath)
        print(f"[DDPG] Model saved to {filepath}")
    
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
        print(f"[DDPG] Model loaded from {filepath}")
