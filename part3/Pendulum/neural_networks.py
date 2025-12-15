import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ActorNetwork(nn.Module):
    """
    Actor 神經網路 (Deterministic Policy)
    * 輸入：state
    * 輸出：確定性動作 (deterministic action)
    * 用於 DDPG 和 TD3
    """
    def __init__(self, state_dim: int, action_dim: int, max_action: float,
                 hidden_dims: list = [256, 256]):
        """
        Args:
            state_dim: 狀態空間維度
            action_dim: 動作空間維度
            max_action: 動作的最大值（用於 tanh 縮放）
            hidden_dims: 隱藏層維度列表
        """
        super(ActorNetwork, self).__init__()
        self.max_action = max_action
        
        # 建立隱藏層
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        self.hidden = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, action_dim)
        
        # 初始化最後一層的權重（避免 tanh 飽和）
        self.output_layer.weight.data.uniform_(-3e-3, 3e-3)
        self.output_layer.bias.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        
        Args:
            state: 狀態張量 [batch_size, state_dim]
        
        Returns:
            action: 動作張量 [batch_size, action_dim]，範圍 [-max_action, max_action]
        """
        x = self.hidden(state)
        # 使用 tanh 激活函數並縮放到 [-max_action, max_action]
        action = self.max_action * torch.tanh(self.output_layer(x))
        return action


class CriticNetwork(nn.Module):
    """
    Critic 神經網路 (Q-value function)
    * 輸入：(state, action)
    * 輸出：Q-value (state-action value)
    * 用於 DDPG
    """
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: list = [256, 256]):
        """
        Args:
            state_dim: 狀態空間維度
            action_dim: 動作空間維度
            hidden_dims: 隱藏層維度列表
        """
        super(CriticNetwork, self).__init__()
        
        input_dim = state_dim + action_dim
        
        # 建立 Q 網路
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        self.hidden = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, 1)
        
        # 初始化最後一層
        self.output_layer.weight.data.uniform_(-3e-3, 3e-3)
        self.output_layer.bias.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        
        Args:
            state: 狀態張量 [batch_size, state_dim]
            action: 動作張量 [batch_size, action_dim]
        
        Returns:
            q_value: Q 值 [batch_size, 1]
        """
        # 連接 state 和 action
        sa = torch.cat([state, action], dim=1)
        x = self.hidden(sa)
        q_value = self.output_layer(x)
        return q_value


class TwinCriticNetwork(nn.Module):
    """
    Twin Critic 神經網路 (用於 TD3)
    * 包含兩個獨立的 Q 網路 (Q1, Q2)
    * 用於減少 Q 值過度估計問題 (Clipped Double Q-Learning)
    """
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: list = [256, 256]):
        """
        Args:
            state_dim: 狀態空間維度
            action_dim: 動作空間維度
            hidden_dims: 隱藏層維度列表
        """
        super(TwinCriticNetwork, self).__init__()
        
        input_dim = state_dim + action_dim
        
        # Q1 網路
        q1_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            q1_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        self.q1_hidden = nn.Sequential(*q1_layers)
        self.q1_output = nn.Linear(prev_dim, 1)
        
        # Q2 網路
        q2_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            q2_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        self.q2_hidden = nn.Sequential(*q2_layers)
        self.q2_output = nn.Linear(prev_dim, 1)
        
        # 初始化輸出層
        self.q1_output.weight.data.uniform_(-3e-3, 3e-3)
        self.q1_output.bias.data.uniform_(-3e-3, 3e-3)
        self.q2_output.weight.data.uniform_(-3e-3, 3e-3)
        self.q2_output.bias.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor):
        """
        前向傳播，計算兩個 Q 值
        
        Args:
            state: 狀態張量 [batch_size, state_dim]
            action: 動作張量 [batch_size, action_dim]
        
        Returns:
            q1, q2: 兩個獨立的 Q 值 [batch_size, 1]
        """
        sa = torch.cat([state, action], dim=1)
        
        # Q1
        q1 = self.q1_hidden(sa)
        q1 = self.q1_output(q1)
        
        # Q2
        q2 = self.q2_hidden(sa)
        q2 = self.q2_output(q2)
        
        return q1, q2
    
    def Q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        僅計算 Q1（用於 Actor 的梯度更新）
        
        Returns:
            q1: Q1 值 [batch_size, 1]
        """
        sa = torch.cat([state, action], dim=1)
        q1 = self.q1_hidden(sa)
        q1 = self.q1_output(q1)
        return q1
