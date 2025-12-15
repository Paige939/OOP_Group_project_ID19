import numpy as np
from typing import Dict, Tuple

class ReplayBuffer:
    """
    經驗回放緩衝區 (Encapsulation)
    * 儲存 (state, action, reward, next_state, done) 轉換
    * 提供隨機批次取樣功能
    * 使用 NumPy 預分配記憶體以提高效率
    """
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        """
        Args:
            capacity: 緩衝區最大容量
            state_dim: 狀態空間維度
            action_dim: 動作空間維度
        """
        self.capacity = capacity
        self.ptr = 0  # 目前寫入位置
        self.size = 0  # 目前已儲存的數量
        
        # 預分配記憶體（提高效率）
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
    
    def store(self, state: np.ndarray, action: np.ndarray, 
              reward: float, next_state: np.ndarray, done: bool):
        """儲存一筆經驗到緩衝區"""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        # 循環覆蓋舊資料
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        隨機取樣一個批次的經驗
        
        Returns:
            字典包含 'states', 'actions', 'rewards', 'next_states', 'dones'
        """
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        return {
            'states': self.states[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_states': self.next_states[indices],
            'dones': self.dones[indices]
        }
    
    def __len__(self) -> int:
        """返回目前緩衝區大小"""
        return self.size
