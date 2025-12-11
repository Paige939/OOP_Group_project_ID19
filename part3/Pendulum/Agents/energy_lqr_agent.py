import numpy as np
from base_agent import Agent
from .energy_agent import EnergyControlAgent
from .lqr_agent import LQRAgent# 相對路徑

class ELAgent(Agent):
    """
    Hybrid Agent (混合型) ('Composition')
    根據狀態「切換」使用不同的策略物件。
    """
    def __init__(self, action_dim: int, max_action: float):
        super().__init__(action=action_dim, max_action=max_action)
        
        # 內部實例化兩個子 Agent
        self.swing_up_agent = EnergyControlAgent(action_dim, max_action)
        self.balance_agent = LQRAgent(action_dim, max_action)
        
    def act(self, observation: np.ndarray) -> np.ndarray:
        cos_theta = observation[0]
        sin_theta = observation[1]
        
        # 計算當前角度 
        theta = np.arctan2(sin_theta, cos_theta) # 0 是最高點, pi/-pi 是最低點

        # --- 切換 ---
        if abs(theta) < 0.5: # 角度小: +- 0.5 弧度內 -> LQR 模式
            # print("Mode: Balancing (LQR)")
            return self.balance_agent.act(observation)
        else: # 能量控制 模式
            # print("Mode: Swing Up (Energy)")
            return self.swing_up_agent.act(observation)

    def reset(self):
        self.swing_up_agent.reset()
        self.balance_agent.reset()