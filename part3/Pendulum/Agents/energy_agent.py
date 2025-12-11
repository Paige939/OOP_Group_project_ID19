import numpy as np
from base_agent import Agent

class EnergyControlAgent(Agent):
    """
    Energy-Based Controller (Swing-up Control)
    原理：透過調節系統總能量，將擺錘從下方擺盪至上方。
    """
    def __init__(self, action_dim: int, max_action: float, g=10.0, m=1.0, l=1.0):
        super().__init__(action=action_dim, max_action=max_action)
        self.g = g # 重力加速度 (Gym Pendulum 預設 10.0)
        self.m = m # 質量 (預設 1.0)
        self.l = l # 長度 (預設 1.0)
        
    def get_energy(self, cos_th, sin_th, th_dot):
        """
        計算當前系統總能量
        位能 Ep = mgl(1 - cos(theta))  <-- Gym 的座標系定義略有不同
        動能 Ek = 0.5 * m * (l^2) * (th_dot^2)
        """
        # 最低點 theta=pi, cos=-1 -> 位能=0; 最高點 theta=0, cos=1 -> 位能=2mgl
        
        p_energy = self.m * self.g * self.l * (cos_th - 1)
        k_energy = 0.5 * self.m * (self.l ** 2) * (th_dot ** 2)
        return p_energy + k_energy # 總能量 # 目標是讓 E 接近 0 (直立靜止)

    def act(self, observation: np.ndarray) -> np.ndarray:
        cos_theta = observation[0]
        sin_theta = observation[1]
        theta_dot = observation[2]

        # 計算當前能量
        energy = self.get_energy(cos_theta, sin_theta, theta_dot)
        
        # Control Law
        # u = -k * energy * theta_dot
        # 如果 energy < 0，且速度與力矩同向，就會增加能量
        k = 8 # 增益參數
        action = -k * energy * theta_dot

        # 動作限制
        action = np.clip(action, -self.max_action, self.max_action)
        
        return np.array([action], dtype=np.float32)

    def reset(self):
        pass