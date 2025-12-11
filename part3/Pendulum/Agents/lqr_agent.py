import numpy as np
from base_agent import Agent
import scipy.linalg # 記得匯入這個

class LQRAgent(Agent):
    """
    LQR (Linear Quadratic Regulator) Agent
    適用於：當擺錘已經在頂端附近時，用來維持穩定。
    """
    def __init__(self, action_dim: int, max_action: float, g=10.0, m=1.0, l=1.0):
        super().__init__(action=action_dim, max_action=max_action)
        '''# 這些 K 值是針對 Pendulum-v1 (m=1, l=1, g=10) 計算出來的經驗值
        # K = [k_theta, k_theta_dot]：對角度的敏感度, 對角速度的敏感度
        self.K = np.array([40.0, 10.0]) '''

        # --- 1. 定義系統模型 (A, B) ---
        # 根據 Gymnasium Pendulum-v1 的原始碼：
        # 角加速度公式：theta_acc = (3g / 2l) * sin(theta) + (3 / ml^2) * torque
        # 線性化後 (sin(theta) ~ theta)：
        # theta_acc = (3g / 2l) * theta + (3 / ml^2) * torque
        
        # A 矩陣 (狀態轉移): [theta, theta_dot]
        # d(theta)/dt = theta_dot
        # d(theta_dot)/dt = (3g/2l) * theta
        a_21 = (3 * g) / (2 * l)  # 結果是 15.0
        A = np.array([[0, 1], 
                      [15.0, 0]])
        
        # B 矩陣 (控制輸入):
        # d(theta_dot)/dt ... + (3/ml^2) * u
        b_21 = 3 / (m * (l ** 2)) # 結果是 3.0
        B = np.array([[0], 
                      [b_21]])

        # --- 2. 定義成本權重 (Q, R) ---
        # Q: 對誤差的懲罰 (State Cost)
        # 我們希望角度(theta)非常準(10)，速度(theta_dot)稍微準就好(1)
        Q = np.array([[10, 0], 
                      [0, 1]])
                      
        # R: 對力量的懲罰 (Control Cost)
        # R 越小，代表我們願意用更大的力氣去修正誤差
        R = np.array([[1]])

        # --- 3. 解 Riccati 方程 (ARE) ---
        # P 是解出來的矩陣
        P = scipy.linalg.solve_continuous_are(A, B, Q, R)

        # --- 4. 計算 K (Feedback Gain) ---
        # K = R^-1 * B^T * P
        # 這行運算後，K 大約會是 [[46.9, 11.2]] (比原本的 40, 10 更精準)
        self.K = np.linalg.inv(R) @ B.T @ P
        
        # 為了除錯方便，可以印出來看看
        print(f"LQR Calculated K: {self.K}")

    def act(self, observation: np.ndarray) -> np.ndarray:
        cos_theta = observation[0]
        sin_theta = observation[1]
        theta_dot = observation[2]

        # 還原角度 theta
        theta = np.arctan2(sin_theta, cos_theta) # 0 表(垂直向上)

        # LQR 控制公式: u = -Kx ---> u = - (k1 * theta + k2 * theta_dot)
        state_vector = np.array([[theta], [theta_dot]]) # 狀態向量: 形狀 (2, 1)
        # 注意矩陣乘法形狀：(1,2) @ (2,1) -> (1,1)
        force = -self.K @ state_vector
        
        # 取出純量數值
        action = force.item()

        # 限制動作範圍 (Clipping)
        action = np.clip(action, -self.max_action, self.max_action)
        
        return np.array([action], dtype=np.float32)

    def reset(self):
        pass