# Project Overview
## Project Structure
```
project
├── part1
├── part2
|   └──  frozen_lake.py
|
└── part3/Pendulum/
          ├── base_agent.py          # Abstract Agent base class (Inheritance)
          ├── manage.py              # Environment wrapper & Experiment manager (Encapsulation)
          ├── neural_networks.py     # Actor/Critic neural networks
          ├── replay_buffer.py       # Experience replay buffer
          ├── Agents/
          │   ├── random_agent.py    # Random baseline agent
          │   ├── DDPG_agent.py      # DDPG implementation
          |   ├── TD3_agent.py       # TD3 implementation
          |   ├── energy_agent.py    # Energy implementation
          |   ├── energy_lqr_agent.py
          |   ├── lqr_agent.py
          │   └── CEM_agent.py       # CEM implementation
          ├── train_main.py          # Training script
          ├── test_main.py           # Testing & comparison script
          ├── main.py                # Simple demo with RandomAgent, CEM_agent
          │
          ├── results/               # Trained models & curves
          │   ├── TD3/
          │   └── DDPG/
          │
          └── requirements.txt       # Dependencies
```
## PART 1
A mountain car game to test if the installation is success.
## PART 2
A frozen lake game to enhance its consistent success rate > 0.7.
## PART 3
### What is Pendulum?
- **State**: 3D vector (`cos(θ)`, `sin(θ)`, `θ̇`) representing the pendulum angle and angular velocity
- **Action**: Continuous torque ∈ [-2, +2]
- **Objective**: Keep the pendulum upright (θ = 0) with minimum effort
- **Reward**: Ranges from -16.28 to 0, higher (closer to 0) is better

### Pendulum Control with Deep Reinforcement Learning 

This project implements **deep reinforcement learning agents** to solve the Pendulum-v1 control problem from OpenAI Gymnasium. The goal is to swing up and balance an inverted pendulum using continuous torque control.

#### Implemented Agents

| Agent | Description |
|-------|-------------|
| **RandomAgent** | Baseline agent with random actions |
| **DDPG** | Deep Deterministic Policy Gradient - Actor-Critic for continuous control |
| **TD3** | Twin Delayed DDPG - Improved version with three key enhancements |
| **CEM** | Cross Entropy Method - Find the optimal set of weights for a linear policy that maximizes the cumulative reward in the Pendulum environment|

#### OOP Principles Demonstrated

1. **Inheritance**: `DDPG_Agent` , `CEM_agent` and `TD3_Agent` inherit from abstract `Agent` base class
2. **Polymorphism**: All agents share the same interface (`act()`, `reset()`), allowing interchangeable use
3. **Encapsulation**: Environment logic wrapped in `PendulumEnvWrapper`, neural networks encapsulated in separate modules
4. **Abstraction**: `Agent` base class defines abstract interface, hiding implementation details


#### Training Results Explanation

After training, results are saved in `results/{AGENT}/`:

**生成檔案說明**

| 檔案 | 說明 |
|------|------|
| `best_model.pth` | **最佳模型**：訓練過程中測試獎勵最高的模型權重，通常是最好的選擇 |
| `final_model.pth` | **最終模型**：訓練結束時的模型權重，不一定是最好的（可能過擬合）|
| `checkpoint_epXX.pth` | **檢查點**：每隔一段時間儲存的模型（如 ep20, ep40...），用於恢復訓練或比較不同階段 |
| `training_curves.png` | **訓練曲線圖**：視覺化訓練過程（詳見下方說明）|
| `training_data.npz` | **原始訓練數據**：包含每個 episode 的獎勵和損失值，可用於自行繪圖分析 |

**DDPG & TD3 訓練曲線圖解釋 (training_curves.png)**

訓練曲線包含 4 個子圖：

**1. Training Rewards（左上）- 訓練獎勵**
- **藍線**：每個 episode 的訓練的 Reward 值
- **紅線**：移動平均線（平滑後的趨勢）          // 每 10 個取平均變成平滑曲線上的一個點的意思
- **預期變化**：從約 -1500 開始，逐漸提升至 -200 ~ -400
- **解讀**：數值越高（越接近 0）= 控制效果越好

**2. Test Rewards（右上）- 測試獎勵**
- **綠點**：評估階段的獎勵（不加探索噪聲）
- 代表策略的真實效能，不受隨機探索影響
- **預期變化**：隨著訓練越減越少，越接近 0 的意思

**3. Critic Loss（左下）- Critic 損失**
- 衡量 Critic 網路預測 Q 值的準確度
- **預期變化**：應該逐漸下降並趨於穩定
- 初期數值較高是正常的

**4. Actor Loss（右下）- Actor 損失**
- 衡量策略改進的方向
- **注意**：負值是正常的（因為我們是最大化 Q 值）
- **預期變化**：應該趨於穩定（不一定會下降）

### Performance Benchmarks
Training: 100 episodes, Test: 5 episods
| Agent  | Mean RewardMax         | Reward   | Min Reward  | Explanation                         |
|:------ |:-----------------------:|---------:|------------:|:------------------------------------|
| Random | -1188.38 ± 291.14       | -751.38  | -1651.14    | 完全隨機，無法控制                   |
| DDPG   | -231.29 ± 64.79         | -125.10  | -328.81     | 有學習效果                           |
| TD3    | -167.90 ± 56.37         | -117.80  | -245.44     | 最佳表現                             |
| CEM    | -200 ~ -1500（Linear）  | —        | —           | Linear vary highly（變動很大）     |
> **Note**: Closer to 0 = better performance

**Random/ CEM/ LQR/ Energy/ Energy+LQR Comparison**

<img width="700" height="250" alt="image" src="https://github.com/user-attachments/assets/16d7bacf-d9db-4f7e-97ea-c0f396b3b9e9" />

# 演算法說明

## DDPG (Deep Deterministic Policy Gradient)

DDPG 是專為**連續動作空間**設計的演算法，結合了 Actor-Critic 架構：

- **Actor 網路**：輸入狀態 s，輸出確定性動作 a = π(s)
- **Critic 網路**：輸入 (s, a)，輸出 Q 值估計 Q(s, a)
- **核心機制**：
  - **經驗回放 (Experience Replay)**：把過去的經驗存起來隨機取樣訓練，打破資料相關性
  - **目標網路 (Target Networks)**：用另一組緩慢更新的網路計算目標值，穩定訓練
  - **軟更新 (Soft Updates)**：目標網路參數緩慢追蹤當前網路 θ' ← τθ + (1-τ)θ'

## TD3 (Twin Delayed DDPG)

TD3 是 DDPG 的改進版本，解決了 Q 值**過度估計**的問題，有三大核心改進：

1. **Clipped Double Q-Learning**（雙 Q 網路裁剪）
   - 使用兩個 Critic 網路 (Q1, Q2)
   - 計算目標時取較小值：y = r + γ × min(Q1', Q2')
   - 減少 Q 值過度樂觀估計的問題

2. **Delayed Policy Updates**（延遲策略更新）
   - Critic 更新多次後，Actor 才更新一次（預設 2:1）(d=2)
   - 讓 Critic 有足夠時間收斂，減少 Actor 更新的誤差
-          // 這裡的參數像 d 都可以再調整看怎樣最好，不過這不是本專案的重點，故跳過不討論​

3. **Target Policy Smoothing**（目標策略平滑）
   - 在目標動作上加入裁剪過的噪聲
   - 讓策略對動作的微小變化更穩健

## CEM (Cross Entropy Method)
- CEM 是一種隨機優化算法，最初用於罕見事件的機率估算，後來被廣泛應用於連續或離散空間的優化問題，包括強化學習中的策略參數優化。
- CEM 並不是直接優化單一的策略參數 $W$，而是優化一個參數的機率分佈（在程式碼中是高斯分佈 $\mathcal{N}(\mu, \sigma)$）。其基本思想是：
1. 假設：最佳的策略參數 $W^*$ 落在當前機率分佈的 高密度區域 內。
2. 方法：在每次迭代中，我們採樣許多策略，然後只選擇表現最好的 精英樣本 (Elite Samples)。
3. 調整：根據這些精英樣本，我們調整機率分佈的參數（ $\mu$ 和 $\sigma$ ），使分佈的中心（ $\mu$ ）向精英的平均值移動，並且分佈的範圍（ $\sigma$ ）縮小。

- 這個過程會讓機率分佈快速地收斂到最佳策略參數的極小區域。

## Classical Control Algorithms (Physics-based)

除了強化學習 (RL) 方法，我們也實作了基於物理模型的經典控制演算法，作為與 RL Agent 的對照組。這些方法不需要訓練神經網路，而是直接利用物理公式進行控制。

### 1. Energy Shaping Control (能量控制)
能量控制的核心概念是利用能量守恆定律。我們計算單擺目前的總能量，並與「直立靜止時的目標能量」進行比較，透過施加力矩來注入或移除能量。

- **原理**：
  - 單擺總能量 $E = \frac{1}{2}ml^2 \dot{\theta}^2 + mgl(\cos\theta - 1)$
  - 目標能量 $E_{target}$ 為直立靜止時的能量 (設為 0)
- **控制律**：
  - $u = -k_E \cdot \dot{\theta} \cdot (E - E_{target})$
  - 當能量不足時 ($E < E_{target}$)，順著速度方向推一把 (起擺)。
  - 當能量過多時 ($E > E_{target}$)，逆著速度方向阻擋 (煞車)。
- **特點**：非常擅長將單擺從底部甩至高點，但在最高點難以精確穩定。

### 2. LQR (Linear Quadratic Regulator)
LQR 是一種最佳控制策略，適用於線性系統。

- **線性化 (Linearization)**：
  - 在單擺直立平衡點 ($\theta \approx 0$) 將非線性物理方程線性化為 $\dot{x} = Ax + Bu$。
  - $A$ 矩陣代表重力與慣性對狀態的影響， $B$ 矩陣代表力矩輸入的影響。
- **最佳化目標**：
  - 定義狀態懲罰矩陣 $Q$ (重視角度誤差) 與控制懲罰矩陣 $R$ (重視省力)。
  - 解 Riccati 方程 (ARE) 算出最佳增益矩陣 $K$。
- **控制律**：
  - $u = -K \cdot x = -(k_1 \theta + k_2 \dot{\theta})$
- **特點**：在平衡點附近極度穩定且節能，但無法處理大幅度的起擺動作，難以從底部往上擺至最高點。

### 3. Hybrid Control (Energy + LQR)
結合了上述兩種方法的優點，形成一套完整的控制策略。

- **策略切換邏輯**：
  1. **Swing-up 階段**：當角度偏差較大時 (如 $|\theta| > 0.2$ rad)，使用 **Energy Shaping** 快速累積能量甩至高點。
  2. **Balancing 階段**：當單擺進入直立點附近時 (如 $|\theta| \le 0.2$ rad)，切換至 **LQR** 進行精確穩壓。
- **優勢**：結合了能量控制的廣域性與 LQR 的局部精確性，通常能獲得穩定且接近理論最佳值的效能。

# How to Run (DDPG & TD3 by 威廷)
### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv         #只有第一次要

# Activate (Linux/WSL)
source venv/bin/activate

# Activate (Windows)
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Training

```bash
# Train TD3 agent (recommended)
python train_main.py --agent TD3 --episodes 100

# Train DDPG agent
python train_main.py --agent DDPG --episodes 100

# Quick test (fewer episodes)
python train_main.py --agent TD3 --episodes 20 --warmup 500
```

### 3. Testing

```bash
# Test TD3 with visualization
python test_main.py --agent TD3 --model results/TD3/best_model.pth --episodes 5

# Test DDPG with visualization
python test_main.py --agent DDPG --model results/DDPG/best_model.pth --episodes 5

# Test without visualization (faster)
python test_main.py --agent TD3 --model results/TD3/best_model.pth --episodes 10 --no-render
python test_main.py --agent DDPG --model results/DDPG/best_model.pth --episodes 10 --no-render

# Compare all agents (Random vs DDPG vs TD3) - demonstrates Polymorphism
python test_main.py --compare           # num_test_episodes = 5                    compare 不會畫圖和產生 results 資料夾，這裡是運行(test)，只有在前面 train 的時候才會有
```

### 4. Other Demo

```bash
# Run RandomAgent, CEM_Agent, EnergyControlAgent, LQRAgent, and ELAgent(Hybrid) demo
python main.py
```


# Dependencies

| Package | Version | Description |
|---------|---------|-------------|
| numpy | ≥1.21.0 | Numerical computing |
| matplotlib | ≥3.5.0 | Plotting training curves |
| gymnasium | ≥0.29.0 | Reinforcement learning environments |
| torch | ≥2.0.0 | Deep learning framework (PyTorch) |
| pygame | 2.6.1 | Render (Visualization) |

Install all dependencies:
```bash
pip install numpy matplotlib gymnasium torch
pip install pygame
```

# Contribution List
| **Member Name** | **Contribution** |
|-----------------|------------------|
| 謝佩均 (B123245004) | Part3 整體專案架構及其OOP實作、Cross Entropy Method實作 & CEM部分reflection, readme, demo slide |
|  江威廷 (B123245021)   | part 3 DDPG & TD3 實作 & reflection paper 主要部分 & readme & demo slide，問題定義，UML圖表，整合不同Agent使用之manage.py |
|   黃柏薰  (B123040046)    |    Part3 lqr & energyControl & 組合Agent 實作 & demo slide&readme(Physics-based) |
