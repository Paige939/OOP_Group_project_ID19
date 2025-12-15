# Project Overview
```
project/
├── part1
├── part2
│   └── frozen_lake.py
└── part3
    ├── agents/
    │   ├── __init__.py
    │   ├── random_agent.py
    |   ├── CEM_agent.py
    |   ├── DDPG_agent.py
    |   ├── TD3_agent.py
    |   ├── energy_agent.py
    |   ├── energyz_lqr_agent.py
    │   └── lqr_agent.py
    ├── base_agent.py
    ├── manage.py
    ├── neural_networks.py
    ├── replay_buffer.py
    └── main.py
```
## PART 1
A mountain car game to test if the installation is success.
## PART 2
A frozen lake game to enhance its consistent success rate > 0.7.
## PART 3
### Overall Structure for Pendulum

#### What is Pendulum?
- **State**: 3D vector (`cos(θ)`, `sin(θ)`, `θ̇`) representing the pendulum angle and angular velocity
- **Action**: Continuous torque ∈ [-2, +2]
- **Objective**: Keep the pendulum upright (θ = 0) with minimum effort
- **Reward**: Ranges from -16.28 to 0, higher (closer to 0) is better

### Pendulum Control with Cross Entropy Method

### Pendulum Control with Deep Reinforcement Learning (威廷)
#### Overview

This project implements **deep reinforcement learning agents** to solve the Pendulum-v1 control problem from OpenAI Gymnasium. The goal is to swing up and balance an inverted pendulum using continuous torque control.

#### Implemented Agents

| Agent | Description |
|-------|-------------|
| **RandomAgent** | Baseline agent with random actions |
| **DDPG** | Deep Deterministic Policy Gradient - Actor-Critic for continuous control |
| **TD3** | Twin Delayed DDPG - Improved version with three key enhancements |

#### OOP Principles Demonstrated

1. **Inheritance**: `DDPG_Agent` and `TD3_Agent` inherit from abstract `Agent` base class
2. **Polymorphism**: All agents share the same interface (`act()`, `reset()`), allowing interchangeable use
3. **Encapsulation**: Environment logic wrapped in `PendulumEnvWrapper`, neural networks encapsulated in separate modules
4. **Abstraction**: `Agent` base class defines abstract interface, hiding implementation details

---

#### Project Structure
重複部分為有括號標示(威廷)之檔案
```
Pendulum/
├── base_agent.py          # Abstract Agent base class (Inheritance)
├── manage.py              # Environment wrapper & Experiment manager (Encapsulation)
├── neural_networks.py     # Actor/Critic neural networks
├── replay_buffer.py       # Experience replay buffer
│
├── Agents/
│   ├── random_agent.py    # Random baseline agent
│   ├── DDPG_agent.py      # DDPG implementation
│   └── TD3_agent.py       # TD3 implementation
│
├── train_main.py          # Training script
├── test_main.py           # Testing & comparison script
├── main.py                # Simple demo with RandomAgent
│
├── results/               # Trained models & curves
│   ├── TD3/
│   └── DDPG/
│
└── requirements.txt       # Dependencies
```

#### Training Results Explanation

After training, results are saved in `results/{AGENT}/`:

#### 生成檔案說明

| 檔案 | 說明 |
|------|------|
| `best_model.pth` | **最佳模型**：訓練過程中測試獎勵最高的模型權重，通常是最好的選擇 |
| `final_model.pth` | **最終模型**：訓練結束時的模型權重，不一定是最好的（可能過擬合）|
| `checkpoint_epXX.pth` | **檢查點**：每隔一段時間儲存的模型（如 ep20, ep40...），用於恢復訓練或比較不同階段 |
| `training_curves.png` | **訓練曲線圖**：視覺化訓練過程（詳見下方說明）|
| `training_data.npz` | **原始訓練數據**：包含每個 episode 的獎勵和損失值，可用於自行繪圖分析 |

#### 訓練曲線圖解讀 (training_curves.png)

訓練曲線包含 4 個子圖：

**1. Training Rewards（左上）- 訓練獎勵**
- **藍線**：每個 episode 的訓練獎勵
- **紅線**：移動平均線（平滑後的趨勢）
- **預期變化**：從約 -1500 開始，逐漸提升至 -200 ~ -400
- **解讀**：數值越高（越接近 0）= 控制效果越好

**2. Test Rewards（右上）- 測試獎勵**
- **綠點**：評估階段的獎勵（不加探索噪聲）
- 代表策略的真實效能，不受隨機探索影響
- **預期變化**：比訓練獎勵更穩定

**3. Critic Loss（左下）- Critic 損失**
- 衡量 Critic 網路預測 Q 值的準確度
- **預期變化**：應該逐漸下降並趨於穩定
- 初期數值較高是正常的

**4. Actor Loss（右下）- Actor 損失**
- 衡量策略改進的方向
- **注意**：負值是正常的（因為我們是最大化 Q 值）
- **預期變化**：應該趨於穩定（不一定會下降）

### Performance Benchmarks

| Agent | Typical Reward | Training Time |
|-------|---------------|---------------|
| Random | -1200 ~ -1500 | N/A |
| DDPG (100 ep) | -300 ~ -600 | ~10 min |
| TD3 (100 ep) | -200 ~ -400 | ~10 min |

> **Note**: Closer to 0 = better performance

---

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
   - Critic 更新多次後，Actor 才更新一次（預設 2:1）
   - 讓 Critic 有足夠時間收斂，減少 Actor 更新的誤差

3. **Target Policy Smoothing**（目標策略平滑）
   - 在目標動作上加入裁剪過的噪聲
   - 讓策略對動作的微小變化更穩健


---


# How to Run
## PART 1
## PART 2
## PART 3
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
# Test with visualization
python test_main.py --agent TD3 --model results/TD3/best_model.pth --episodes 5

# Test without visualization (faster)
python test_main.py --agent TD3 --model results/TD3/best_model.pth --episodes 10 --no-render

# Compare all agents (Random vs DDPG vs TD3) - demonstrates Polymorphism
python test_main.py --compare           # num_test_episodes = 5
```

### 4. Quick Demo

```bash
# Run RandomAgent demo
python main.py
```

---

# Dependencies

| Package | Version | Description |
|---------|---------|-------------|
| numpy | ≥1.21.0 | Numerical computing |
| matplotlib | ≥3.5.0 | Plotting training curves |
| gymnasium | ≥0.29.0 | Reinforcement learning environments |
| torch | ≥2.0.0 | Deep learning framework (PyTorch) |

Install all dependencies:
```bash
pip install numpy matplotlib gymnasium torch
```

---
# Contribution List


