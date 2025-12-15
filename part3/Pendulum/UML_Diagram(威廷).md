# Part 3 Pendulum - UML 類別圖

## Class Diagram

```mermaid
classDiagram
    direction TB
    
    %% ===== Abstract Base Class =====
    class Agent {
        <<abstract>>
        +action: int
        +max_action: float
        +__init__(action, max_action)
        +act(observation)* np.ndarray
        +reset()*
        +pre_episode(env, episode_len)
    }
    
    %% ===== Concrete Agents =====
    class RandomAgent {
        +act(observation) np.ndarray
        +reset()
    }
    
    class DDPG_Agent {
        -device: torch.device
        -actor: ActorNetwork
        -critic: CriticNetwork
        -actor_target: ActorNetwork
        -critic_target: CriticNetwork
        -replay_buffer: ReplayBuffer
        -gamma: float
        -tau: float
        -batch_size: int
        -exploration_noise: float
        -training_mode: bool
        +act(observation, add_noise) np.ndarray
        +reset()
        +store_transition(state, action, reward, next_state, done)
        +train_step() Tuple[float, float]
        +set_training_mode(mode)
        +save(filepath)
        +load(filepath)
        -_soft_update(source, target)
    }
    
    class TD3_Agent {
        -device: torch.device
        -actor: ActorNetwork
        -critic: TwinCriticNetwork
        -actor_target: ActorNetwork
        -critic_target: TwinCriticNetwork
        -replay_buffer: ReplayBuffer
        -policy_delay: int
        -policy_noise: float
        -noise_clip: float
        -total_iterations: int
        +act(observation, add_noise) np.ndarray
        +reset()
        +store_transition(state, action, reward, next_state, done)
        +train_step() Tuple[float, float]
        +set_training_mode(mode)
        +save(filepath)
        +load(filepath)
        -_soft_update(source, target)
    }
    
    class CEM_Agent {
        -num_samples: int
        -elite_frac: float
        -num_elite: int
        -weights_mean: np.ndarray
        -weights_std: np.ndarray
        -best_weights: np.ndarray
        -save_path: str
        +act(observation) np.ndarray
        +reset()
        +pre_episode(env, episode_len)
        -evaluate(env, weights, episode_len) float
    }
    
    class LQRAgent {
        -K: np.ndarray
        -g: float
        -m: float
        -l: float
        +act(observation) np.ndarray
        +reset()
    }
    
    class EnergyControlAgent {
        -g: float
        -m: float
        -l: float
        +act(observation) np.ndarray
        +reset()
        +get_energy(cos_th, sin_th, th_dot) float
    }
    
    class ELAgent {
        -swing_up_agent: EnergyControlAgent
        -balance_agent: LQRAgent
        +act(observation) np.ndarray
        +reset()
    }
    
    %% ===== Neural Networks (nn.Module) =====
    class ActorNetwork {
        -max_action: float
        -hidden: nn.Sequential
        -output_layer: nn.Linear
        +forward(state) torch.Tensor
    }
    
    class CriticNetwork {
        -hidden: nn.Sequential
        -output_layer: nn.Linear
        +forward(state, action) torch.Tensor
    }
    
    class TwinCriticNetwork {
        -q1_hidden: nn.Sequential
        -q1_output: nn.Linear
        -q2_hidden: nn.Sequential
        -q2_output: nn.Linear
        +forward(state, action) Tuple
        +Q1(state, action) torch.Tensor
    }
    
    %% ===== Replay Buffer =====
    class ReplayBuffer {
        -capacity: int
        -ptr: int
        -size: int
        -states: np.ndarray
        -actions: np.ndarray
        -rewards: np.ndarray
        -next_states: np.ndarray
        -dones: np.ndarray
        +store(state, action, reward, next_state, done)
        +sample(batch_size) Dict
        +__len__() int
    }
    
    %% ===== Environment Wrapper =====
    class PendulumEnvWrapper {
        -env: gym.Env
        -current_observation: np.ndarray
        +state: int
        +action: int
        +max_action: float
        +reset() np.ndarray
        +step(action) Tuple
        +render()
        +close()
        +get_state() np.ndarray
    }
    
    %% ===== Experiment Manager =====
    class Experiment {
        -env: PendulumEnvWrapper
        -agent: Agent
        -episode_len: int
        +run_episode(render) float
        +close()
    }
    
    %% ========== Relationships ==========
    
    %% Inheritance (繼承)
    Agent <|-- RandomAgent : extends
    Agent <|-- DDPG_Agent : extends
    Agent <|-- TD3_Agent : extends
    Agent <|-- CEM_Agent : extends
    Agent <|-- LQRAgent : extends
    Agent <|-- EnergyControlAgent : extends
    Agent <|-- ELAgent : extends
    
    %% Composition (組合) - DDPG
    DDPG_Agent *-- ActorNetwork : actor
    DDPG_Agent *-- CriticNetwork : critic
    DDPG_Agent *-- ReplayBuffer : buffer
    
    %% Composition (組合) - TD3
    TD3_Agent *-- ActorNetwork : actor
    TD3_Agent *-- TwinCriticNetwork : twin_critic
    TD3_Agent *-- ReplayBuffer : buffer
    
    %% Composition (組合) - ELAgent (混合策略)
    ELAgent *-- EnergyControlAgent : swing_up
    ELAgent *-- LQRAgent : balance
    
    %% Aggregation (聚合) - Experiment
    Experiment o-- PendulumEnvWrapper : env
    Experiment o-- Agent : agent
```

---

## 如何查看 UML 圖

### 方法 1：VS Code 擴展
1. 安裝 **"Markdown Preview Mermaid Support"** 擴展
2. 打開此檔案，按 `Ctrl+Shift+V` 預覽

### 方法 2：線上工具
1. 打開 https://mermaid.live/
2. 複製上面的 mermaid 代碼貼上
3. 即可看到 UML 圖

### 方法 3：GitHub
- 直接把這個 .md 檔案 push 到 GitHub，GitHub 會自動渲染 Mermaid 圖表
