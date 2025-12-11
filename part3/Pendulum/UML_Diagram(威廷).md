# Part 3 Pendulum - UML 類別圖

## Class Diagram

```mermaid
classDiagram
    direction TB
    
    %% Abstract Base Class
    class Agent {
        &lt;&lt;abstract&gt;&gt;
        +action: int
        +max_action: float
        +__init__(action, max_action)
        +act(observation)* np.ndarray
        +reset()*
    }
    
    %% Concrete Agents
    class RandomAgent {
        +act(observation, add_noise) np.ndarray
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
        +act(observation, add_noise) np.ndarray
        +reset()
        +store_transition(s, a, r, s', done)
        +train_step() Tuple
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
        +act(observation, add_noise) np.ndarray
        +reset()
        +store_transition(s, a, r, s', done)
        +train_step() Tuple
        +save(filepath)
        +load(filepath)
        -_soft_update(source, target)
    }
    
    %% Neural Networks
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
    
    %% Buffer
    class ReplayBuffer {
        -capacity: int
        -ptr: int
        -size: int
        -states: np.ndarray
        -actions: np.ndarray
        -rewards: np.ndarray
        -next_states: np.ndarray
        -dones: np.ndarray
        +store(s, a, r, s', done)
        +sample(batch_size) Dict
        +__len__() int
    }
    
    %% Environment
    class PendulumEnvWrapper {
        -env: gym.Env
        +state: int
        +action: int
        +max_action: float
        -current_observation: np.ndarray
        +reset() np.ndarray
        +step(action) Tuple
        +render()
        +close()
        +get_state() np.ndarray
    }
    
    %% Experiment Manager
    class Experiment {
        -env: PendulumEnvWrapper
        -agent: Agent
        -episode_len: int
        +run_episode(render) float
        +train_episode() Tuple
        +close()
    }
    
    %% Inheritance
    Agent <|-- RandomAgent
    Agent <|-- DDPG_Agent
    Agent <|-- TD3_Agent
    
    %% Composition
    DDPG_Agent *-- ActorNetwork
    DDPG_Agent *-- CriticNetwork
    DDPG_Agent *-- ReplayBuffer
    
    TD3_Agent *-- ActorNetwork
    TD3_Agent *-- TwinCriticNetwork
    TD3_Agent *-- ReplayBuffer
    
    %% Association
    Experiment o-- PendulumEnvWrapper
    Experiment o-- Agent
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
