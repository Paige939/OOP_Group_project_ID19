"""
Pendulum 測試主程式
* 載入訓練好的模型
* 進行視覺化評估
* 支援多個 episode 的統計分析
"""

import numpy as np
from manage import PendulumEnvWrapper, Experiment
from Agents.DDPG_agent import DDPG_Agent
from Agents.TD3_agent import TD3_Agent
import argparse
import os


def test_agent(agent_type: str = "TD3",
               model_path: str = None,
               num_episodes: int = 10,
               render: bool = True):
    """
    測試訓練好的智能體
    
    Args:
        agent_type: "DDPG" 或 "TD3"
        model_path: 模型檔案路徑
        num_episodes: 測試回合數
        render: 是否視覺化
    """
    # 建立環境（測試時使用視覺化）
    render_mode = "human" if render else None
    env_wrapper = PendulumEnvWrapper(render_mode=render_mode)
    
    # 建立智能體
    print(f"建立 {agent_type} Agent...")
    if agent_type == "DDPG":
        agent = DDPG_Agent(
            action=env_wrapper.action,
            max_action=env_wrapper.max_action,
            state_dim=env_wrapper.state
        )
    elif agent_type == "TD3":
        agent = TD3_Agent(
            action=env_wrapper.action,
            max_action=env_wrapper.max_action,
            state_dim=env_wrapper.state
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    # 載入模型
    if model_path and os.path.exists(model_path):
        agent.load(model_path)
        print(f"✓ 模型已載入: {model_path}")
    else:
        print(f"⚠ 找不到模型檔案: {model_path}")
        print("  使用未訓練的模型進行測試...")
    
    # 設定為測試模式（不添加探索噪聲）
    agent.set_training_mode(False)
    
    # 建立實驗管理器
    experiment = Experiment(env=env_wrapper, agent=agent, episode_len=200)
    
    # 執行測試
    print(f"\n{'='*60}")
    print(f"開始測試 {agent_type} Agent")
    print(f"測試回合數: {num_episodes}")
    print(f"{'='*60}\n")
    
    rewards = []
    for episode in range(1, num_episodes + 1):
        total_reward = experiment.run_episode(render=render)
        rewards.append(total_reward)
        print(f"Episode {episode:2d} | Total Reward: {total_reward:8.2f}")
    
    # 統計結果
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    max_reward = np.max(rewards)
    min_reward = np.min(rewards)
    
    print(f"\n{'='*60}")
    print(f"測試結果統計:")
    print(f"  平均獎勵: {mean_reward:8.2f} ± {std_reward:6.2f}")
    print(f"  最大獎勵: {max_reward:8.2f}")
    print(f"  最小獎勵: {min_reward:8.2f}")
    print(f"{'='*60}\n")
    
    experiment.close()


def compare_agents():
    """
    比較 RandomAgent, DDPG, TD3 的效能 (展現 Polymorphism)
    """
    from Agents.random_agent import RandomAgent
    
    print("\n" + "="*70)
    print("Compare RandomAgent, DDPG, TD3 - 展現 Polymorphism")
    print("="*70 + "\n")
    
    env_wrapper = PendulumEnvWrapper(render_mode=None)
    num_test_episodes = 5
    
    # 準備不同的智能體（都繼承自同一個 Agent 基類）
    agents = {
        'Random': RandomAgent(
            action=env_wrapper.action,
            max_action=env_wrapper.max_action
        ),
        'DDPG': DDPG_Agent(
            action=env_wrapper.action,
            max_action=env_wrapper.max_action,
            state_dim=env_wrapper.state
        ),
        'TD3': TD3_Agent(
            action=env_wrapper.action,
            max_action=env_wrapper.max_action,
            state_dim=env_wrapper.state
        )
    }
    
    # 嘗試載入訓練好的模型（如果存在）
    for name in ['DDPG', 'TD3']:
        model_path = f"results/{name}/best_model.pth"
        if os.path.exists(model_path):
            agents[name].load(model_path)
            print(f"✓ 載入 {name} 模型")
        else:
            print(f"⚠ 未找到 {name} 模型，使用未訓練版本")
    
    print()
    
    # 測試所有智能體（Polymorphism - 使用相同介面）
    results = {}
    for name, agent in agents.items():
        print(f"測試 {name} Agent...")
        if hasattr(agent, 'set_training_mode'):
            agent.set_training_mode(False)
        
        experiment = Experiment(env=env_wrapper, agent=agent, episode_len=200)
        rewards = []
        
        for ep in range(num_test_episodes):
            reward = experiment.run_episode(render=False)
            rewards.append(reward)
        
        results[name] = {
            'mean': np.mean(rewards),
            'std': np.std(rewards),
            'max': np.max(rewards),
            'min': np.min(rewards)
        }
        
        print(f"  平均獎勵: {results[name]['mean']:8.2f} ± {results[name]['std']:6.2f}\n")
    
    # 顯示比較結果
    print("="*70)
    print(f"{'Agent':<15} {'Mean Reward':<20} {'Max Reward':<15} {'Min Reward':<15}")
    print("-"*70)
    for name, stats in results.items():
        print(f"{name:<15} {stats['mean']:8.2f} ± {stats['std']:6.2f}    "
              f"{stats['max']:8.2f}      {stats['min']:8.2f}")
    print("="*70 + "\n")
    
    env_wrapper.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Pendulum Agent')
    parser.add_argument('--agent', type=str, default='TD3', choices=['DDPG', 'TD3'],
                       help='Agent type (default: TD3)')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model file (e.g., results/TD3_xxx/best_model.pth)')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of test episodes (default: 10)')
    parser.add_argument('--no-render', action='store_true',
                       help='Disable rendering')
    parser.add_argument('--compare', action='store_true',
                       help='Compare all agents (Random, DDPG, TD3)')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_agents()
    else:
        test_agent(
            agent_type=args.agent,
            model_path=args.model,
            num_episodes=args.episodes,
            render=not args.no_render
        )
