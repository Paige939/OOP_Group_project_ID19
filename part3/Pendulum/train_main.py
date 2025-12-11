"""
Pendulum 訓練主程式
* 示範 DDPG 和 TD3 的訓練流程
* 展現 Polymorphism：不同 Agent 使用相同介面
* 包含訓練曲線記錄和模型儲存
"""

import numpy as np
import matplotlib.pyplot as plt
from manage import PendulumEnvWrapper, Experiment
from Agents.DDPG_agent import DDPG_Agent
from Agents.TD3_agent import TD3_Agent
import argparse
import os
from datetime import datetime


def train_agent(agent_type: str = "TD3", 
                total_episodes: int = 100,
                warmup_steps: int = 1000,
                save_interval: int = 20,
                test_interval: int = 5):
    """
    訓練智能體
    
    Args:
        agent_type: "DDPG" 或 "TD3"
        total_episodes: 總訓練回合數
        warmup_steps: 隨機探索步數（填充緩衝區）
        save_interval: 儲存模型的間隔
        test_interval: 測試評估的間隔
    """
    # 建立環境
    env_wrapper = PendulumEnvWrapper(render_mode=None)
    
    # 建立智能體 (Polymorphism - 使用相同介面但不同實作)
    if agent_type == "DDPG":
        agent = DDPG_Agent(
            action=env_wrapper.action,
            max_action=env_wrapper.max_action,
            state_dim=env_wrapper.state,
            gamma=0.99,
            tau=0.005,
            actor_lr=1e-3,
            critic_lr=1e-3,
            buffer_size=50000,
            batch_size=256,
            exploration_noise=0.1
        )
    elif agent_type == "TD3":
        agent = TD3_Agent(
            action=env_wrapper.action,
            max_action=env_wrapper.max_action,
            state_dim=env_wrapper.state,
            gamma=0.99,
            tau=0.005,
            actor_lr=1e-3,
            critic_lr=1e-3,
            buffer_size=50000,
            batch_size=256,
            exploration_noise=0.1,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_delay=2
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    # 建立實驗管理器
    experiment = Experiment(env=env_wrapper, agent=agent, episode_len=200)
    
    # 建立儲存目錄
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # save_dir = f"results/{agent_type}_{timestamp}"    #這個會有時間標記
    save_dir = f"results/{agent_type}"
    os.makedirs(save_dir, exist_ok=True)
    
    # 記錄訓練數據
    train_rewards = []
    test_rewards = []
    critic_losses = []
    actor_losses = []
    
    print(f"\n{'='*60}")
    print(f"開始訓練 {agent_type} Agent")
    print(f"總回合數: {total_episodes}, 暖身步數: {warmup_steps}")
    print(f"儲存目錄: {save_dir}")
    print(f"{'='*60}\n")
    
    # ==================== 暖身階段（隨機探索） ====================
    print("Phase 1: 暖身階段 - 隨機探索收集初始經驗...")
    obs = env_wrapper.reset()
    for step in range(warmup_steps):
        # 隨機動作
        action = np.random.uniform(-env_wrapper.max_action, 
                                  env_wrapper.max_action, 
                                  size=env_wrapper.action)
        next_obs, reward, done, _ = env_wrapper.step(action)
        
        # 儲存經驗
        agent.store_transition(obs, action, reward, next_obs, done)
        
        obs = next_obs if not done else env_wrapper.reset()
        
        if (step + 1) % 100 == 0:
            print(f"  暖身進度: {step + 1}/{warmup_steps}")
    
    print(f"✓ 暖身完成！緩衝區大小: {len(agent.replay_buffer)}\n")
    
    # ==================== 訓練階段 ====================
    print("Phase 2: 訓練階段")
    best_reward = -np.inf
    
    for episode in range(1, total_episodes + 1):
        # 訓練一個 episode
        total_reward, avg_critic_loss, avg_actor_loss = experiment.train_episode()
        
        # 記錄數據
        train_rewards.append(total_reward)
        critic_losses.append(avg_critic_loss)
        actor_losses.append(avg_actor_loss)
        
        # 每隔一段時間進行測試評估
        if episode % test_interval == 0:
            agent.set_training_mode(False)  # 切換到測試模式
            test_reward = experiment.run_episode(render=False)
            test_rewards.append(test_reward)
            agent.set_training_mode(True)   # 切換回訓練模式
            
            print(f"Episode {episode:3d} | "
                  f"Train: {total_reward:7.2f} | "
                  f"Test: {test_reward:7.2f} | "
                  f"Critic Loss: {avg_critic_loss:6.3f} | "
                  f"Actor Loss: {avg_actor_loss:6.3f}")
            
            # 儲存最佳模型
            if test_reward > best_reward:
                best_reward = test_reward
                agent.save(f"{save_dir}/best_model.pth")
                print(f"  → 🏆 新最佳模型！獎勵: {best_reward:.2f}")
        
        # 定期儲存檢查點
        if episode % save_interval == 0:
            agent.save(f"{save_dir}/checkpoint_ep{episode}.pth")
    
    # 儲存最終模型
    agent.save(f"{save_dir}/final_model.pth")
    
    # ==================== 繪製訓練曲線 ====================
    print("\n繪製訓練曲線...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 訓練獎勵
    axes[0, 0].plot(train_rewards, alpha=0.6, label='Episode Reward')
    axes[0, 0].plot(np.convolve(train_rewards, np.ones(10)/10, mode='valid'), 
                    'r-', linewidth=2, label='Moving Average (10)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Training Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 測試獎勵
    test_episodes = list(range(test_interval, total_episodes + 1, test_interval))
    axes[0, 1].plot(test_episodes, test_rewards, 'go-', linewidth=2, markersize=4)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Total Reward')
    axes[0, 1].set_title('Test Rewards')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Critic Loss
    axes[1, 0].plot(critic_losses, alpha=0.8)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Critic Loss')
    axes[1, 0].set_title('Critic Loss Curve')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Actor Loss
    axes[1, 1].plot(actor_losses, alpha=0.8, color='orange')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Actor Loss')
    axes[1, 1].set_title('Actor Loss Curve')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_curves.png", dpi=150)
    print(f"✓ 訓練曲線已儲存至 {save_dir}/training_curves.png")
    
    # 儲存訓練數據
    np.savez(f"{save_dir}/training_data.npz",
             train_rewards=train_rewards,
             test_rewards=test_rewards,
             critic_losses=critic_losses,
             actor_losses=actor_losses)
    print(f"✓ 訓練數據已儲存至 {save_dir}/training_data.npz")
    
    print(f"\n{'='*60}")
    print(f"訓練完成！")
    print(f"最佳測試獎勵: {best_reward:.2f}")
    print(f"最終訓練獎勵: {train_rewards[-1]:.2f}")
    print(f"模型已儲存至: {save_dir}")
    print(f"{'='*60}\n")
    
    experiment.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Pendulum Agent')
    parser.add_argument('--agent', type=str, default='TD3', choices=['DDPG', 'TD3'],
                       help='Agent type (default: TD3)')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Total training episodes (default: 100)')
    parser.add_argument('--warmup', type=int, default=1000,
                       help='Warmup steps for random exploration (default: 1000)')
    
    args = parser.parse_args()
    
    train_agent(
        agent_type=args.agent,
        total_episodes=args.episodes,
        warmup_steps=args.warmup
    )
