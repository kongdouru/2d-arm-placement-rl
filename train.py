import gymnasium as gym
import numpy as np
import torch
import os
import time
from td3 import TD3
from replay_buffer import ReplayBuffer

# ✅ 环境参数
env_name = "Reacher-v5"
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# ✅ 初始化 agent 和 buffer
agent = TD3(state_dim, action_dim, max_action)
replay_buffer = ReplayBuffer(state_dim, action_dim)

# ✅ 超参数
max_timesteps = 100_000
start_timesteps = 5_000
batch_size = 256
expl_noise = 0.1  # 高斯探索噪声
eval_freq = 5000  # 每多少步评估一次
save_model = True
save_dir = "results"

os.makedirs(save_dir, exist_ok=True)

# ✅ 训练循环
state, _ = env.reset()
episode_reward = 0
episode_timesteps = 0
episode_num = 0

for t in range(1, max_timesteps + 1):
    episode_timesteps += 1

    # 初始阶段随机动作探索
    if t < start_timesteps:
        action = env.action_space.sample()
    else:
        action = agent.select_action(np.array(state))
        action += np.random.normal(0, expl_noise, size=action_dim)
        action = action.clip(env.action_space.low, env.action_space.high)

    next_state, reward, done, truncated, _ = env.step(action)
    done_bool = float(done or truncated)

    replay_buffer.add(state, action, next_state, reward, done_bool)

    state = next_state
    episode_reward += reward

    # 如果 episode 结束
    if done or truncated:
        print(f"Episode {episode_num} | Steps: {episode_timesteps} | Reward: {episode_reward:.2f}")
        state, _ = env.reset()
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

    # 开始训练
    if t >= start_timesteps:
        agent.train(replay_buffer, batch_size)

    # 定期保存模型
    if save_model and t % eval_freq == 0:
        torch.save(agent.actor.state_dict(), f"{save_dir}/actor_{t}.pth")
        torch.save(agent.critic.state_dict(), f"{save_dir}/critic_{t}.pth")
