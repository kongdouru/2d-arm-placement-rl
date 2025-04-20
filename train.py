
# train.py
import numpy as np
import torch
import os
from environments.custom_reacher_env import CustomReacherEnv
from td3 import TD3
from replay_buffer import ReplayBuffer

env = CustomReacherEnv(render_mode=None, only_first_phase=False)
state_dim  = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

agent        = TD3(state_dim, action_dim, max_action)
replay_buffer= ReplayBuffer(state_dim, action_dim)

# 超参数
max_timesteps   = 50_000
start_timesteps = 2_000
batch_size      = 256
expl_noise      = 0.1
eval_freq       = 5_000
log_freq        = 1_000
save_dir        = "results"
os.makedirs(save_dir, exist_ok=True)
best_eval_reward= -np.inf

state, _ = env.reset()
episode_reward=0.0
episode_timesteps=0
episode_num=0

print("开始训练...")
for t in range(1, max_timesteps+1):
    episode_timesteps += 1
    if t < start_timesteps:
        action = env.action_space.sample()
    else:
        action = agent.select_action(np.array(state))
        action += np.random.normal(0, expl_noise, size=action_dim)
        action = np.clip(action, env.action_space.low, env.action_space.high)

    next_state, reward, done, truncated, _ = env.step(action)
    replay_buffer.add(state, action, next_state, reward, float(done or truncated))
    state = next_state
    episode_reward += reward

    if done or truncated:
        print(f"Episode {episode_num} | Steps: {episode_timesteps} | Reward: {episode_reward:.2f}")
        state, _ = env.reset()
        episode_reward=0.0
        episode_timesteps=0
        episode_num+=1

    if t % log_freq == 0:
        ft = env.data.site_xpos[env.fingertip_id][:2]
        t1 = env.data.site_xpos[env.target1_id][:2]
        t2 = env.data.site_xpos[env.target2_id][:2]
        d1 = np.linalg.norm(ft - t1)
        d2 = np.linalg.norm(ft - t2)
        print(f"Time {t} | EpReward {episode_reward:.2f} | d1={d1:.3f}, d2={d2:.3f}")

    if t >= start_timesteps:
        agent.train(replay_buffer, batch_size)

    if t % eval_freq == 0:
        # 简单 eval: avg reward
        avg_r = 0.0
        for _ in range(5):
            s,_ = env.reset()
            done=False
            while not done:
                a = agent.select_action(np.array(s))
                s, r, done, *_ = env.step(a)
                avg_r += r
        avg_r /=5
        print(f"Eval {t}: avg_r={avg_r:.2f}")
        if avg_r>best_eval_reward:
            best_eval_reward=avg_r
            print("新最优，保存模型")
            torch.save(agent.actor.state_dict(), f"{save_dir}/best_actor.pth")
            torch.save(agent.critic.state_dict(), f"{save_dir}/best_critic.pth")
        torch.save(agent.actor.state_dict(), f"{save_dir}/actor_{t}.pth")
        torch.save(agent.critic.state_dict(), f"{save_dir}/critic_{t}.pth")
