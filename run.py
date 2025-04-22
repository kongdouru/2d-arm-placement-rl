import torch
import numpy as np
import time
from environments.custom_reacher_env import CustomReacherEnv
from td3 import Actor

# 初始化环境时设置为 human 模式，这样会显示渲染窗口
env = CustomReacherEnv(render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# 加载训练好的 Actor 模型（请修改路径，确保文件存在）
actor = Actor(state_dim, action_dim, max_action).cuda()
actor.load_state_dict(torch.load("results_td3/best_actor.pth", map_location="cuda"))
actor.eval()

n_episodes = 5
for ep in range(n_episodes):
    state, _ = env.reset()
    ep_reward = 0
    done = False
    while not done:
        # 得到 action：训练时的策略输出
        state_tensor = torch.FloatTensor(state.reshape(1, -1)).cuda()
        action = actor(state_tensor).cpu().data.numpy().flatten()
        state, reward, done, truncated, _ = env.step(action)
        ep_reward += reward
        env.render()  # 更新渲染窗口
        time.sleep(0.1)  # 控制渲染速度

        if done or truncated:
            break
    print(f"Episode {ep}: Total Reward = {ep_reward:.2f}")
    
env.close()
