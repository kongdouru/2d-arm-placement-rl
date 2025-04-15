from environments.custom_reacher_env import CustomReacherEnv
import numpy as np
import time

env = CustomReacherEnv(render_mode="human")
obs, _ = env.reset()

for _ in range(200):
    action = np.array([0.2, 0.2])  # 只控制 joint0、joint1
    obs, reward, done, _, _ = env.step(action)
    print("Reward:", reward)
    time.sleep(0.05)

env.close()
