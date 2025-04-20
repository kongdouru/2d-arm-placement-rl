from environments.custom_reacher_env import CustomReacherEnv
import numpy as np
import time

env = CustomReacherEnv(render_mode="human")
obs, _ = env.reset()

for i in range(300):
    action = np.random.uniform(low=-0.5, high=0.5, size=(2,))
    obs, reward, done, _, _ = env.step(action)
    env.render()
    print(f"Step {i}: Reward = {reward:.3f}, Obs = {obs}")
    time.sleep(0.05)

env.close()
