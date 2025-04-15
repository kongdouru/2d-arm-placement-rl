import numpy as np
import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import mujoco
import os

class CustomReacherEnv(MujocoEnv, gym.utils.EzPickle):
    def __init__(self, render_mode=None):
        xml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../assets/reacher_fixed_target.xml"))
        self.frame_skip = 2

        # 动作空间: [joint0, joint1, object_x, object_y]
        self.action_space = Box(low=-0.5, high=0.5, shape=(4,), dtype=np.float32)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)

        gym.utils.EzPickle.__init__(self)
        MujocoEnv.__init__(self, model_path=xml_path, frame_skip=self.frame_skip,
                           observation_space=self.observation_space, render_mode=render_mode)

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

        # 记录 object 的 joint id
        self.object_x_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obj_x")
        self.object_y_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obj_y")


    def get_obs(self):
        qpos = self.data.qpos[:4].copy()
        qvel = self.data.qvel[:4].copy()

        fingertip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "fingertip")
        object_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "object")
        target_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target")

        fingertip = self.data.site_xpos[fingertip_id][:2].copy()
        object_pos = self.data.site_xpos[object_id][:2].copy()
        target_pos = self.data.site_xpos[target_id][:2].copy()

        return np.concatenate([qpos, qvel, fingertip, object_pos, target_pos])

    def step(self, action):
        # 动作拆分
        joint_action = action[:2]
        object_delta = action[2:]

        # 控制机械臂关节
        self.data.ctrl[0] = np.clip(joint_action[0], -0.5, 0.5)
        self.data.ctrl[1] = np.clip(joint_action[1], -0.5, 0.5)

        # 移动物体的 joint 位置
        self.data.qpos[self.object_x_id] = np.clip(self.data.qpos[self.object_x_id] + object_delta[0], -0.3, 0.3)
        self.data.qpos[self.object_y_id] = np.clip(self.data.qpos[self.object_y_id] + object_delta[1], -0.3, 0.3)

        self.do_simulation(self.data.ctrl, self.frame_skip)
        obs = self.get_obs()

        dist = np.linalg.norm(obs[10:12] - obs[12:14])  # object 到 target 的距离
        reward = -dist + 0.01 * -np.square(action).sum()
        done = dist < 0.02

        return obs, reward, done, False, {}

    def reset_model(self):
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        self.set_state(qpos, qvel)
        return self.get_obs()
