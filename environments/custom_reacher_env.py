# environments/custom_reacher_env.py
import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.envs.mujoco import MujocoEnv

class CustomReacherEnv(MujocoEnv, gym.utils.EzPickle):
    def __init__(self, render_mode=None, only_first_phase=True):
        self.only_first_phase = only_first_phase  # 如果 True，第一阶段完成即结束
        xml_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../assets/reacher_fixed_target.xml")
        )
        self.frame_skip = 2

        # 观测：qpos(2), qvel(2), fingertip(2), target1(2), target2(2)
        self.observation_space = Box(-np.inf, np.inf, (10,), np.float32)
        self.action_space      = Box(-0.5, 0.5,   (2,), np.float32)

        gym.utils.EzPickle.__init__(self)
        MujocoEnv.__init__(
            self,
            model_path=xml_path,
            frame_skip=self.frame_skip,
            observation_space=self.observation_space,
            render_mode=render_mode,
        )

        # 保存初始状态
        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

        # 分阶段：先碰 target1 再碰 target2
        self.phase = 0
        self.current_step = 0
        self.max_steps = 1000

        # 预先查 joint / site id
        self.j_target1_x = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target1_x")
        self.j_target1_y = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target1_y")
        self.j_target2_x = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target2_x")
        self.j_target2_y = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target2_y")

        self.s_fingertip = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,  "fingertip")
        self.s_target1   = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target1_site")
        self.s_target2   = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target2_site")

        # 为外部脚本 alias
        self.fingertip_id = self.s_fingertip
        self.target1_id   = self.s_target1
        self.target2_id   = self.s_target2

    def get_obs(self):
        qpos      = self.data.qpos[:2].copy()
        qvel      = self.data.qvel[:2].copy()
        fingertip = self.data.site_xpos[self.s_fingertip][:2].copy()
        t1        = self.data.site_xpos[self.s_target1 ][:2].copy()
        t2        = self.data.site_xpos[self.s_target2 ][:2].copy()
        return np.concatenate([qpos, qvel, fingertip, t1, t2])

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action
        self.do_simulation(action, self.frame_skip)
        obs = self.get_obs()
        self.current_step += 1

        fingertip = obs[4:6]
        done = False

        # 阶段1：撞红球
        if self.phase == 0:
            dist = np.linalg.norm(fingertip - obs[6:8])
            reward = -dist
            if dist < 0.02:
                reward += 10.0
                if self.only_first_phase:
                    done = True
                    return obs, reward, done, False, {}
                self.phase = 1
        else:
            # 阶段2：撞绿球
            dist = np.linalg.norm(fingertip - obs[8:10])
            reward = -dist
            if dist < 0.02:
                reward += 10.0

        # 平滑动作惩罚
        reward -= 0.01 * np.sum(np.square(action))

        # 超时惩罚
        if self.current_step >= self.max_steps:
            reward -= 5.0
            done = True

        # 如果 both 阶段，第二阶段达成终止
        if not self.only_first_phase and self.phase == 1 and dist < 0.02:
            done = True

        return obs, reward, done, False, {}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(-0.1, 0.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(-0.1, 0.1, size=self.model.nv)
        low, high = -0.27, 0.27
        qpos[self.j_target1_x] = self.np_random.uniform(low, high)
        qpos[self.j_target1_y] = self.np_random.uniform(low, high)
        qpos[self.j_target2_x] = self.np_random.uniform(low, high)
        qpos[self.j_target2_y] = self.np_random.uniform(low, high)
        self.set_state(qpos, qvel)
        self.phase = 0
        self.current_step = 0
        return self.get_obs()