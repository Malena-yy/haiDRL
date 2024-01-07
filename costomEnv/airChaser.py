import sys
import numpy as np
import time
import math
import copy
import gym
from gym import spaces
from gym.utils import seeding
from pprint import pprint

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

'''
本游戏环境为在空中的追踪者chaser在一个固定持续时间T内追踪射击地面目标对象target的过程，其中
· 地图大小为400*400
· 追踪者chaser数量为1，移动高度为height，最大移动速度为max_velocity，角度为angle，有效射击距离为range，以 v in [0,1], theta in [-1,1]为动作
· 目标对象target位置随机生成，任一时刻只有一个目标对象存在，当chaser完成对一个目标对象的射击后，随机生成一个新的目标对象
'''


# 定义chaser
class Chaser(tk.Tk):
    def __init__(self, render_mode):
        super().__init__()
        assert render_mode in ["human", "rgb_array"]
        self.render_mode = render_mode
        self.X_min = 0
        self.Y_min = 0
        self.X_max = 400
        self.Y_max = 400  # 地图边界
        self.n_chaser = 1  # 追击者数量
        self.n_target = 1  # 目标对象数量
        self.max_episode_steps = 400  # 一个周期为 400 step
        self.max_velocity = 20  # chaser最大速度 20m/s
        self.height = 10.  # chaser飞行高度 10m
        self.range = 15.  # 有效射击距离 15m
        self.over_edge = 0  # 飞越边界
        #   以 v in [0,1],theta in [-1,1]为动作
        self.action_space = spaces.Box(low=np.array([0., -1.]), high=np.array([1., 1.]),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([-1., -1., 0., 0.]), high=np.array([1., 1., 1., 1.]),
                                            dtype=np.float32)
        self.state_dim = 4  # 状态空间为chaser与target相对位置，chaser位置
        self.state = np.zeros(self.state_dim)
        self.chaser_container = []
        self.target_container = []
        self.chaser_location = np.random.randint(self.X_min, self.X_max, size=[self.n_chaser, 2])  # chaser位置
        self.target_location = np.random.randint(self.X_min, self.X_max, size=[self.n_target, 2])  # target位置
        self.build_maze()
        self.curr_step = 0  # 当前周期step

    # 创建地图
    def build_maze(self):
        self.title('MAP')
        self.geometry('{0}x{1}'.format(self.X_max, self.Y_max))  # Tkinter 的几何形状
        # 创建画布Canvas,白色背景,宽self.X_max,高self.Y_max
        self.canvas = tk.Canvas(self, bg='white', width=self.X_max, height=self.Y_max)
        self.canvas.pack()

    def set_random_seed(self, seed=None):
        np.random.seed(seed)
        seeding.np_random(seed)

    # 环境重置
    def reset(self, seed=None):
        self.set_random_seed(seed)

        if self.chaser_container:
            for i in range(self.n_chaser):
                self.canvas.delete(self.chaser_container[i])
            self.chaser_container = []

            for i in range(self.n_target):
                self.canvas.delete(self.target_container[i])
            self.target_container = []

        self.chaser_location = np.random.randint(self.X_min, self.X_max, size=[self.n_chaser, 2])  # chaser位置
        self.target_location = np.random.randint(self.X_min, self.X_max, size=[self.n_target, 2])  # target位置

        # 创建方形chaser对象
        for i in range(self.n_chaser):
            chaser = self.canvas.create_oval(
                self.chaser_location[i][0] - self.range, self.chaser_location[i][1] - self.range,
                self.chaser_location[i][0] + self.range, self.chaser_location[i][1] + self.range,
                fill='yellow')
            self.chaser_container.append(chaser)

        # 创建椭圆target对象
        for i in range(self.n_target):
            target = self.canvas.create_rectangle(
                self.target_location[i][0] - 5, self.target_location[i][1] - 5,
                self.target_location[i][0] + 5, self.target_location[i][1] + 5,
                fill='red')
            self.target_container.append(target)

        self.over_edge = 0
        self.curr_step = 0

        self.state = np.concatenate(((self.target_location - self.chaser_location).flatten() / self.X_max,
                                     self.chaser_location.flatten() / self.X_max))
        if self.render_mode == "human":
            self.render()
        return self.state, {}

    # 传入当前状态和输入动作输出下一个状态和奖励
    def step(self, action):
        detX = action[0] * self.max_velocity * math.cos(action[1] * math.pi)
        detY = action[0] * self.max_velocity * math.sin(action[1] * math.pi)
        state_ = np.zeros(self.state_dim)
        xy_ = copy.deepcopy(self.chaser_location)
        for i in range(self.n_chaser):  # chaser位置更新
            xy_[i][0] += detX
            xy_[i][1] += detY
            # 当无人机更新后的位置超出地图范围时
            if self.X_min <= xy_[i][0] <= self.X_max:
                if self.Y_min <= xy_[i][1] <= self.Y_max:
                    self.over_edge = 0.
                else:
                    xy_[i][0] -= detX
                    xy_[i][1] -= detY
                    self.over_edge += 1.
            else:
                xy_[i][0] -= detX
                xy_[i][1] -= detY
                self.over_edge += 1.
            self.canvas.move(self.chaser_container[i], xy_[i][0] - self.chaser_location[i][0],
                             xy_[i][1] - self.chaser_location[i][1])

        # chaser位置更新后，state更新
        self.chaser_location = xy_
        # 状态空间归一化
        state_[:2] = (self.target_location - self.chaser_location).flatten() / self.X_max
        state_[2:4] = self.chaser_location.flatten() / self.X_max
        self.state = state_
        terminated = False
        truncated = False
        # 奖励的定义——接近target/不要飞越地图边界
        reward = -(abs(state_[0]) + abs(state_[1])) * 10 - self.over_edge
        for i in range(self.n_target):
            dis = math.sqrt(
                pow(self.target_location[i][0] - self.chaser_location[0][0], 2) + pow(
                    self.target_location[i][1] - self.chaser_location[0][1], 2) + pow(
                    self.height, 2))  # 原始距离
            if dis <= self.range:
                terminated = True
                reward += 100
                state_ = self.reset_target()
                break
        self.curr_step += 1
        if self.curr_step == self.max_episode_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()
        return state_, reward, terminated, truncated, {}

    # 重选目标用户
    def reset_target(self):
        for i in range(self.n_target):
            self.canvas.delete(self.target_container[i])
        self.target_container = []

        self.target_location = np.random.randint(self.X_min, self.X_max, size=[self.n_target, 2])  # target位置
        self.state[:2] = (self.target_location - self.chaser_location).flatten() / self.X_max

        # 创建椭圆target对象
        for i in range(self.n_target):
            target = self.canvas.create_rectangle(
                self.target_location[i][0] - 5, self.target_location[i][1] - 5,
                self.target_location[i][0] + 5, self.target_location[i][1] + 5,
                fill='red')
            self.target_container.append(target)
        if self.render_mode == "human":
            self.render()
        return self.state

    # 调用Tkinter的update方法, 0.01秒去走一步。
    def render(self, t=0.01):
        time.sleep(t)
        self.update()

    def sample_action(self):
        v = np.random.rand()
        theta = -1 + 2 * np.random.rand()
        return [v, theta]

    def close(self):
        self.destroy()


def get_env_args(env_name, render=False) -> dict:
    '''
    参考：https://github.com/AI4Finance-Foundation/ElegantRL/blob/master/elegantrl/train/config.py
    '''
    if render:
        env = Chaser(render_mode="human")
        state, info = env.reset(seed=2024)
        for _ in range(100):
            action = env.action_space.sample()
            state, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                state, info = env.reset(seed=2024)
        env.destroy()
    else:
        env = Chaser(render_mode="rgb_array")
    max_step = env.max_episode_steps
    state_shape = env.observation_space.shape
    state_dim = state_shape[0] if len(state_shape) == 1 else state_shape  # sometimes state_dim is a list
    action_space = env.action_space
    if_discrete = isinstance(action_space, gym.spaces.Discrete)
    if if_discrete:  # make sure it is discrete action space
        action_dim = getattr(action_space, 'n')
    elif isinstance(action_space, gym.spaces.Box):  # make sure it is continuous action space
        action_dim = action_space.shape[0]
    else:
        raise RuntimeError('\n| Error in get_gym_env_info(). Please set these value manually:'
                           '\n  `state_dim=int; action_dim=int; if_discrete=bool;`'
                           '\n  And keep action_space in range (-1, 1).')
    env_args = {'env_name': env_name,
                'max_step': max_step,
                'state_dim': state_dim,
                'action_space': env.action_space,
                'action_dim': action_dim,
                'if_discrete': if_discrete, }
    print("============= env info ============")
    pprint(env_args)
    return env_args


if __name__ == '__main__':
    get_env_args(env_name="airChaser", render=True)
