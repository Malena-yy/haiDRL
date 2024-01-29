import argparse
import time
import numpy as np
import sys
import os
from collections import namedtuple
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import gym
import matplotlib.pyplot as plt
from pathlib import Path
from pprint import pprint
import pandas as pd
import ast
import wandb

base_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_dir)

font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 16,
         }

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', default="Pendulum-v1", type=str)
parser.add_argument('--algo_name', default="iql", type=str)
parser.add_argument('--offline_data_path', default="./offline_data/Pendulum-v1_sac.csv", type=str)
parser.add_argument('--max_train_step', default=80000, type=int)
parser.add_argument('--eval_episode_cnt', default=5, type=int)
parser.add_argument('--eval_per_step', default=200, type=int)
parser.add_argument('--hidden_dim', default=256, type=int)
parser.add_argument('--buffer_capacity', default=int(1e5), type=int)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--lr_actor', default=0.0001, type=float)
parser.add_argument('--lr_critic', default=0.0001, type=float)
parser.add_argument('--lr_value', default=0.0001, type=float)
parser.add_argument('--expectile', default=0.7, type=float)
parser.add_argument('--temperature', default=1.0, type=float)
parser.add_argument('--gamma', default=0.99, type=float)
parser.add_argument('--replace_tau', default=0.005, type=float)
parser.add_argument('--random_seed', default=2024, type=int)
parser.add_argument("--save_interval", default=100, type=int)
# eval
parser.add_argument("--load_run", default=1, type=int, help="load the model of run(load_run)")
parser.add_argument("--load_episode", default=300, type=int, help="load the model trained for `load_episode` episode")
parser.add_argument('--eval_episode', default=30, type=int, help="the number of episode to eval the model")
# logging choice
parser.add_argument("--use_tensorboard", default=0, type=int, help="1 as true, 0 as false")
parser.add_argument("--use_wandb", default=0, type=int, help="1 as true, 0 as false")

device = 'cpu'
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPSILON = 1e-6


# ====================================  UTILS =======================================
def make_save_path(env_name, algo_name, is_make_dir=True):
    base_path = Path(__file__).resolve().parent
    model_path = base_path / Path("./result") / env_name.replace("-", "_") / algo_name

    if not model_path.exists():
        curr_run = "run1"
    else:
        exst_run_nums = [
            int(str(folder.name).split("run")[1])
            for folder in model_path.iterdir()
            if str(folder.name).startswith("run")
        ]
        if len(exst_run_nums) == 0:
            curr_run = "run1"
        else:
            curr_run = "run%i" % (max(exst_run_nums) + 1)
    save_path = model_path / curr_run
    if is_make_dir:
        os.makedirs(save_path, exist_ok=True)
    return save_path, curr_run


def save_config(args, save_path):
    import yaml
    file = open(os.path.join(str(save_path), 'config.yaml'), mode='w', encoding='utf-8')
    yaml.dump(vars(args), file)
    file.close()


# ========================================= ReplayBuffer ======================================
class ReplayBuffer:
    """
    强化学习储存训练数据的训练池
    """

    def __init__(self):
        self.Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])
        self.buffer = []

    def add(self, state, action, reward, next_state, done):
        """保存一个经验元组"""
        experience = self.Transition(state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        """随机抽取一批经验"""
        tem = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*tem)
        states, actions, rewards, next_states, dones = np.stack(states), np.stack(actions), np.stack(rewards), np.stack(
            next_states), np.stack(dones)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """返回当前缓冲区的大小"""
        return len(self.buffer)


# ====================================  NET CONSTRUCTION ===============================
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.uniform_(m.bias, a=-0.003, b=0.003)


class Actor(nn.Module):
    def __init__(self, state_dim, action_space, hidden_dim=128):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_space.shape[0]
        self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
        self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)

        self.linear1 = nn.Linear(self.state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.apply(weights_init_)
        self.mean_linear = nn.Linear(hidden_dim, self.action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, self.action_dim)

    def forward(self, state):
        x = torch.relu(self.linear1(state))
        x = torch.relu(self.linear2(x))
        mu = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mu, log_std

    def evaluate(self, obs):
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mu, std)
        action = dist.rsample()
        return action, dist

    def get_action(self, obs):
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mu, std)
        sample_action = dist.rsample()
        sample_action = torch.tanh(sample_action) * self.action_scale + self.action_bias
        return sample_action

    def get_det_action(self, obs):
        mu, _ = self.forward(obs)
        mean_action = torch.tanh(mu) * self.action_scale + self.action_bias
        return mean_action


class QNET(nn.Module):
    def __init__(self, state_dim, action_space, hidden_dim=128):
        super(QNET, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_space.shape[0]

        # q1
        self.linear11 = nn.Linear(self.state_dim + self.action_dim, hidden_dim)
        self.linear12 = nn.Linear(hidden_dim, hidden_dim)
        self.linear13 = nn.Linear(hidden_dim, 1)

        # q2
        self.linear21 = nn.Linear(self.state_dim + self.action_dim, hidden_dim)
        self.linear22 = nn.Linear(hidden_dim, hidden_dim)
        self.linear23 = nn.Linear(hidden_dim, 1)
        self.apply(weights_init_)

    def forward(self, state, action):
        x1 = torch.relu(self.linear11(torch.cat([state, action], 1)))
        x1 = torch.relu(self.linear12(x1))
        q1 = self.linear13(x1)

        x2 = torch.relu(self.linear21(torch.cat([state, action], 1)))
        x2 = torch.relu(self.linear22(x2))
        q2 = self.linear23(x2)
        return q1, q2


class VNET(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super(VNET, self).__init__()
        self.state_dim = state_dim

        self.linear1 = nn.Linear(self.state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = torch.relu(self.linear1(state))
        x = torch.relu(self.linear2(x))
        v = self.linear3(x)

        return v


# ====================================  ALGO CONSTRUCTION ===============================
class IQL:
    def __init__(self, args, state_dim=None, action_space=None, replay_buffer=None, writer=None):
        super(IQL, self).__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.action_space = action_space
        self.state_dim = state_dim
        self.action_dim = action_space.shape[0]
        self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
        self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)
        self.lr_critic = args.lr_critic
        self.lr_value = args.lr_value
        self.lr_actor = args.lr_actor
        self.expectile = args.expectile
        self.temperature = args.temperature

        self.replay_buffer = replay_buffer
        self.replace_tau = args.replace_tau

        self.critic_net = QNET(state_dim, action_space, args.hidden_dim)
        self.target_critic_net = QNET(state_dim, action_space, args.hidden_dim)
        self.target_critic_net.load_state_dict(self.critic_net.state_dict())
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), lr=self.lr_critic)

        self.value_net = VNET(state_dim, args.hidden_dim)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.lr_value)

        self.actor_net = Actor(state_dim, action_space, args.hidden_dim)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=self.lr_actor)

        self.training_step = 0
        self.writer = writer

    def l2_loss(self, diff, expectile=0.8):
        weight = torch.where(diff > 0, expectile, (1 - expectile))
        return weight * (diff ** 2)

    def select_action(self, state, train=True):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        if not train:
            action = self.actor_net.get_det_action(state)
        else:
            action = self.actor_net.get_action(state)
        return action.detach().cpu().numpy()[0]

    def update(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.tensor(states, dtype=torch.float).to(device)
        actions = torch.tensor(actions, dtype=torch.float).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float).to(device)
        dones = torch.tensor(dones, dtype=torch.float).to(device).unsqueeze(1)
        # update value net
        with torch.no_grad():
            q1, q2 = self.target_critic_net(states, actions)
            min_Q = torch.min(q1, q2)
        value = self.value_net(states)
        value_loss = self.l2_loss(min_Q - value, self.expectile).mean()
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # update actor network
        with torch.no_grad():
            v = self.value_net(states)
            q1, q2 = self.target_critic_net(states, actions)
            min_q_pi = torch.min(q1, q2)

        '''
            where temperature in [0;1) is an inverse temperature.
            For smaller hyperparameter values, the objective behaves similarly to behavioral cloning, 
            while for larger values, it attempts to recover the  maximum of the Q-function.
        '''
        exp_a = torch.exp(min_q_pi - v) * self.temperature
        exp_a = torch.min(exp_a, torch.FloatTensor([100.0]))
        _, dist = self.actor_net.evaluate(states)
        log_probs = dist.log_prob(actions)
        actor_loss = -(exp_a * log_probs).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # update critic network
        with torch.no_grad():
            next_v = self.value_net(next_states)
            q_target = rewards + (self.gamma * (1 - dones) * next_v)
        q1, q2 = self.critic_net(states, actions)
        q1_loss = 0.5 * F.mse_loss(q1, q_target)
        q2_loss = 0.5 * F.mse_loss(q2, q_target)
        q_loss = torch.mean(q1_loss + q2_loss)
        self.critic_net_optimizer.zero_grad()
        q_loss.backward()
        # nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
        self.critic_net_optimizer.step()

        for target_param, source_param in zip(self.target_critic_net.parameters(), self.critic_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.replace_tau) + source_param.data * self.replace_tau)

        self.training_step += 1
        if self.args.use_wandb:
            wandb.log({'loss/critic loss': q_loss.item(),
                       "loss/actor loss": actor_loss.item(),
                       "loss/value loss": value_loss.item(),
                       "loss/adv":(min_q_pi - v).items()
                       })
        if self.args.use_tensorboard:
            self.writer.add_scalar('loss/critic loss', q_loss.item(), self.training_step)
            self.writer.add_scalar('loss/actor loss', actor_loss.item(), self.training_step)
            self.writer.add_scalar('loss/value loss', value_loss.item(), self.training_step)

        return {'loss/critic loss': q_loss.item(),
                "loss/actor loss": actor_loss.item(),
                "loss/value loss": value_loss.item(),
                "loss/adv": (min_q_pi - v).items()
                }

    def save(self, save_path, episode):
        base_path = os.path.join(save_path, 'trained_model')
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        actor_model_path = os.path.join(base_path, "actor_" + str(episode) + ".pth")
        torch.save(self.actor_net.state_dict(), actor_model_path)
        critic_model_path = os.path.join(base_path, "critic_" + str(episode) + ".pth")
        torch.save(self.critic_net.state_dict(), critic_model_path)
        value_model_path = os.path.join(base_path, "value_" + str(episode) + ".pth")
        torch.save(self.value_net.state_dict(), value_model_path)
        print("success saving model in {}".format(str(base_path)))

    def load(self, load_path, episode):
        print(f'\nBegin to load model: ')
        run_path = os.path.join(load_path, 'trained_model')
        actor_model_path = os.path.join(run_path, "actor_" + str(episode) + ".pth")
        critic_model_path = os.path.join(run_path, "critic_" + str(episode) + ".pth")
        value_model_path = os.path.join(run_path, "value_" + str(episode) + ".pth")
        print(f'actor_model_path: {actor_model_path}')
        print(f'critic_model_path: {critic_model_path}')
        print(f'value_model_path: {value_model_path}')

        if os.path.exists(critic_model_path) and os.path.exists(actor_model_path) and os.path.exists(value_model_path):
            actor = torch.load(actor_model_path, map_location=device)
            critic = torch.load(critic_model_path, map_location=device)
            value = torch.load(value_model_path, map_location=device)
            self.actor_net.load_state_dict(actor)
            self.critic_net.load_state_dict(critic)
            self.target_critic_net.load_state_dict(self.critic_net.state_dict())
            self.value_net.load_state_dict(value)
            print("Model loaded!")
        else:
            sys.exit(f'Model not founded!')


# ====================================  EXECUTE FUNCTION  ===============================
def get_env_args(env_name, render=False) -> dict:
    '''
    参考：https://github.com/AI4Finance-Foundation/ElegantRL/blob/master/elegantrl/train/config.py
    '''
    if render:
        env = gym.make(env_name, render_mode='human')
        state, info = env.reset(seed=args.seed)
        for _ in range(10):
            action = env.action_space.sample()
            state, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                state, info = env.reset(seed=args.seed)
    else:
        env = gym.make(env_name, render_mode="rgb_array")
    max_step = getattr(env, '_max_episode_steps', 12345)
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
    env.close()
    return env_args


def train_with_offline_data(args):
    begin_time = time.time()
    replay_buffer = ReplayBuffer()
    print("offline_data_path:{}".format(args.offline_data_path))
    data_loader = pd.read_csv(args.offline_data_path)
    print("offline_data_index:{}".format(data_loader.index))
    print("offline_data_shape:{}".format(data_loader.shape))
    data_loader["state"] = data_loader["state"].apply(ast.literal_eval)
    data_loader["action"] = data_loader["action"].apply(ast.literal_eval)
    data_loader["next_state"] = data_loader["next_state"].apply(ast.literal_eval)
    for index, row in data_loader.iterrows():
        replay_buffer.add(row["state"], row["action"], row["reward"], row["next_state"], row["done"])
    print("success to add {} offline_data to replay_buffer".format(replay_buffer.__len__()))
    # render gym or not
    RENDER = False
    if RENDER:
        env = gym.make(args.env_name, render_mode='human')
    else:
        env = gym.make(args.env_name, render_mode="rgb_array")
    state_dim = env.observation_space.shape[0]
    print("state_space:{}".format(env.observation_space))
    action_space = env.action_space
    print("action_space:{}".format(action_space))
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    # define save path
    save_path, curr_run = make_save_path(args.env_name, args.algo_name)
    print("run_dir:{}".format(save_path))
    if args.use_wandb:
        wandb.init(config=args,
                   project=args.env_name,
                   name=args.algo_name + "_" + curr_run + "_" + datetime.datetime.now().strftime(
                       "%Y-%m-%d_%H-%M-%S"),
                   dir=save_path,
                   job_type="training",
                   reinit=True,
                   monitor_gym=True)
    if args.use_tensorboard:
        tf_writer = SummaryWriter(os.path.join(str(save_path), "tf_log"))
    else:
        tf_writer = None
    # 创建智能体
    model = IQL(args, state_dim, action_space, replay_buffer, writer=tf_writer)

    train_step = 0
    plt_step = []
    plt_ep_reward = []
    while train_step < args.max_train_step:
        train_step += 1
        train_info = model.update()  # model training
        # eval
        if train_step != 0 and train_step % args.eval_per_step == 0:
            print(train_info)
            for episode in range(args.eval_episode_cnt):
                state, info = env.reset(seed=args.random_seed)
                step = 0
                ep_reward = 0
                while True:
                    action = model.select_action(state, True)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    step += 1
                    state = next_state
                    # 累加奖励
                    ep_reward += reward
                    if terminated or truncated:
                        eval_info = {"train_step": train_step, "eval_episode": episode, "step": step,
                                     "ep_reward": round(ep_reward, 2)}
                        print(eval_info)
                        if args.use_wandb:
                            wandb.log({'reward/ep_reward': ep_reward,
                                       "reward/ep_step": step})
                        if args.use_tensorboard:
                            tf_writer.add_scalar('reward/step', step, episode)
                            tf_writer.add_scalar('reward/ep_reward', ep_reward, episode)
                        break
                plt_step.append(step)
                plt_ep_reward.append(ep_reward)
    model.save(save_path, args.max_train_step)
    env.close()
    wandb.finish()
    # 保存模型配置，即生成config.yaml
    save_config(args, save_path)
    # plot result
    p1 = plt.figure(figsize=(16, 8))

    ax1 = p1.add_subplot(1, 2, 1)
    ax1.tick_params(labelsize=12)
    ax1.grid(linestyle='-.')
    ax2 = p1.add_subplot(1, 2, 2)
    ax2.tick_params(labelsize=12)
    ax2.grid(linestyle='-.')

    ax1.plot([i + 1 for i in range(len(plt_step))], plt_step)
    ax1.set_xlabel('Number of training episodes', font1)
    ax1.set_ylabel('steps', font1)

    label1 = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label1]

    label1 = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label1]

    ax2.plot([i + 1 for i in range(len(plt_ep_reward))], plt_ep_reward)
    ax2.set_xlabel('Number of training episodes', font1)
    ax2.set_ylabel(r'ep_reward', font1)

    label2 = ax2.get_xticklabels() + ax2.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label2]

    plt.subplots_adjust(wspace=0.3)
    fig_path = os.path.join(save_path, "figs")
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    plt.savefig(os.path.join(fig_path, "train_fig.jpg"))
    print("train figure is saved in {}".format(str(os.path.join(fig_path, "train_fig.jpg"))))
    print("训练时间:{}".format(time.time() - begin_time))


def eval_gym(args):
    args.use_wandb = 0
    args.use_tensorboard = 0
    print("env_name:{}".format(args.env_name))
    RENDER = False
    if RENDER:
        env = gym.make(args.env_name, render_mode='human')
    else:
        env = gym.make(args.env_name, render_mode="rgb_array")
    state_dim = env.observation_space.shape[0]
    print("state_space:{}".format(env.observation_space))
    action_space = env.action_space
    print("action_space:{}".format(action_space))
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    # define save path
    save_path, _ = make_save_path(args.env_name, args.algo_name, is_make_dir=False)
    print("run_dir:{}".format(save_path))
    # 创建智能体
    model = IQL(args, state_dim, action_space)
    load_path = os.path.join(os.path.dirname(save_path), "run" + str(args.load_run))
    print("load_path:{}".format(load_path))
    model.load(load_path, episode=args.load_episode)
    episode = 0
    plot_x = []
    plot_ep = []
    while episode < args.eval_episode:
        state, info = env.reset()
        episode += 1
        step = 0
        ep_reward = 0
        while True:
            action = model.select_action(state, False)
            next_state, reward, terminated, truncated, info = env.step(action)
            step += 1
            state = next_state
            ep_reward += reward
            if terminated or truncated:
                print("Episode:", episode, "; step:", step, "; Episode Return:", '%.2f' % ep_reward)
                plot_x.append(episode)
                plot_ep.append(ep_reward)
                break
    fig = plt.figure(figsize=(10, 8))

    ax1 = fig.add_subplot(1, 1, 1)
    ax1.tick_params(labelsize=12)
    ax1.grid(linestyle='-.')

    ax1.plot(plot_x, plot_ep)
    ax1.set_xlabel('Number of evaluation episodes', font1)
    ax1.set_ylabel('ep_reward', font1)
    label1 = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label1]

    # plt.subplots_adjust(wspace=0.3)
    fig_path = os.path.join(load_path, "figs")
    if not os.path.exists(load_path):
        os.makedirs(fig_path)
    plt.savefig(os.path.join(fig_path, "eval_result.jpg"))
    print("eval figure is saved in {}".format(str(os.path.join(fig_path, "eval_result.jpg"))))


if __name__ == '__main__':
    args = parser.parse_args()
    # show env information,
    # get_env_args(args.env_name, render=False)
    train_with_offline_data(args)
    # specify which model to be load and eval
    # args.load_run = 1
    # args.load_episode = 60000
    # eval_gym(args)
