import argparse
import time
import numpy as np
import sys
import os
from collections import namedtuple
import torch.optim as optim
from torch.distributions import Categorical
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
import wandb

base_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_dir)

font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 16,
         }

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', default="CartPole-v1", type=str)
parser.add_argument('--algo_name', default="discrete_sac", type=str)
parser.add_argument('--max_episode', default=300, type=int)
parser.add_argument('--max_step', default=500, type=int)
parser.add_argument('--hidden_dim', default=128, type=int)
parser.add_argument('--buffer_capacity', default=int(1e4), type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr_actor', default=0.002, type=float)
parser.add_argument('--lr_critic', default=0.002, type=float)
parser.add_argument('--lr_alpha', default=0.002, type=float)
parser.add_argument('--replace_tau', default=0.005, type=float)
parser.add_argument('--alpha', default=0.2, type=float)
parser.add_argument('--max_entropy_ratio', default=0.5, type=float)
parser.add_argument('--gamma', default=0.99, type=float)
parser.add_argument("--update_time", default=1, type=int)
parser.add_argument('--random_seed', default=2024, type=int)
parser.add_argument("--save_interval", default=100, type=int)
# eval
parser.add_argument("--load_run", default=1, type=int, help="load the model of run(load_run)")
parser.add_argument("--load_episode", default=300, type=int, help="load the model trained for `load_episode` episode")
parser.add_argument('--eval_episode', default=30, type=int, help="the number of episode to eval the model")
# logging choice
parser.add_argument("--use_tensorboard", default=1, type=int, help="1 as true, 0 as false")
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


# ====================================  NET CONSTRUCTION ===============================
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.uniform_(m.bias, a=-0.003, b=0.003)


class DiscreteActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DiscreteActor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.linear1 = nn.Linear(self.state_dim, hidden_dim)
        self.apply(weights_init_)
        self.output_linear = nn.Linear(hidden_dim, self.action_dim)

    def forward(self, state):
        x = torch.relu(self.linear1(state))
        output = torch.softmax(self.output_linear(x), dim=1)
        return output

    def sample(self, state):
        prob = self.forward(state)
        distribution = Categorical(torch.Tensor(prob))
        sample_action = distribution.sample().unsqueeze(-1)  # [batch, 1]
        z = (prob == 0.0).float() * 1e-8
        logprob = torch.log(prob + z)
        greedy_action = torch.argmax(prob, dim=-1).unsqueeze(-1)  # 1d tensor

        return sample_action, prob, logprob, greedy_action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # q1
        self.linear11 = nn.Linear(self.state_dim, hidden_dim)
        self.linear12 = nn.Linear(hidden_dim, self.action_dim)

        # q2
        self.linear21 = nn.Linear(self.state_dim, hidden_dim)
        self.linear22 = nn.Linear(hidden_dim, self.action_dim)
        self.apply(weights_init_)

    def forward(self, state):
        x1 = torch.relu(self.linear11(state))
        q1 = self.linear12(x1)

        x2 = torch.relu(self.linear21(state))
        q2 = self.linear22(x2)
        return q1, q2


# ====================================  ALGO CONSTRUCTION ===============================
class DiscreteSAC:
    def __init__(self, args, state_dim=None, action_dim=None, writer=None):
        super(DiscreteSAC, self).__init__()
        self.args = args
        self.alpha = args.alpha
        self.batch_size = args.batch_size
        self.update_time = args.update_time
        self.gamma = args.gamma
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr_actor = args.lr_actor
        self.lr_critic = args.lr_critic
        self.alpha = args.alpha
        self.lr_alpha = args.lr_alpha
        self.replace_tau = args.replace_tau
        self.Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])
        self.buffer = []
        self.buffer_capacity = args.buffer_capacity
        self.buffer_pointer = 0

        self.critic_net = Critic(state_dim, action_dim, args.hidden_dim)
        self.target_critic_net = Critic(state_dim, action_dim, args.hidden_dim)
        self.actor_net = DiscreteActor(state_dim, action_dim, args.hidden_dim)

        self.target_critic_net.load_state_dict(self.critic_net.state_dict())
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), lr=self.lr_critic)

        self.target_entropy = args.max_entropy_ratio * -np.log(1 / self.action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)

        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr_alpha)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=self.lr_actor)
        self.training_step = 0
        self.writer = writer

    def cal_entropy(self, action_prob, action_log_prob, is_discrete=True):
        if is_discrete:
            # 离散版本，参考https://github.com/boyu-ai/Hands-on-RL/blob/main/%E7%AC%AC14%E7%AB%A0-SAC%E7%AE%97%E6%B3%95.ipynb
            entropy = -torch.sum(action_prob * action_log_prob, dim=1, keepdim=True)
            return entropy
        else:
            # 连续版本
            entropy = -action_log_prob
        return entropy

    def store_transition(self, state, action, reward, next_state, done):
        transition = self.Transition(state, action, reward, next_state, done)
        if len(self.buffer) < self.buffer_capacity:
            self.buffer.append(None)
        self.buffer[self.buffer_pointer] = transition
        self.buffer_pointer = (self.buffer_pointer + 1) % self.buffer_capacity

    def select_action(self, state, train=True):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        sample_action, prob, logprob, greedy_action = self.actor_net.sample(state)
        if train:
            action = sample_action
        else:
            action = greedy_action
        return action.item()

    def update(self):
        state = torch.tensor(np.array([t.state for t in self.buffer]), dtype=torch.float).to(device)
        action = torch.tensor(np.array([t.action for t in self.buffer]), dtype=torch.long).to(device).unsqueeze(1)
        reward = torch.tensor(np.array([t.reward for t in self.buffer]), dtype=torch.float).to(device).unsqueeze(1)
        next_state = torch.tensor(np.array([t.next_state for t in self.buffer]), dtype=torch.float).to(device)
        terminal = torch.tensor(np.array([t.done for t in self.buffer]), dtype=torch.float).to(device).unsqueeze(1)
        for i in range(self.update_time):
            index = np.random.choice(len(self.buffer), self.batch_size)
            with torch.no_grad():
                next_state_action, next_action_prob, next_action_log_prob, _ = self.actor_net.sample(
                    next_state[index])
                q1_next_target, q2_next_target = self.target_critic_net(next_state[index])
                # v1
                min_qvalue = torch.sum(next_action_prob * torch.min(q1_next_target, q2_next_target), dim=1,
                                       keepdim=True)
                min_q_next_target = min_qvalue + self.log_alpha.exp() * self.cal_entropy(next_action_prob,
                                                                                         next_action_log_prob)
                next_q_value = reward[index] + (1 - terminal[index]) * self.gamma * min_q_next_target

            q1, q2 = self.critic_net(state[index])
            q1 = q1.gather(1, action[index].long())
            q2 = q2.gather(1, action[index].long())  # [batch, 1] , pick the actin-value for the given batched actions

            q1_loss = 0.5 * F.mse_loss(q1, next_q_value.detach())
            q2_loss = 0.5 * F.mse_loss(q2, next_q_value.detach())
            q_loss = q1_loss + q2_loss
            # update critic network
            self.critic_net_optimizer.zero_grad()
            q_loss.backward()
            # nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
            self.critic_net_optimizer.step()

            # update actor network
            q1_pi, q2_pi = self.critic_net(state[index])
            pi, pi_prob, pi_log_prob, _ = self.actor_net.sample(state[index])
            # v1
            min_qvalue = torch.sum(pi_prob * torch.min(q1_pi, q2_pi), dim=1, keepdim=True)  # 直接根据概率计算期望
            pi_loss = -torch.mean(self.log_alpha.exp() * self.cal_entropy(pi_prob, pi_log_prob) + min_qvalue)

            # pi_loss = (pi_prob*(self.alpha.detach() * pi_log_prob - min_q_pi)).sum(1).mean()
            self.actor_optimizer.zero_grad()
            pi_loss.backward()
            # nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            # update alpha
            alpha_loss = torch.mean(
                (self.cal_entropy(pi_prob, pi_log_prob) - self.target_entropy).detach() * self.log_alpha.exp())

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().detach()

            self.training_step += 1

            if self.args.use_wandb:
                wandb.log({'loss/critic loss': q_loss.item(),
                           "loss/actor loss": pi_loss.item(),
                           "loss/alpha loss": alpha_loss.item(),
                           "loss/alpha": self.alpha.clone().item()
                           })
            if self.args.use_tensorboard:
                self.writer.add_scalar('loss/critic loss', q_loss.item(), self.training_step)
                self.writer.add_scalar('loss/actor loss', pi_loss.item(), self.training_step)
                self.writer.add_scalar('loss/alpha loss', alpha_loss.item(), self.training_step)
                self.writer.add_scalar('loss/alpha', self.alpha.clone().item(), self.training_step)
        for target_param, source_param in zip(self.target_critic_net.parameters(), self.critic_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.replace_tau) + source_param.data * self.replace_tau)

    def save(self, save_path, episode):
        base_path = os.path.join(save_path, 'trained_model')
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        model_actor_path = os.path.join(base_path, "actor_" + str(episode) + ".pth")
        torch.save(self.actor_net.state_dict(), model_actor_path)
        model_critic_path = os.path.join(base_path, "critic_" + str(episode) + ".pth")
        torch.save(self.critic_net.state_dict(), model_critic_path)
        param_alpha_path = os.path.join(base_path, "alpha_" + str(episode) + ".pth")
        torch.save(self.log_alpha, param_alpha_path)
        print("success saving model in {}".format(str(base_path)))

    def load(self, load_path, episode):
        print(f'\nBegin to load model: ')
        run_path = os.path.join(load_path, 'trained_model')
        model_actor_path = os.path.join(run_path, "actor_" + str(episode) + ".pth")
        model_critic_path = os.path.join(run_path, "critic_" + str(episode) + ".pth")
        param_alpha_path = os.path.join(run_path, "alpha_" + str(episode) + ".pth")
        print(f'Actor path: {model_actor_path}')
        print(f'Critic path: {model_critic_path}')
        print(f'param_alpha_path: {param_alpha_path}')

        if os.path.exists(model_critic_path) and os.path.exists(model_actor_path):
            actor = torch.load(model_actor_path, map_location=device)
            critic = torch.load(model_critic_path, map_location=device)
            self.actor_net.load_state_dict(actor)
            self.critic_net.load_state_dict(critic)
            self.target_critic_net.load_state_dict(self.critic_net.state_dict())
            print("Model loaded!")
        else:
            sys.exit(f'Model not founded!')

        if os.path.exists(param_alpha_path):
            self.log_alpha = torch.load(param_alpha_path, map_location=device)
            print("log_alpha loaded!")


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


def train_gym(args):
    begin_time = time.time()
    print("env_name:{}".format(args.env_name))
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
    action_dim = action_space.n
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
    model = DiscreteSAC(args, state_dim, action_dim, writer=tf_writer)

    episode = 0
    train_count = 0
    plt_ep = []
    plt_step = []
    plt_ep_reward = []
    while episode < args.max_episode:
        state, info = env.reset(seed=args.random_seed)
        episode += 1
        step = 0
        ep_reward = 0
        while True:
            action = model.select_action(state, True)
            next_state, reward, terminated, truncated, info = env.step(action)
            step += 1
            model.store_transition(state, action, reward, next_state, terminated)
            if len(model.buffer) >= 500:
                model.update()  # model training
                train_count += 1
            state = next_state
            # 累加奖励
            ep_reward += reward
            if terminated or truncated:
                print("Episode:", episode, "; step:", step, "; Episode Return:", '%.2f' % ep_reward,
                      '; Trained episode:', train_count)
                if args.use_wandb:
                    wandb.log({'reward/ep_reward': ep_reward,
                               "reward/ep_step": step})
                if args.use_tensorboard:
                    tf_writer.add_scalar('reward/step', step, episode)
                    tf_writer.add_scalar('reward/ep_reward', ep_reward, episode)
                plt_ep.append(episode)
                plt_step.append(step)
                plt_ep_reward.append(ep_reward)
                break
        if (episode >= 0.5 * args.max_episode and episode % args.save_interval == 0) or episode == args.max_episode:
            model.save(save_path, episode)
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

    ax1.plot(plt_ep, plt_step)
    ax1.set_xlabel('Number of training episodes', font1)
    ax1.set_ylabel('steps', font1)

    label1 = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label1]

    label1 = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label1]

    ax2.plot(plt_ep, plt_ep_reward)
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
    action_dim = action_space.n
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    # define save path
    save_path, _ = make_save_path(args.env_name, args.algo_name, is_make_dir=False)
    print("run_dir:{}".format(save_path))
    # 创建智能体
    model = DiscreteSAC(args, state_dim, action_dim)
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
            action = model.select_action(state, True)
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
    train_gym(args)
    # specify which model to be load and eval
    args.load_run = 1
    args.load_episode = args.max_episode
    eval_gym(args)
