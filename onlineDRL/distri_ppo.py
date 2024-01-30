import argparse
import time
import numpy as np
import sys
import os
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.optim as optim
from torch.distributions import Categorical
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import gym
import matplotlib.pyplot as plt
from pathlib import Path
from pprint import pprint
from multiprocessing import Process, Pipe
from multiprocessing.managers import BaseManager
import wandb

base_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_dir)

font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 16,
         }

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', default="CartPole-v1", type=str)
parser.add_argument('--algo_name', default="distri_ppo", type=str)
parser.add_argument('--max_episode', default=400, type=int)
parser.add_argument('--max_step', default=500, type=int)
parser.add_argument('--hidden_dim', default=128, type=int)
parser.add_argument('--buffer_capacity', default=int(1e4), type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr_actor', default=0.002, type=float)
parser.add_argument('--lr_critic', default=0.002, type=float)
parser.add_argument('--max_grad_norm', default=0.5, type=float)
parser.add_argument('--clip_param', default=0.2, type=float)
parser.add_argument('--gamma', default=0.99, type=float)
parser.add_argument("--update_time", default=5, type=int)
parser.add_argument('--random_seed', default=2024, type=int)
parser.add_argument("--save_interval", default=100, type=int)
# distribute
parser.add_argument("--num_workers", default=1, type=int,
                    help="rollout workers number pre GPU (adjust it to get high GPU usage)")
parser.add_argument("--num_threads", default=2, type=int,
                    help="cpu_num for pytorch, `torch.set_num_threads(self.num_threads)`")
parser.add_argument("--gpu_id", default=-1, type=int,
                    help="`int` means the ID of single GPU, -1 means CPU")
parser.add_argument("--explore_episode", default=5, type=int,
                    help="collect explore_episode while interacting with env, then update networks")
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


class ReplayBuffer:
    def __init__(self):
        self.buffer = []

    def add(self, state, action, action_probs, reward, next_state, done):
        self.buffer.append((state, action, action_probs, reward, next_state, done))

    def add_batch(self, batch_transition):
        self.buffer.extend(batch_transition)

    def clean_buffer(self):
        self.buffer = []

    def get_buffer(self):
        return self.buffer

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, action_probs, reward, next_state, done = zip(*transitions)
        return np.array(state), np.array(action), np.array(action_probs), np.array(reward), np.array(
            next_state), np.array(done)

    def size(self):
        return len(self.buffer)


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
        self.linear2 = nn.Linear(hidden_dim, int(hidden_dim / 2))
        self.apply(weights_init_)
        self.output_linear = nn.Linear(int(hidden_dim / 2), self.action_dim)

    def forward(self, state):
        x = torch.relu(self.linear1(state))
        x = torch.relu(self.linear2(x))
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
    def __init__(self, state_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.state_dim = state_dim

        self.linear11 = nn.Linear(self.state_dim, hidden_dim)
        self.linear12 = nn.Linear(hidden_dim, int(hidden_dim / 2))
        self.linear13 = nn.Linear(int(hidden_dim / 2), 1)

        self.apply(weights_init_)

    def forward(self, state):
        x1 = torch.relu(self.linear11(state))
        x1 = torch.relu(self.linear12(x1))
        q1 = self.linear13(x1)
        return q1


# ====================================  ALGO CONSTRUCTION ===============================
class DistriPPO:
    def __init__(self, args, state_dim=None, action_dim=None, is_learner=False, worker_id=-1):
        super(DistriPPO, self).__init__()
        self.args = args
        self.random_seed = args.random_seed
        self.clip_param = args.clip_param
        self.max_grad_norm = args.max_grad_norm
        self.batch_size = args.batch_size
        self.update_time = args.update_time
        self.gamma = args.gamma
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr_actor = args.lr_actor
        self.lr_critic = args.lr_critic

        self.critic_net = Critic(state_dim, args.hidden_dim)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=self.lr_critic)

        self.actor_net = DiscreteActor(state_dim, action_dim, args.hidden_dim)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=self.lr_actor)

        self.training_step = 0
        self.is_learner = is_learner
        self.batch_size = args.batch_size
        self.worker_id = worker_id
        self.device = device
        self.explore_episode = args.explore_episode
        self.episode = 0
        self.step = 0

    def select_action(self, state, train=True):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        sample_action, prob, logprob, greedy_action = self.actor_net.sample(state)
        if train:
            action = sample_action
        else:
            action = greedy_action
        return action.item(), prob[:, action.item()].item()

    def explore_with_env(self, env):
        """
        Collect trajectories through the actor-environment interaction for a **single** environment instance.

        env: RL training environment. env.reset() env.step(). It should be a vector env.
        return: `[(state, action, reward, next_state, terminated) * self.accumulated_step] ` for off-policy
                logging_dict  for log show
        """
        accumulated_transition = []
        logging_dict = {}
        for ep in range(self.explore_episode):
            state, info = env.reset(seed=self.random_seed)
            step = 0
            ep_reward = 0
            while True:
                action, action_prob = self.select_action(state, train=True)
                next_state, reward, terminated, truncated, info = env.step(action)  # next_state
                accumulated_transition.append((state, action, action_prob, reward, next_state, terminated))
                step += 1
                ep_reward += reward
                state = next_state
                if terminated or truncated:
                    self.episode += 1
                    next_state, _ = env.reset()
                    if self.worker_id == 0:
                        logging_dict = {'reward/episode': self.episode,
                                        'reward/ep_reward': ep_reward,
                                        'reward/final_reward': reward,
                                        "reward/ep_step": step}
                    else:
                        logging_dict = {'reward/episode_{}'.format(self.worker_id): self.episode,
                                        'reward/ep_reward_{}'.format(self.worker_id): ep_reward,
                                        'reward/final_reward_{}'.format(self.worker_id): reward,
                                        "reward/ep_step_{}".format(self.worker_id): step}
                    break
        return accumulated_transition, logging_dict

    def update(self, curr_buffer):
        logging_dict = {}
        state = torch.tensor(np.array([t[0] for t in curr_buffer]), dtype=torch.float).to(device)
        action = torch.tensor(np.array([t[1] for t in curr_buffer]), dtype=torch.int64).view(-1, 1).to(device)
        old_action_probs = torch.tensor(np.array([t[2] for t in curr_buffer]), dtype=torch.float).view(-1,
                                                                                                       1).to(
            device)
        reward = [t[3] for t in curr_buffer]
        terminal = [t[-1] for t in curr_buffer]
        # Monte Carlo estimate of state rewards:
        Gt = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(reward), reversed(terminal)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            Gt.insert(0, discounted_reward)
        Gt = torch.tensor(Gt, dtype=torch.float)
        for i in range(self.update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(curr_buffer))), self.batch_size, False):
                Gt_index = Gt[index].view(-1, 1)
                V = self.critic_net(state[index])
                delta = Gt_index - V
                advantage = delta.detach()
                action_prob = self.actor_net(state[index]).gather(1, action[index])
                ratio = action_prob / old_action_probs[index]
                surr1 = ratio * advantage
                surr2 = (torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage)

                pi_loss = -torch.min(surr1, surr2).mean()
                self.actor_optimizer.zero_grad()
                pi_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                value_loss = F.mse_loss(Gt_index, V)
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                self.training_step += 1
                logging_dict = {"loss/critic loss": value_loss.item(),
                                "loss/actor loss": pi_loss.item(),
                                "train/training_step": self.training_step
                                }
        return logging_dict

    def save(self, save_path, episode):
        base_path = os.path.join(save_path, 'trained_model')
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        model_actor_path = os.path.join(base_path, "actor_" + str(episode) + ".pth")
        torch.save(self.actor_net.state_dict(), model_actor_path)
        model_critic_path = os.path.join(base_path, "critic_" + str(episode) + ".pth")
        torch.save(self.critic_net.state_dict(), model_critic_path)
        print("success saving model in {}".format(str(base_path)))

    def load(self, load_path, episode):
        print(f'\nBegin to load model: ')
        run_path = os.path.join(load_path, 'trained_model')
        model_actor_path = os.path.join(run_path, "actor_" + str(episode) + ".pth")
        model_critic_path = os.path.join(run_path, "critic_" + str(episode) + ".pth")
        print(f'Actor path: {model_actor_path}')
        print(f'Critic path: {model_critic_path}')

        if os.path.exists(model_critic_path) and os.path.exists(model_actor_path):
            actor = torch.load(model_actor_path, map_location=device)
            critic = torch.load(model_critic_path, map_location=device)
            self.actor_net.load_state_dict(actor)
            self.critic_net.load_state_dict(critic)
            print("Model loaded!")
        else:
            sys.exit(f'Model not founded!')


# ====================================  WORKER CONSTRUCTION  ===============================
class Learner(Process):
    def __init__(self, replay_buffer, learner_pipe: Pipe, worker_pipes: [Pipe], evaluator_pipe: Pipe, all_args):
        super().__init__()
        self.recv_pipe = learner_pipe[0]
        self.send_pipes = [worker_pipe[1] for worker_pipe in worker_pipes]
        self.eval_pipe = evaluator_pipe[1]
        self.args = all_args
        self.replay_buffer = replay_buffer

    def run(self):
        args = self.args
        torch.set_grad_enabled(False)

        '''init agent'''
        agent = DistriPPO(args, args.state_dim, args.action_dim, is_learner=True, worker_id=-1)

        save_path = args.save_path
        num_workers = args.num_threads
        batch_size = args.batch_size

        '''Learner send actor to Workers'''
        for send_pipe in self.send_pipes:
            send_pipe.send(agent.actor_net)
        del args

        while True:
            '''agent update network using training data'''
            if self.replay_buffer.size() > batch_size * 4:
                curr_train_data = self.replay_buffer.get_buffer()
                torch.set_grad_enabled(True)
                logging_dict = agent.update(curr_train_data)
                torch.set_grad_enabled(False)
                logging_dict["train/replay_buffer"] = len(curr_train_data)
                self.eval_pipe.send(logging_dict)
                ''' clear replay_buffer, be sure the train data is come from the newest model '''
                self.replay_buffer.clean_buffer()

            '''Learner send actor to Workers'''
            for send_pipe in self.send_pipes:
                send_pipe.send(agent.actor_net)

            # if logging_dict["train/training_step"] != 0 and logging_dict["train/training_step"] % 200 == 0:
            #     agent.save(save_path, logging_dict["train/training_step"])

            '''Learner receive finish flag from Workers'''
            is_finish = False
            recv_worker_info = []
            for _ in range(num_workers):
                worker_id, worker_episode, train_signal = self.recv_pipe.recv()
                recv_worker_info.append((worker_id, worker_episode, train_signal))
                if not train_signal:
                    is_finish = True
                    break
            if is_finish:
                break

        '''Learner send the terminal signal to workers after break the loop'''
        for send_pipe in self.send_pipes:
            send_pipe.send(None)
        self.eval_pipe.send(None)

        '''save'''
        agent.save(save_path, 0)


class Worker(Process):
    def __init__(self, replay_buffer, worker_pipe: Pipe, learner_pipe: Pipe, evaluator_pipe: Pipe, worker_id: int,
                 all_args):
        super().__init__()
        self.recv_pipe = worker_pipe[0]  # receive actor_net
        self.send_pipe = learner_pipe[1]  # send accumulated_transition
        self.eval_pipe = evaluator_pipe[1]  # send explore log
        self.worker_id = worker_id
        self.args = all_args
        self.replay_buffer = replay_buffer

    def run(self):
        args = self.args
        worker_id = self.worker_id
        torch.set_grad_enabled(False)

        '''init environment'''
        env = gym.make(args.env_name, render_mode="rgb_array")

        '''init agent'''
        agent = DistriPPO(args, args.state_dim, args.action_dim, is_learner=False, worker_id=worker_id)

        max_episode = args.max_episode
        '''loop'''
        del args

        while True:
            '''Worker receive actor from Learner'''
            actor_net = self.recv_pipe.recv()
            if actor_net is None:
                break
            else:
                agent.actor_net = actor_net

            '''Worker explore_with_env send the training data to Learner'''
            accumulated_transition, logging_dict = agent.explore_with_env(env)
            self.replay_buffer.add_batch(accumulated_transition)
            logging_dict["replay_buffer"] = self.replay_buffer.size()
            if len(logging_dict) > 0:
                self.eval_pipe.send(logging_dict)
            if agent.episode >= max_episode:
                self.send_pipe.send((worker_id, agent.episode, False))
            else:
                self.send_pipe.send((worker_id, agent.episode, True))

        env.close() if hasattr(env, 'close') else None


class Evaluator(Process):
    def __init__(self, evaluator_pipe: Pipe, all_args):
        super().__init__()
        self.recv_pipe = evaluator_pipe[0]  # receive log from learner and worker
        self.args = all_args

    def run(self):
        args = self.args
        torch.set_grad_enabled(False)

        '''wandb(weights & biases): Track and visualize all the pieces of your machine learning pipeline.'''
        if args.use_wandb:
            wandb.init(config=args,
                       project=args.env_name,
                       name=args.algo_name + "_" + args.run_index + "_" + datetime.datetime.now().strftime(
                           "%Y-%m-%d_%H-%M-%S"),
                       dir=args.save_path,
                       job_type="training",
                       reinit=True,
                       monitor_gym=True)

        while True:
            '''Evaluator receive training log from Learner'''
            logging_dict = self.recv_pipe.recv()
            if logging_dict is None:
                break
            if "train/training_step" not in logging_dict:
                print("logging_dict:{}".format(logging_dict))
            if args.use_wandb:
                wandb.log(logging_dict)
        if args.use_wandb:
            wandb.finish()


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


def distri_train_gym(args, replay_buffer):
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
    args.state_dim = state_dim
    args.action_dim = action_dim
    args.save_path = save_path
    args.run_index = curr_run

    '''build the Pipe'''
    '''
    返回一对 Connection 对象 (conn1, conn2) ， 分别表示管道的两端。
    如果 duplex 被置为 True (默认值)，那么该管道是双向的。如果 duplex 被置为 False ，那么该管道是单向的，即 conn1 只能用于接收消息，而 conn2 仅能用于发送消息。
    '''
    worker_pipes = [Pipe(duplex=False) for _ in range(args.num_threads)]  # receive, send
    learner_pipe = Pipe(duplex=False)
    evaluator_pipe = Pipe(duplex=False)

    '''build Process'''
    learner = Learner(replay_buffer=replay_buffer, learner_pipe=learner_pipe, worker_pipes=worker_pipes,
                      evaluator_pipe=evaluator_pipe, all_args=args)
    workers = [
        Worker(replay_buffer=replay_buffer, worker_pipe=worker_pipe, learner_pipe=learner_pipe,
               evaluator_pipe=evaluator_pipe, worker_id=worker_id, all_args=args)
        for worker_id, worker_pipe in enumerate(worker_pipes)]
    evaluator = Evaluator(evaluator_pipe=evaluator_pipe, all_args=args)

    '''start Process'''
    process_list = [learner, *workers, evaluator]
    [process.start() for process in process_list]
    [process.join() for process in process_list]
    print("执行时间：{}".format(time.time() - begin_time))


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
    model = DistriPPO(args, state_dim, action_dim)
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
            action, action_prob = model.select_action(state, train=False)
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
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    plt.savefig(os.path.join(fig_path, "eval_result.jpg"))
    print("eval figure is saved in {}".format(str(os.path.join(fig_path, "eval_result.jpg"))))


if __name__ == '__main__':
    args = parser.parse_args()
    # show env information,
    # get_env_args(args.env_name, render=False)
    # the replay buffer is a class, have to use torch manager to make it a proxy for sharing across processes
    BaseManager.register('ReplayBuffer', ReplayBuffer)
    manager = BaseManager()
    manager.start()
    replay_buffer = manager.ReplayBuffer()  # share the replay buffer through manager
    distri_train_gym(args, replay_buffer)
    # specify which model to be load and eval
    args.load_run = 1
    args.load_episode = 0
    eval_gym(args)
