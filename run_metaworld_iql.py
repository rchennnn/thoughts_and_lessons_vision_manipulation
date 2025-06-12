# source: https://github.com/gwthomas/IQL-PyTorch
# https://arxiv.org/pdf/2110.06169.pdf
# line 453 for critic loss

tasks = [
    'assembly',
    'xbin-picking',
    'coffee-pull',
    'dial-turn',
    'door-close',
    'hammer',
    'pick-place',
    'faucet-close',
    'pick-place-wall',
    'plate-slide',
    'push-wall',
    'push',
    'soccer',
    'stick-push',
    'sweep',
    'window-close',
    'window-open',
    'fetch-push',
    'fetch-place',
    'hand-hammer',
    'hand-door',
]

import copy
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# import d4rl
import imageio 
import h5py
# import gymnasium as gym
import numpy as np
# import pyrallis
import torch
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F
# import wandb
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR, ConstantLR

import importlib

# from experts.fetch import FetchPlayer  
import utils.encoders as encoders
from iql.iq_loss import iq_loss

from metaworld import MT1

TensorBatch = List[torch.Tensor]

EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda"
    env: str = "push-v2"  # OpenAI gym environment name
    train_seed: int = 42 # from make_embeddings_metaworld.sh for make mode
    train_trajectories: int = 2000 # how many trajectories to use for training
    pos_embed: bool = True
    img_size: int = 224
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_seed: int = 42 #1337
    eval_trajectories: int = 20
    eval_freq: int = 10#int(5e3)  # How often (time steps) we evaluate
    n_episodes: int = 20  # How many episodes run during evaluation
    n_steps: int = 40
    max_timesteps: int = 100#int(1e6)  # Max time steps to run environment
    checkpoints_path: Optional[str] = None #'envs/iql/trained_agents/push'  # Save path
    log_path: Optional[str] = None #'envs/iql/logs/push/run_dropout.log' # experiment output logs
    load_model: str = ""  # Model load file name, "" doesn't load
    encoder: str = "sam"
    traj_len: int = 100
    # IQL
    buffer_size: int = 50_000 #250_000  # Replay buffer size # original 2500000
    batch_size: int = 1024  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    tau: float = 0.005  # Target network update rate
    beta: float = 3  # Inverse temperature. Small beta -> BC, big beta -> maximizing Q
    iql_tau: float = 0.7  # Coefficient for asymmetric loss
    iql_deterministic: bool = False  # Use deterministic actor
    normalize: bool = False  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    vf_lr: float = 3e-4  # V function learning rate
    qf_lr: float = 3e-4  # Critic learning rate
    actor_lr: float = 1.5e-4  # Actor learning rate
    actor_dropout: Optional[float] = None #0.3 #None  # Adroit uses dropout for policy network
    # Wandb logging
    project: str = "CORL"
    group: str = "IQL-D4RL"
    name: str = "IQL"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


# def wrap_env(
#     env: gym.Env,
#     state_mean: Union[np.ndarray, float] = 0.0,
#     state_std: Union[np.ndarray, float] = 1.0,
#     reward_scale: float = 1.0,
# ) -> gym.Env:
#     # PEP 8: E731 do not assign a lambda expression, use a def
#     def normalize_state(state):
#         return (
#             state - state_mean
#         ) / state_std  # epsilon should be already added in std.

#     def scale_reward(reward):
#         # Please be careful, here reward is multiplied by scale!
#         return reward_scale * reward

#     env = gym.wrappers.TransformObservation(env, normalize_state)
#     if reward_scale != 1.0:
#         env = gym.wrappers.TransformReward(env, scale_reward)
#     return env


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
        conf: TrainConfig = None
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0
        self.conf = conf
        
        # self._states = torch.zeros(
        #     (buffer_size, state_dim), dtype=torch.float32, device=device
        # )
        # self._positions = torch.zeros(
        #     (buffer_size, state_dim), dtype=torch.float32, device=device
        # )
        # self._states_memory_size_gb = self._states.numel() * self._states.element_size() / (1024 ** 3)
        # print(f"Memory size of _states during init: {self._states_memory_size_gb:.6f} GB")
        # self._actions = torch.zeros(
        #     (buffer_size, action_dim), dtype=torch.float32, device=device
        # )
        # self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        # # self._next_states = torch.zeros(
        # #     (buffer_size, state_dim), dtype=torch.float32, device=device
        # # )
        # self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)

        self._device = device
        self.encoding_map= {'sam': 'sam_embeddings224', 
                            'dino': 'dino_embeddings',
                            'clip': 'clip_embeddings',
                            'mae': 'mae_embeddings',
                            'mvp': 'mvp_embeddings',
                            'r3m': 'r3m_embeddings',
                            'vip': 'vip_embeddings',
                            'vc1': 'vc1_embeddings',
                            'moco': 'moco_embeddings',
                            'ibot': 'ibot_embeddings'
                            }

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)
    
    def _flatten(self, data: np.ndarray) -> np.ndarray:
        # currently giving identity
        return data.reshape((data.shape[0]*data.shape[1], *data.shape[2:]))
        # return data

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        # self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")
    
    def load_hdf5_dataset(self, path: str, encoding_type: str, trajectories: int = 2950):
        h = h5py.File(path, 'r')
        num_traj = trajectories
        encoding = self.encoding_map[encoding_type]
        self.traj_steps = conf.traj_len
        n_transitions = num_traj * self.traj_steps
        print(n_transitions, self._buffer_size)
        
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        print(h['state'].shape)
        image_embeddings = h[encoding][:num_traj,:self.traj_steps].reshape(num_traj, self.traj_steps, -1)
        # goal_embeddings = np.stack([image_embeddings[:, -1, :]]*self.traj_steps, axis=1)
        # print(goal_embeddings.shape)
        
        positions = h["state"][:num_traj,:self.traj_steps].reshape(num_traj, self.traj_steps, -1)
        if conf.pos_embed:
            init_obs = positions
            goal_obs = np.stack([init_obs[:, -1, :]]*self.traj_steps, axis=1)
        else:
            init_obs = None
            goal_obs = None
        
        if conf.pos_embed:
            print(f'Using observations', init_obs.shape, goal_obs.shape)
        
        # goal_obs = np.stack([init_obs[:, -1, :]]*self.traj_steps, axis=1)
        # observations = h['observations'][:]
        
        # observations = np.concatenate((image_embeddings, positions,
        #                                goal_embeddings, goal_positions
        #                                ), 
        #                               axis = -1)
        if conf.pos_embed:
            observations = np.concatenate((image_embeddings, init_obs,
                                    #    goal_embeddings, 
                                       goal_obs), 
                                      axis = -1)
        else:
            observations = np.concatenate((image_embeddings,
                                    #    goal_embeddings,
                                       ), 
                                      axis = -1)

        
        # observations = image_embeddings
        print(observations.shape)
        # observations = normalize_states(
        #     observations, 0, 1
        # )
        
        terminal_marker = np.zeros((num_traj, self.traj_steps, 1))
        terminal_marker[:,-1] = 1
        _temp_next_state = observations[:,1:]
        _to_append = np.expand_dims(_temp_next_state[:,-1], 1)
        print(_to_append.shape, _temp_next_state.shape)
        _next_state = np.concatenate((_temp_next_state, _to_append), axis=1)
        
        
        rewards = torch.arange(self.traj_steps).view(1, self.traj_steps, 1).expand(num_traj, -1, -1).float()
        
        # self._states[:n_transitions] = self._to_tensor(self._flatten(observations[:num_traj]))
        
        
        # self._states = self._to_tensor(self._flatten(observations[:num_traj]))
        # self._actions = self._to_tensor(self._flatten(h["actions"][:num_traj]))
        # self._positions = self._to_tensor(self._flatten(positions[:num_traj]))
        # # self._rewards = self._to_tensor(self._flatten(np.zeros((num_traj, self.traj_steps, 1)))) # sparse rewards - no reward used
        # self._rewards = self._to_tensor(self._flatten(rewards))
        # # self._next_states = self._to_tensor(self._flatten(_next_state))
        # # step_counts = np.arange(1, self.traj_steps + 1).reshape(1, self.traj_steps, 1)
        # # self._next_step_counts = np.tile(step_counts, (num_traj, 1, 1))
        # self._next_states = self._flatten(_next_state)
        # self._dones = self._to_tensor(self._flatten(terminal_marker))
        
        self._states = self._to_tensor(observations[:num_traj])
        self._actions = self._to_tensor(h["actions"][:num_traj])
        self._positions = self._to_tensor(positions[:num_traj])
        # self._rewards = self._to_tensor(self._flatten(np.zeros((num_traj, self.traj_steps, 1)))) # sparse rewards - no reward used
        self._rewards = self._to_tensor(rewards)
        # self._next_states = self._to_tensor(self._flatten(_next_state))
        # step_counts = np.arange(1, self.traj_steps + 1).reshape(1, self.traj_steps, 1)
        # self._next_step_counts = np.tile(step_counts, (num_traj, 1, 1))
        self._next_states = self._flatten(_next_state)
        self._dones = self._to_tensor(terminal_marker)
        
        print('Data loaded')
        observations_size_gb = observations.nbytes / (1024 ** 3)
        print(f"Size of observations in GB: {observations_size_gb:.6f} GB")
        available_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert bytes to GB
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # Convert bytes to GB
        free_memory = total_memory - available_memory
        print(f"Available CUDA memory: {free_memory:.2f} GB")
    
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        trajectory_indices = indices // self.traj_steps  # Integer division to get the trajectory index
        step_indices = indices % self.traj_steps 
        next_state_indices = np.where(
            (step_indices + 1) < self.traj_steps,  # Ensure we don't go out of bounds
            step_indices + 1,  # Get the next step in the same trajectory
            step_indices # no transition
        )   
        
        goals = self._states[trajectory_indices, -1]
        
        states = self._states[trajectory_indices, step_indices]
        states = torch.cat((states, goals), dim=-1)
        
        # positions = self._positions[indices]
        actions = self._actions[trajectory_indices, step_indices]
        rewards = self._rewards[trajectory_indices, step_indices]
        
        next_states = self._states[trajectory_indices, next_state_indices]
        next_states = torch.cat((next_states, goals), dim=-1)
        
        dones = self._dones[trajectory_indices, step_indices]
        
        return [states, actions, rewards, next_states, dones]

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        raise NotImplementedError


# def set_seed(
#     seed: int, env = None, deterministic_torch: bool = False
# ):
    # if env is not None:
    #     env.reset(seed = seed)
    #     env.action_space.seed(seed)
    # os.environ["PYTHONHASHSEED"] = str(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    # torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(deterministic_torch)


# def wandb_init(config: dict) -> None:
#     wandb.init(
#         config=config,
#         project=config["project"],
#         group=config["group"],
#         name=config["name"],
#         id=str(uuid.uuid4()),
#     )
#     wandb.run.save()

def make_gif(image_list, name):
    imageio.mimsave(f'/home/ubuntu/metaworld_encoders/logs/{name}.gif', image_list)
    # imageio.mimsave(f'/home/rce5022/metaworld_encoders/logs/{name}.gif', image_list)

def get_encoder(encoder_name):
    emap = {
        'sam': encoders.SAMPolicy,
        'dino': encoders.DinoV2Policy,
        'clip': encoders.CLIPPolicy,
        'mae': encoders.MAEPolicy,
        'mvp': encoders.MVPPolicy,
        'r3m': encoders.R3MPolicy,
        'vip': encoders.VIPPolicy,
        'vc1': encoders.VC1Policy,
        'moco': encoders.MoCoV3Policy,
        'ibot': encoders.IBOTPolicy
    }
    policy_object = emap[encoder_name](device='cuda')
    policy_object.encoder.eval()
    return policy_object.encoder, policy_object.embed_size, policy_object.preprocess

def randomize_background(image):
    target_pixel = np.array([114, 218, 145]).reshape(3,1,1)
    diff = np.linalg.norm(image - target_pixel, axis=0)
    mask = diff < 4
    # random_colors = np.random.choice(np.arange(256), size=(3, image.shape[1], image.shape[2]))
    random_colors = np.zeros((3, image.shape[1], image.shape[2]))
    random_colors[0,:,:] = 255
    image[:,mask] = random_colors[:,mask]
    print('f')
    return image

@torch.no_grad()
def eval_actor(
    env, 
    tasks,
    actor: nn.Module, device: str, encoder = None, preprocess=None, expert = None, config: TrainConfig = None) -> np.ndarray:
    
    env_name = config.env
    p = expert()
    def play_expert(env, task, p):
        env.set_task(task)
        obs, info = env.reset()
        done = False
        for _ in range(150):
            a = p.get_action(obs=obs)
            obs, _, _, _, info = env.step(a)
            done = int(info["success"]) == 1
            arm_state = env.get_arm_state()
            # if done:
            #     img = env.render().transpose((2,0,1))
            #     done = True
                # break
        if done:
            img = env.render().transpose((2,0,1))
        else:
            img = None

        return arm_state, img, done
    
    actor.eval()
    
    episode_rewards = []
    pbar = tqdm(range(config.eval_trajectories))
    goal_imgs = []
    completed_tasks = 0
    for task_num, task in enumerate(tasks):
        if completed_tasks >= config.eval_trajectories:
            break
        goal_arm_state, goal_img, expert_converged = play_expert(env, task, p)
        
        if not expert_converged and goal_img is None:
            continue
        env.set_task(task)
        obs, info = env.reset()
        
        goal_img_copy = goal_img.copy().astype(np.float32)
        # goal_img_copy = randomize_background(goal_img_copy)
        if preprocess:
            goal_img_copy = preprocess(torch.from_numpy(goal_img_copy).unsqueeze(0).to(device))
        goal_embedding = encoder(goal_img_copy)
        goal_embedding  = torch.flatten(goal_embedding, start_dim = 1).detach().cpu().numpy().squeeze()
        imgs = []
        goal_imgs.append(goal_img.transpose((1,2,0)))
        info = {'success': False}
        for _ in range(150):
            if info['success']:
                continue
            img = env.render()
            arm_state = env.get_arm_state()
            img_copy = img.copy().transpose((2,0,1)).astype(np.float32)
            # img_copy = randomize_background(img_copy)
            if preprocess:
                img_copy = preprocess(torch.from_numpy(img_copy).unsqueeze(0).to(device))   
            init_embedding = encoder(img_copy)
            init_embedding  = torch.flatten(init_embedding, start_dim = 1).detach().cpu().numpy().squeeze()
            if conf.pos_embed:
                augment_state = np.concatenate((init_embedding, arm_state, goal_embedding, goal_arm_state), axis=-1)
            else:
                augment_state = np.concatenate((init_embedding, goal_embedding), axis=-1)
            # augment_state = np.concatenate((init_embedding, arm_state, goal_embedding, goal_arm_state), axis=-1)
            
            action = actor.act(augment_state, device)
            imgs.append(img)
            obs, _, _, _, info = env.step(action)

            # episode_reward += reward
        completed_tasks += 1
        episode_rewards.append(int(info['success']))
        pbar.set_description(f'{str(np.mean(episode_rewards))}. {completed_tasks}/{config.eval_trajectories}')
        # save the last trajectory gif
        # make_gif(imgs, name=f'{env_name}_{task_num}')
        # make_gif(goal_img, name=f'{env_name}_{task_num}_goal')
        # goal_img_path = f'/home/ubuntu/metaworld_encoders/logs/{env_name}_{task_num}_goal.png'
        # goal_img_path = f'/home/rce5022/metaworld_encoders/logs/{env_name}_{task_num}_goal.png'
        # imageio.imwrite(goal_img_path, goal_img.transpose((1, 2, 0)))
        
    actor.train()
    return np.asarray(episode_rewards)


def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(dataset, env_name, max_episode_steps=1000):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class MLP(nn.Module):
    def __init__(
        self,
        dims,
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        output_activation_fn: Callable[[], nn.Module] = None,
        squeeze_output: bool = False,
        dropout: Optional[float] = None,
        ln: Optional[bool] = False
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())

            if dropout is not None:
                layers.append(nn.Dropout(dropout))
            if ln:
                layers.append(nn.LayerNorm(normalized_shape=dims[i + 1]))

        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> Normal:
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        return Normal(mean, std)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        dist = self(state)
        action = dist.mean if not self.training else dist.sample()
        action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
        return action.cpu().data.numpy().flatten()


class DeterministicPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
            dropout=dropout,
        )
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return (
            torch.clamp(self(state) * self.max_action, -self.max_action, self.max_action)
            .cpu()
            .data.numpy()
            .flatten()
        )


class TwinQ(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2
    ):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)

    def both(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)


class ImplicitQLearning:
    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        q_network: nn.Module,
        q_optimizer: torch.optim.Optimizer,
        v_network: nn.Module,
        v_optimizer: torch.optim.Optimizer,
        iql_tau: float = 0.7,
        beta: float = 3.0,
        max_steps: int = 1000000,
        discount: float = 0.99,
        tau: float = 0.005,
        device: str = "cpu",
    ):
        torch.autograd.set_detect_anomaly(True)

        self.max_action = max_action
        self.qf = q_network
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.vf = v_network
        self.actor = actor
        self.v_optimizer = v_optimizer
        self.q_optimizer = q_optimizer
        self.actor_optimizer = actor_optimizer
        self.actor_lr_schedule = ConstantLR(self.actor_optimizer, factor=2/3) # CosineAnnealingLR(self.actor_optimizer, max_steps)
        self.iql_tau = iql_tau
        self.beta = beta
        self.discount = discount
        self.tau = tau

        self.total_it = 0
        self.device = device

    def _update_v(self, observations, next_observations, actions, log_dict) -> torch.Tensor:
        # Update value function
        with torch.no_grad():
            target_q = self.q_target(observations, actions)
            next_v = self.vf(next_observations)
        
        v = self.vf(observations) 
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        log_dict["value_loss"] = v_loss.item()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        
        return adv, v.detach(), next_v
    
    # def _update_v(self, observations, actions, log_dict) -> torch.Tensor:
    #     # Update value function
    #     with torch.no_grad():
    #         target_q = self.q_target(observations, actions)

    #     v = self.vf(observations)
    #     adv = target_q - v
    #     v_loss = asymmetric_l2_loss(adv, self.iql_tau)
    #     log_dict["value_loss"] = v_loss.item()
    #     self.v_optimizer.zero_grad()
    #     v_loss.backward()
    #     self.v_optimizer.step()
    #     return adv

    def _update_q(
        self,
        next_v: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
        log_dict: Dict,
    ):
        targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        log_dict["q_loss"] = q_loss.item()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        soft_update(self.q_target, self.qf, self.tau)

    def _update_policy(
        self,
        adv: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_dict: Dict,
    ):
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.actor(observations)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
        elif torch.is_tensor(policy_out):
            if policy_out.shape != actions.shape:
                raise RuntimeError("Actions shape missmatch")
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses)
        log_dict["actor_loss"] = policy_loss.item()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()
        
    def _update_inverse_q(self, obs, next_obs, actions, rewards, dones, current_v, next_v, log_dict):
        q_val = self.qf(obs, actions)
    
        batch_reorg = obs, next_obs, actions, rewards, dones, torch.ones_like(dones, device=self.device, dtype=torch.bool)
        q_loss, _ = iq_loss(agent = self.qf, current_Q= q_val, current_v= current_v, next_v=next_v, gamma=self.discount, batch=batch_reorg)
        log_dict["q_loss"] = q_loss.item()
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        # Assert to check if the weights of self.qf have been updated
        for param in self.qf.parameters():
            assert param.grad is not None, "The weights of the qf network have not been updated."
        
        soft_update(self.q_target, self.qf, self.tau)

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        # updated to use inverse q loss without gradient error Uses torch==2.3.0 and torchvision==0.18.0
        self.total_it += 1
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        ) = batch
        log_dict = {}

        preturb_obs = observations
        preturb_next_obs = next_observations
        adv, curr_v, next_v = self._update_v(preturb_obs, preturb_next_obs, actions, log_dict)
        rewards = rewards.squeeze(dim=-1)
        rewards = torch.zeros_like(rewards)
        dones = dones.squeeze(dim=-1)
        self._update_inverse_q(preturb_obs, preturb_next_obs, actions, rewards, dones, curr_v, next_v, log_dict)

        self._update_policy(adv, preturb_obs, actions, log_dict)
        
        return log_dict
    
    # def train(self, batch: TensorBatch) -> Dict[str, float]:
    #     # no inverse q loss
    #     self.total_it += 1
    #     (
    #         observations,
    #         actions,
    #         rewards,
    #         next_observations,
    #         dones,
    #     ) = batch
    #     log_dict = {}

    #     with torch.no_grad():
    #         next_v = self.vf(next_observations)
    #     # Update value function
    #     adv = self._update_v(observations=observations, actions=actions, log_dict=log_dict)
    #     rewards = rewards.squeeze(dim=-1)
    #     dones = dones.squeeze(dim=-1)
    #     # Update Q function
    #     self._update_q(next_v, observations, actions, rewards, dones, log_dict)
    #     # Update actor
    #     self._update_policy(adv, observations, actions, log_dict)

    #     return log_dict
    
    # def train(self, batch: TensorBatch) -> Dict[str, float]:
    #     # copied from quest. requires torch==2.0.1 and torchvision==0.15.2
    #     self.total_it += 1
    #     (
    #         observations,
    #         actions,
    #         rewards,
    #         next_observations,
    #         dones,
    #     ) = batch
    #     log_dict = {}

    #     curr_v = self.vf(observations)
    #     with torch.no_grad():
    #         next_v = self.vf(next_observations)
    #     # Update value function
    #     adv = self._update_v(observations, actions, log_dict)
    #     rewards = rewards.squeeze(dim=-1)
    #     dones = dones.squeeze(dim=-1)
    #     # Update Q function
    #     # self._update_q(next_v, observations, actions, rewards, dones, log_dict)
    #     # Update Q with IQ_Loss
    #     q_val = self.q_target(observations, actions)
    #     batch_reorg = observations, next_observations, actions, rewards, dones, torch.ones_like(dones, device=self.device, dtype=torch.bool)
    #     q_loss, _ = iq_loss(agent = self.qf, current_Q= q_val, current_v= curr_v, next_v=next_v, gamma=self.discount, batch=batch_reorg)
    #     log_dict["q_loss"] = q_loss.item()
    #     self.q_optimizer.zero_grad()
    #     q_loss.backward()
    #     self.q_optimizer.step()
    #     soft_update(self.q_target, self.qf, self.tau)
    #     # Update actor
    #     self._update_policy(adv, observations, actions, log_dict)

    #     return log_dict
        

    def state_dict(self) -> Dict[str, Any]:
        return {
            "qf": self.qf.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "vf": self.vf.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "actor_lr_schedule": self.actor_lr_schedule.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.qf.load_state_dict(state_dict["qf"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.q_target = copy.deepcopy(self.qf)

        self.vf.load_state_dict(state_dict["vf"])
        self.v_optimizer.load_state_dict(state_dict["v_optimizer"])

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_lr_schedule.load_state_dict(state_dict["actor_lr_schedule"])

        self.total_it = state_dict["total_it"]


# @pyrallis.wrap()
def train(config: TrainConfig):
    env_name = config.env
    seed = config.train_seed
    trajectories = config.train_trajectories
    mt1 = MT1(env_name, seed=config.eval_seed, num_goals= int(config.eval_trajectories * 2)) # offline, num trajectories = training size
    eval_tasks = mt1.train_tasks
    env = mt1.train_classes[env_name](render_mode='rgb_array',)
    # print(env.mujoco_renderer.default_cam_config)
    # env.mujoco_renderer.default_cam_config = {
    #     "distance": 1.75,
    #     "azimuth": 135.0,
    #     "elevation": -45.0, # -90 is top facing down
    #     "lookat": [0, 0.43, 0.2], # this is a weird positioning issue on lambda... see if replicates on quest or nustat
    #     # [0] is the direction parallel to the edge of table closest to robot. + is moving towards right of robot if facing the table at the robots position
    #     # [1] is the direction perp to the edge of table closest to robot. + is moving in front of robot 
    #     # [2] + is the vertical direction up.
    # }
    
    # test render
    env.mujoco_renderer.default_cam_config = {'distance': 1.75, 'azimuth': 135.0, 'elevation': -45.0,}

    env.set_task(eval_tasks[0])
    env.reset()
    img = env.render()
    imageio.imwrite('test_render.png', img)
    
    # get the expert policy p
    try:
        module_name = env_name
        from_part = 'metaworld.policies.sawyer_' + env_name.replace('-', '_') + '_policy' 
        mod_name_env = env_name.replace('-', ' ')
        mod_name_env = ''.join([i.capitalize() for i in mod_name_env.split()])
        import_part = 'Sawyer' + mod_name_env + 'Policy'
        module = importlib.import_module(from_part)
        p = getattr(module, import_part)
        print('expert:', module, 'module:', import_part)
    except:
        print('Failed to import expert policy module')
        print(from_part, import_part)
        raise ModuleNotFoundError
    
    
    
    encoder, encoder_size, preprocess = get_encoder(config.encoder)

    # state_dim = env.observation_space['observation'].shape[0] + env.observation_space['desired_goal'].shape[0] 
    state_dim = 2 * encoder_size  # + 2 * 10 # + env.observation_space['desired_goal'].shape[0] 
    if config.pos_embed:
        state_dim += 2 * 10
    action_dim = env.action_space.shape[0]

    # dataset = d4rl.qlearning_dataset(env)
    
    # if config.normalize_reward:
    #     modify_reward(dataset, config.env)
    
    # dataset = h5py.File('/projects/p31777/ryan/encoders_data/hammer_data_new.hdf5', 'r')

    # if config.normalize:
    #     state_mean, state_std = compute_mean_std(dataset["observations"][:], eps=1e-3)
    # else:
    #     state_mean, state_std = 0, 1

    # dataset["observations"] = normalize_states(
    #     dataset["observations"], state_mean, state_std
    # )
    # dataset["next_observations"] = normalize_states(
    #     dataset["next_observations"], state_mean, state_std
    # )
    # env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
    )
    
    # fetch env data keys
    # ['actions', 'clip_embeddings_scale', 'desired-goals', 'dino_embeddings_scale', 
    # 'info', 'mae_embeddings_scale', 'mvp_embeddings_scale', 
    # 'observations', 'qpos', 'qvel', 'rewards', 'sam_embeddings96']
    
    env_name_data = env_name[:-3]
    env_name_data = env_name_data.replace('-', '_')
    print(env_name_data)
    
    replay_buffer.load_hdf5_dataset(f'/home/ubuntu/metaworld/{env_name_data}_data.hdf5', encoding_type=config.encoder, trajectories=trajectories)
    # replay_buffer.load_hdf5_dataset(f'/shares/bcs516/ryan/metaworld_data_lambda/{env_name_data}_data.hdf5', encoding_type=config.encoder, trajectories=trajectories)
    print('loaded dataset into replay buffer')
    
    max_action = float(env.action_space.high[0])

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Set seeds
    seed = config.seed
    # set_seed(seed)

    q_network = TwinQ(state_dim, action_dim).to(config.device)
    v_network = ValueFunction(state_dim).to(config.device)
    actor = (
        DeterministicPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout
        )
        if config.iql_deterministic
        else GaussianPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout, n_hidden=3
        )
    ).to(config.device)
    v_optimizer = torch.optim.NAdam(v_network.parameters(), lr=config.vf_lr)
    q_optimizer = torch.optim.NAdam(q_network.parameters(), lr=config.qf_lr)
    actor_optimizer = torch.optim.NAdam(actor.parameters(), lr=config.actor_lr)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "q_network": q_network,
        "q_optimizer": q_optimizer,
        "v_network": v_network,
        "v_optimizer": v_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        # IQL
        "beta": config.beta,
        "iql_tau": config.iql_tau,
        "max_steps": config.max_timesteps,
    }

    print("---------------------------------------")
    print(f"Training IQL, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize actor
    trainer = ImplicitQLearning(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    # wandb_init(asdict(config))

    evaluations = []
    max_eval_score = 0
    for t in tqdm(range(int(config.max_timesteps))):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)
        
        # wandb.log(log_dict, step=trainer.total_it)
        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            eval_scores = eval_actor(
                env,
                eval_tasks,
                actor,
                device=config.device,
                encoder = encoder,
                preprocess=preprocess,
                expert=p,
                config=config,
            )
            eval_score = eval_scores.mean()
            if eval_score >= max_eval_score:
                max_eval_score = eval_score
            # normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
            normalized_eval_score = eval_score * 100
            evaluations.append(normalized_eval_score)
            print("---------------------------------------")
            print(
                f"Evaluation over {config.n_episodes} episodes: "
                f"{eval_score:.3f} , D4RL score: {normalized_eval_score:.3f} "
                f"Critic loss: {np.mean(log_dict['q_loss'])} "
                f"Actor loss: {np.mean(log_dict['actor_loss'])} "
            )
            print("---------------------------------------")
            if config.log_path:
                with open(config.log_path, 'a') as f:
                    f.write(f'{eval_score:.3f}, {np.mean(log_dict["q_loss"])}, {np.mean(log_dict["actor_loss"])}\n')
            if t + 1 >= int(config.max_timesteps): # eval_score >= max_eval_score:
                if config.checkpoints_path is not None:
                    torch.save(
                        trainer.state_dict(),
                        os.path.join(config.checkpoints_path, f"checkpoint_{t}_{eval_score}.pt"),
                    )
            # wandb.log(
            #     {"d4rl_normalized_score": normalized_eval_score}, step=trainer.total_it
            # )

def test_render(config: TrainConfig):
    env_name = config.env

def arg_parse():
    args = argparse.ArgumentParser()
    args.add_argument('--base_encoder', type=str, default='sam')
    args.add_argument('--iql_seed', type=int, default=0)
    args.add_argument('--env_name', type=str, default='push-v2')
    args.add_argument('--traj_len', type=int, default=100)
    args.add_argument('--num_traj', type=int, default=600)
    args.add_argument('--train_steps', type=int, default=1e4)
    args.add_argument('--pos_embed', type=int, default=0)
    return args.parse_known_args()

if __name__ == "__main__":

    args, _ = arg_parse()
    
    model = args.base_encoder 
    seed = args.iql_seed
    
    # seeds = [1804573525,  983782397, 2767154603]
    # encs = ['sam', 'dino', 'clip', 'mvp', 'mae', 'r3m']
    
    conf = TrainConfig()
    conf.encoder = args.base_encoder
    conf.batch_size = 1024
    conf.seed = seed
    conf.train_trajectories = args.num_traj # 1000 traj at 150 each traj = 15k
    conf.buffer_size = conf.train_trajectories * args.traj_len # buffer size needs to be as large as the num of traj * steps per traj
    conf.eval_freq = args.train_steps # 1e4
    conf.max_timesteps = args.train_steps # 1e5
    conf.log_path = None #f'remaining_exps/push/{model}_{seed}_finalscore.log'
    conf.env = args.env_name # needs -v2
    conf.pos_embed = args.pos_embed
    conf.traj_len = args.traj_len
    train(conf)

# python -m run_metaworld_iql_fromdisk --base_encoder sam --iql_seed 42 --env_name push-v2 --traj_len 100 --num_traj 2000 --train_steps 10000 --pos_embed 1


# read off line 12 at /home/rce5022/rl/13-encoders/encoders/envs/iql/logs/push/sam_{seed}.log ~ 60000 = 12 * 5000 steps
# read off line 6 at /home/rce5022/rl/13-encoders/encoders/envs/iql/logs/push_distract/sam_{seed}_gc.log.

