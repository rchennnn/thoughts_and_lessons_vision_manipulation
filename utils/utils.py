from typing import List, Tuple
import numpy as np
import torch
import argparse
import sys
import os

def sample_future_for_gcbc(obs: np.ndarray, seed:int = None) -> Tuple[np.ndarray, np.ndarray]:
    '''
    obs can be images, embeddings, or state space.
    '''
    if seed:
        rng = np.random.RandomState(seed=seed)
    else:
        rng = np.random.RandomState(seed=None)
    future_obs = np.zeros_like(obs)
    num_traj = obs.shape[0]
    steps = obs.shape[1]
    for traj in range(num_traj):
        for step in range(steps):
            future_idx = steps - 1 # rng.randint(step, steps)
            future_obs[traj, step] = obs[traj, future_idx]
            
    return future_obs

def flatten_first_two(np_arr):
    return np_arr.reshape((np_arr.shape[0]*np_arr.shape[1], *np_arr.shape[2:]))

def flatten_first_two_dict(d):
    for k, v in d.items():
        d[k] = flatten_first_two(v)
    return d

def parse_bool_string(val: str):
    return True if val == 'True' else False

def to_torch(tensor_list: List[np.ndarray], device: torch.device, dtype: np.dtype):
    for i in range(len(tensor_list)):
        tensor_list[i] = torch.from_numpy(tensor_list[i]).to(device=device, dtype=dtype).squeeze()
    return tensor_list

def to_torch_dict(d: dict, device: torch.device, dtype: np.dtype):
    for k, a in d.items():
        d[k] = torch.from_numpy(a).to(device=device, dtype=dtype)
    return d

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--base_encoder', type=str, help='Type of base encoder', default='sam')
    args.add_argument('--num_train', type=int, help='Number of training trajectories', default=100)
    args.add_argument('--num_eval', type=int, help='Number of eval trajectories', default=100)
    args.add_argument('--seed', type=int, help='Seed for selecting data set', default=None)
    args.add_argument('--pos_contrib', type=int, help='Size of position embedding space', default=16)
    args.add_argument('--silent', type=parse_bool_string, help='Show training and env stats', default=False)
    args.add_argument('--img_size', type=int, help='How large should images be', default=224)
    args.add_argument('--jitter', type=float, help='Jitter scale', default=None)
    args.add_argument('--device', type=int, help='Devices for lightning to use', default = 1)
    args.add_argument('--num_epochs', type=int, help='Epochs to train', default=101)
    args.add_argument('--env_name', type=str, help='Which environment', default='push')
    
    return args.parse_args()

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__