from collections import OrderedDict
from typing import Iterable, Iterator, List, Optional, Sized, Union
import h5py
import numpy as np
from torch.utils.data import Dataset
import torch
import gymnasium as gym
from torch.utils.data.sampler import Sampler, BatchSampler
import matplotlib.pyplot as plt

_ENV_NAME = {
    'push': 'FetchPush-v2',
    'place': 'FetchPickAndPlace-v2',
    'door': 'AdroitHandDoor-v1',
    'hammer': 'AdroitHandHammer-v1'
}

_TARGETS = {
    'push': 'goal',
    'place': 'goal',
    'door': 'door_body_pos',
    'hammer': 'board_pos'
}

_SIZES ={
    'AdroitHandDoor-v1': 27,
    'door': 27,
    'AdroitHandHammer-v1': 27,
    'hammer': 27,
    'AdroitHandRelocate-v1': 30,
    'relocate': 27,
    'FetchReach-v2': [0,1,2],
    'reach': [0,1,2],
    'FetchPush-v2': [0,1,2,9,10,20,21,22,23,24],
    'push': [0,1,2,9,10,20,21,22,23,24],
    'FetchPickAndPlace-v2': [0,1,2,9,10], #[0,1,2,9,10]
    'place': [0,1,2,9,10],#[0,1,2,9,10,20,21,22,23,24],
}

class EnvironmentImagesData(Dataset):
    def __init__(self, env_name: str, image_size: int, data_path: str, cache_size: int, number_samples: int) -> None:
        super().__init__()
        self.env_short_name = env_name
        self.obs_size = _SIZES[self.env_short_name]
        self.env_name = _ENV_NAME[env_name]
        self.env = gym.make(self.env_name, render_mode = 'rgb_array', height=image_size, width=image_size)
        self.env.reset()
        self.data_path = data_path
        with h5py.File(data_path, 'r') as h:
            if self.env_short_name in ['door', 'hammer']:
                self.num_traj = h['actions'].shape[0]
                self.num_step = h['actions'].shape[1]
                
            elif self.env_short_name in ['push', 'place']:
                self.num_traj = h['acts'].shape[0]
                self.num_step = h['acts'].shape[1]
                
        self.number_samples = number_samples
        self.cache_size = cache_size
        self.cache = OrderedDict()
        _ = self._stupid() # required or image will not be centered properly (mujoco rendering quirk)

    def _stupid(self, trajectory: int = 0):
        self.env.reset()
        with h5py.File(self.data_path, 'r') as h:
            init_qpos = h['qpos'][trajectory, 0]
            init_qvel = h['qvel'][trajectory, 0]
            if self.env_short_name in ['push', 'place']:
                init_target = h['desired-goals'][trajectory, 0]
            elif self.env_short_name in ['door', 'hammer']:
                init_target = h['targets'][trajectory, 0]
        
        init_state_dict = {'qpos': init_qpos, 'qvel': init_qvel, _TARGETS[self.env_short_name] : init_target}
        self.env.set_env_state(init_state_dict)
        init_img = self.env.render()
        
        if isinstance(init_img, tuple):
            init_img = init_img[2].transpose((2,0,1))
        return init_img             
    
    def __getitem__(self, index):
        if index in self.cache:
            return self.cache[index]
        else:
            trajectory = index // self.num_step
            step = index % self.num_step
                        
            _ = self._stupid(trajectory)
            
            with h5py.File(self.data_path, 'r') as h:
                if isinstance(self.obs_size, int):
                    init_state_space = torch.from_numpy(h['observations'][trajectory, step, 0: self.obs_size]).to('cuda', dtype=torch.float32)
                    goal_state_space = torch.from_numpy(h['observations'][trajectory, self.num_step - 1, 0: self.obs_size]).to('cuda', dtype=torch.float32)
                elif isinstance(self.obs_size, list):
                    init_state_space = torch.from_numpy(h['obs'][trajectory, step, self.obs_size]).to('cuda', dtype=torch.float32)
                    goal_state_space = torch.from_numpy(h['obs'][trajectory, self.num_step - 1, self.obs_size]).to('cuda', dtype=torch.float32)
                    
                init_qpos = h['qpos'][trajectory, step]
                init_qvel = h['qvel'][trajectory, step]
                goal_qpos = h['qpos'][trajectory, -1]
                goal_qvel = h['qvel'][trajectory, -1]
                
                if self.env_short_name in ['push', 'place']:
                    init_target = h['desired-goals'][trajectory, step]
                    goal_target = h['desired-goals'][trajectory, -1]
                    
                elif self.env_short_name in ['door', 'hammer']: 
                    init_target = h['targets'][trajectory, step]
                    goal_target = h['targets'][trajectory, -1]
                
                if self.env_short_name in ['push', 'place']:
                    action = torch.from_numpy(h['acts'][trajectory, step]).to('cuda', dtype=torch.float32)
                elif self.env_short_name in ['door', 'hammer']:
                    action = torch.from_numpy(h['actions'][trajectory, step]).to('cuda', dtype=torch.float32)
            
            init_state_dict = {'qpos': init_qpos, 'qvel': init_qvel, _TARGETS[self.env_short_name] : init_target}
            goal_state_dict = {'qpos': goal_qpos, 'qvel': goal_qvel, _TARGETS[self.env_short_name] : goal_target}
            
            self.env.reset()
            self.env.set_env_state(init_state_dict)
            init_img = self.env.render()
            
            self.env.reset()
            self.env.set_env_state(goal_state_dict)
            goal_img = self.env.render()
            
            if isinstance(init_img, tuple):
                init_img = init_img[2].transpose((2,0,1))
                goal_img = goal_img[2].transpose((2,0,1))
            else:
                init_img = init_img.transpose((2,0,1))
                goal_img = goal_img.transpose((2,0,1))
            
            torch_init_img = torch.from_numpy(init_img.copy()).to('cuda', dtype=torch.float32)
            torch_goal_img = torch.from_numpy(goal_img.copy()).to('cuda', dtype=torch.float32)
            
            if len(self.cache) >= self.cache_size:
                self.cache.popitem(False)
            
            self.cache[index] = torch_init_img, torch_goal_img, action, init_state_space, goal_state_space
            
        return torch_init_img, torch_goal_img, action, init_state_space, goal_state_space
    
    def __len__(self):
        return self.number_samples

class MySampler(Sampler):
    def __init__(self, data_source, num_batches, batch_size, seed) -> None:
        super().__init__(data_source)
        self.largest = len(data_source)
        rng = np.random.RandomState(seed=seed)
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.indices = rng.randint(self.largest, size=(num_batches, batch_size))
        self.indices = np.sort(self.indices, axis = 1)
        
    def __iter__(self):
        for i in range(self.num_batches):
            yield self.indices[i]
    
    def __len__(self):
        return self.largest         
    

class EnvironmentEmbeddingsData(Dataset):
    def __init__(self, env_name: str, data_path: str, embedding_name:str, number_samples: int, cache_size: int = 100) -> None:
        super().__init__()
        self.env_short_name = env_name
        self.obs_size = _SIZES[self.env_short_name]
        self.data = h5py.File(data_path, 'r')
        
        if self.env_short_name in ['door', 'hammer']:
            self.num_traj = self.data['actions'].shape[0]
            self.num_step = self.data['actions'].shape[1]
        elif self.env_short_name in ['push', 'place']:
            self.num_traj = self.data['acts'].shape[0]
            self.num_step = self.data['acts'].shape[1]
            
        self.number_samples = number_samples
        self.embedding_name = embedding_name
        self.cache_size = cache_size
        self._cache()
        
    def _cache(self):
        self.cache = OrderedDict()
        
    def __getitem__(self, index):
        trajectory = index // self.num_step
        step = index % self.num_step
                
        if index in self.cache:
            return self.cache[index]
        else:
            if len(self.cache) >= self.cache_size:
                self.cache.popitem(False)
            
            if isinstance(self.obs_size, int):
                init_state_space = torch.from_numpy(self.data['observations'][trajectory, step, 0: self.obs_size]).to('cuda', dtype=torch.float32)
                goal_state_space = torch.from_numpy(self.data['observations'][trajectory, -1, 0: self.obs_size]).to('cuda', dtype=torch.float32)
            elif isinstance(self.obs_size, list):
                init_state_space = torch.from_numpy(self.data['obs'][trajectory, step, self.obs_size]).to('cuda', dtype=torch.float32)
                goal_state_space = torch.from_numpy(self.data['obs'][trajectory, -1, self.obs_size]).to('cuda', dtype=torch.float32)
            
            init_embedding = torch.from_numpy(self.data[self.embedding_name][trajectory, step]).to('cuda', dtype=torch.float32)
            goal_embedding = torch.from_numpy(self.data[self.embedding_name][trajectory, -1]).to('cuda', dtype=torch.float32)
            
            if self.env_short_name in ['door', 'hammer']:
                action = torch.from_numpy(self.data['actions'][trajectory, step]).to('cuda', dtype=torch.float32)
            if self.env_short_name in ['push', 'place']:
                action = torch.from_numpy(self.data['acts'][trajectory, step]).to('cuda', dtype=torch.float32)

            self.cache[index] = (init_embedding, goal_embedding, action, init_state_space, goal_state_space)
        
        return init_embedding, goal_embedding, action, init_state_space, goal_state_space, #qvel, qpos, target
    
    def __len__(self):
        return self.number_samples