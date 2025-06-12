from typing import Tuple
from utils.utils import sample_future_for_gcbc, flatten_first_two_dict, to_torch_dict
from utils.policy_net import PolicyNetwork

import h5py
import gymnasium_robotics
import numpy as np
import gymnasium as gym
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import imageio

from metaworld.policies.policy import Policy

SIZES ={
    'FetchReach-v2': [0,1,2],
    'reach': [0,1,2],
    'FetchPush-v2': [0,1,2,9,10,20,21,22,23,24],
    'FetchPushDistractions-v2': [0,1,2,9,10,20,21,22,23,24],
    'push': [0,1,2,9,10,20,21,22,23,24],
    'FetchPickAndPlace-v2': [0,1,2,9,10],
    'FetchPickAndPlaceDistractions-v2': [0,1,2,9,10],
    'place': [0,1,2,9,10],
}       
        
def get_data(data_dir, num_train, num_test, traj_length, device, embedding_name, name, seed):
    h = h5py.File(data_dir, 'r')
    
    obs_idxs = SIZES[name]
    
    tr_sam = {'init_embed': None, 'goal_embed': None, 'init_state': None, 'goal_state': None, 'actions': None}
    ev_sam = tr_sam.copy()
    
    tr_sam['actions'] = h['actions'][0:num_train, 0: traj_length]
    ev_sam['actions'] = h['actions'][num_train : num_train + num_test, 0: traj_length]
    
    tr_sam['init_embed'] = h[embedding_name][0:num_train, 0: traj_length]
    tr_sam['goal_embed'] = sample_future_for_gcbc(tr_sam['init_embed'], seed = seed)
    tr_sam['init_state'] = h['state'][0:num_train, 0: traj_length, :]
    tr_sam['goal_state'] = sample_future_for_gcbc(tr_sam['init_state'], seed = seed)
    
    ev_sam['init_embed'] = h[embedding_name][num_train : num_train + num_test, 0: traj_length]
    ev_sam['goal_embed'] = sample_future_for_gcbc(ev_sam['init_embed'], seed = seed)
    ev_sam['init_state'] = h['state'][num_train : num_train + num_test, 0: traj_length, :] 
    ev_sam['goal_state'] = sample_future_for_gcbc(ev_sam['init_state'], seed = seed)    
    
    ll = [tr_sam, ev_sam]
    for k, l in enumerate(ll):
        l = to_torch_dict(l, device=device, dtype=torch.float32)
        ll[k] = flatten_first_two_dict(l)
        
    return ll


    
def train_policy(init_embeds: Tuple[np.ndarray, np.ndarray], goal_embeds: Tuple[np.ndarray, np.ndarray], 
                 init_states : Tuple[np.ndarray, np.ndarray], goal_states : Tuple[np.ndarray, np.ndarray], 
                 actions: Tuple[np.ndarray, np.ndarray],
                 epochs: int, mb_size: int = 32, model: PolicyNetwork = None, rng: np.random.RandomState = None, jitter: float = 0):
    
    '''
    Tuples in params [0] for training and [1] for eval
    '''
    
    for epoch in range(epochs):
        epoch_loss = {'train': 0.0, 'test': 0.0}
        for _ in range(100):
            if rng:
                tr_idx = rng.randint(len(actions[0]), size=mb_size)
                ev_idx = rng.randint(len(actions[1]), size=mb_size)
            else:
                tr_idx = np.random.randint(len(actions[0]), size=mb_size)
                ev_idx = np.random.randint(len(actions[1]), size=mb_size)
                
            loss = model.train_probe(init_images_embs_mb=init_embeds[0][tr_idx], 
                                     goal_images_embs_mb=goal_embeds[0][tr_idx], 
                                     actions_mb=actions[0][tr_idx],
                                     init_pos_mb=init_states[0][tr_idx],
                                     goal_pos_mb=goal_states[0][tr_idx],
                                     jitter=jitter
                                     )
            epoch_loss['train'] += loss

            loss = model.test_probe(init_images_embs_mb=init_embeds[1][ev_idx], 
                                    goal_images_embs_mb=goal_embeds[1][ev_idx], 
                                    actions_mb=actions[1][ev_idx],
                                    init_pos_mb=init_states[1][ev_idx],
                                    goal_pos_mb=goal_states[1][ev_idx]
                                    )
            epoch_loss['test'] += loss
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}. Train loss: {epoch_loss["train"] / 100}. Test loss: {epoch_loss["test"] / 100}')
    
    print(f'Final: Train loss: {epoch_loss["train"] / 100}. Test loss: {epoch_loss["test"] / 100}')
    
    return model

def make_gif(img_list, name):
    imageio.mimsave(f'{name}.gif', img_list)


def play_metaworld_policy(the_policy: PolicyNetwork, env_name: str, 
                          seed: int,
                          trajectories: int,
                          size: int,
                          expert: Policy):
    from metaworld import MT1
    
    mt1 = MT1(env_name, seed=seed, num_goals= int(trajectories * 2))
    env = mt1.train_classes[env_name](render_mode='rgb_array',)
    env.mujoco_renderer.default_cam_config = {'distance': 1.75, 'azimuth': 135.0, 'elevation': -45.0,}
    policy = expert()
    log_successes = 0
    succeeded_tasks = []
    counter = 0
    while log_successes < trajectories:
        task = mt1.train_tasks[counter]
        env.set_task(task)
        obs, info = env.reset()
        exp_img = []
        for step in range(150):
            a = policy.get_action(obs)
            obs, _, _, _, info = env.step(a)
            exp_img.append(env.render())
            done = int(info['success']) == 1
        if done:
            log_successes += 1
            goal_state = env.get_arm_state()
            goal_img = env.render().transpose((2,0,1))
        else:
            counter += 1
            continue 
        
        env.set_task(task)
        obs, info = env.reset()
        img_list = []
        for step in range(150):
            img = env.render()
            img_list.append(img)
            img = img.transpose((2,0,1))
            state = env.get_arm_state()
            a_ = policy.get_action(obs)
            a = the_policy.act(image=img, goal=goal_img,
                               init_pos=state, goal_pos=goal_state)

            obs, _, _, _, info = env.step(a)
            done = int(info['success']) == 1
            if done:
                break
        if not done:
            succeeded_tasks.append(0)
            print(f'Trajectory {log_successes}. \t\t Failed \t\t Current Rate {np.mean(succeeded_tasks)}')
        else:
            succeeded_tasks.append(1)
            print(f'Trajectory {log_successes}. \t\t Success \t\t Current Rate {np.mean(succeeded_tasks)}')
        counter += 1
    return succeeded_tasks
    


def render_env(env_name, size, steps_per_eps):
    if env_name in ['FetchPush-v2', 'FetchPushDistractions-v2']:
        env = gym.make(env_name, render_mode='rgb_array', width=size, height=size, max_episode_steps=steps_per_eps, fixed_x = True, fixed_z = False)
    elif env_name in ['FetchPickAndPlace-v2', 'FetchPickAndPlaceDistractions-v2']:
        env = gym.make(env_name, render_mode='rgb_array', width=size, height=size, max_episode_steps=steps_per_eps, fixed_x = False, fixed_z = True)
    return env