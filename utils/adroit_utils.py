from typing import Tuple, List, Optional
from utils.utils import sample_future_for_gcbc, flatten_first_two_dict, to_torch_dict
from utils.policy_net import PolicyNetwork
from segment_anything.modeling import ImageEncoderViT
from experts.adroit import AdroitPlayer

import h5py
import numpy as np
import gymnasium as gym
import torch
import matplotlib.pyplot as plt
import imageio
import time
        
SIZES ={
    'AdroitHandDoor-v1': 27,
    'door': 27,
    'AdroitHandHammer-v1': 27,
    'hammer': 27,
    'AdroitHandRelocate-v1': 30,
    'relocate': 27,
}

_ENV_NAME = {
    'door' : 'AdroitHandDoor-v1',
    'hammer': 'AdroitHandHammer-v1'
}

_TARGET_NAME = {
    'door': 'door_body_pos',
    'hammer': 'board_pos'
}

def get_data(data_dir, num_train, num_test, traj_length, device, embedding_name, seed, env_name):
    h = h5py.File(data_dir, 'r')
    obs_size = SIZES[env_name]
    
    tr_sam = {'init_embed': None, 'goal_embed': None, 'init_state': None, 'goal_state': None, 'actions': None}
    ev_sam = tr_sam.copy()
    
    tr_sam['actions'] = h['actions'][0:num_train, 0: traj_length]
    ev_sam['actions'] = h['actions'][num_train : num_train + num_test, 0: traj_length]
    
    tr_sam['init_embed'] = h[embedding_name][0:num_train, 0: traj_length]
    tr_sam['goal_embed'] = sample_future_for_gcbc(tr_sam['init_embed'], seed = seed)
    tr_sam['init_state'] = h['observations'][0:num_train, 0: traj_length, 0:obs_size]
    tr_sam['goal_state'] = sample_future_for_gcbc(tr_sam['init_state'], seed = seed)
    
    ev_sam['init_embed'] = h[embedding_name][num_train : num_train + num_test, 0: traj_length]
    ev_sam['goal_embed'] = sample_future_for_gcbc(ev_sam['init_embed'], seed = seed)
    ev_sam['init_state'] = h['observations'][num_train : num_train + num_test, 0: traj_length, 0:obs_size]
    ev_sam['goal_state'] = sample_future_for_gcbc(ev_sam['init_state'], seed = seed)    
    
    ll = [tr_sam, ev_sam]
    for k, l in enumerate(ll):
        l = to_torch_dict(l, device=device, dtype=torch.float32)
        ll[k] = flatten_first_two_dict(l)
        
    return ll

def get_image_data(data_dir, num_train, num_test, traj_length, device, seed, env_name, image_size):
    import gymnasium as gym 
    
    h = h5py.File(data_dir, 'r')
    obs_size = SIZES[env_name]
    
    tr_sam = {'init_image': None, 'goal_image': None, 'init_state': None, 'goal_state': None, 'actions': None}    
    ev_sam = tr_sam.copy()
    
    tr_sam['actions'] = h['actions'][0:num_train, 0: traj_length]
    ev_sam['actions'] = h['actions'][num_train : num_train + num_test, 0: traj_length]

    tr_sam['init_qpos'] = h['qpos'][0:num_train, 0: traj_length]
    tr_sam['init_qvel'] = h['qvel'][0:num_train, 0: traj_length]
    tr_sam['init_target'] = h['targets'][0:num_train, 0: traj_length]
    tr_sam['goal_qpos'] = sample_future_for_gcbc(tr_sam['init_qpos'], seed = seed)    
    tr_sam['goal_qvel'] = sample_future_for_gcbc(tr_sam['init_qvel'], seed = seed)
    tr_sam['goal_target'] = sample_future_for_gcbc(tr_sam['init_target'], seed = seed)
    tr_sam['init_state'] = h['observations'][0:num_train, 0: traj_length, 0:obs_size]
    tr_sam['goal_state'] = sample_future_for_gcbc(tr_sam['init_state'], seed = seed)
        
    tr_sam['init_image'] = np.zeros((num_train, traj_length, 3, image_size, image_size))
    tr_sam['goal_image'] = np.zeros((num_train, traj_length, 3, image_size, image_size))
    
    ev_sam['init_qpos'] = h['qpos'][num_train : num_train + num_test, 0: traj_length]
    ev_sam['init_qvel'] = h['qvel'][num_train : num_train + num_test, 0: traj_length]
    ev_sam['init_target'] = h['targets'][num_train : num_train + num_test, 0: traj_length]
    ev_sam['goal_qpos'] = sample_future_for_gcbc(ev_sam['init_qpos'], seed = seed)    
    ev_sam['goal_qvel'] = sample_future_for_gcbc(ev_sam['init_qvel'], seed = seed)
    ev_sam['goal_target'] = sample_future_for_gcbc(ev_sam['init_target'], seed = seed)
    ev_sam['init_state'] = h['observations'][num_train : num_train + num_test, 0: traj_length, 0:obs_size]
    ev_sam['goal_state'] = sample_future_for_gcbc(ev_sam['init_state'], seed = seed)
        
    ev_sam['init_image'] = np.zeros((num_test, traj_length, 3, image_size, image_size))
    ev_sam['goal_image'] = np.zeros((num_test, traj_length, 3, image_size, image_size))
    
    env = gym.make(_ENV_NAME[env_name], render_mode = 'rgb_array', width = image_size, height=image_size)
    env.reset()
    for traj_num in range(num_train):
        init_state_dict = {'qpos': tr_sam['init_qpos'][traj_num, 0], 'qvel': tr_sam['init_qvel'][traj_num, 0],
                               _TARGET_NAME[env_name]: tr_sam['init_target'][traj_num, 0]}
        env.set_env_state(init_state_dict)
        for frame_num in range(traj_length):
            init_state_dict = {'qpos': tr_sam['init_qpos'][traj_num, frame_num], 'qvel': tr_sam['init_qvel'][traj_num, frame_num],
                               _TARGET_NAME[env_name]: tr_sam['init_target'][traj_num, frame_num]}
            goal_state_dict = {'qpos': tr_sam['goal_qpos'][traj_num, frame_num], 'qvel': tr_sam['goal_qvel'][traj_num, frame_num], 
                               _TARGET_NAME[env_name]: tr_sam['goal_target'][traj_num, frame_num]}
            env.set_env_state(init_state_dict)
            init_img = env.render().transpose((2,0,1))
            env.set_env_state(goal_state_dict)
            goal_img = env.render().transpose((2,0,1))            
            tr_sam['init_image'][traj_num, frame_num] = init_img
            tr_sam['goal_image'][traj_num, frame_num] = goal_img

            
    env.reset()
    for traj_num in range(num_test):
        for frame_num in range(traj_length):
            init_state_dict = {'qpos': ev_sam['init_qpos'][traj_num, frame_num], 'qvel': ev_sam['init_qvel'][traj_num, frame_num],
                               _TARGET_NAME[env_name]: ev_sam['init_target'][traj_num, frame_num]}
            goal_state_dict = {'qpos': ev_sam['goal_qpos'][traj_num, frame_num], 'qvel': ev_sam['goal_qvel'][traj_num, frame_num], 
                               _TARGET_NAME[env_name]: ev_sam['goal_target'][traj_num, frame_num]}
            env.set_env_state(init_state_dict)
            init_img = env.render().transpose((2,0,1))
            env.set_env_state(goal_state_dict)
            goal_img = env.render().transpose((2,0,1))
            ev_sam['init_image'][traj_num, frame_num] = init_img
            ev_sam['goal_image'][traj_num, frame_num] = goal_img

    
    ll = [tr_sam, ev_sam]
    for k, l in enumerate(ll):
        l = to_torch_dict(l, device=device, dtype=torch.float32)
        ll[k] = flatten_first_two_dict(l)
    
    # tr_sam = to_torch_dict(tr_sam, device=device, dtype=torch.float32)
    # tr_sam = flatten_first_two_dict(tr_sam)
    
    return ll
    

def train_probe(init_embeds: Tuple[np.ndarray, np.ndarray], goal_embeds: Tuple[np.ndarray, np.ndarray], 
                init_states : Tuple[np.ndarray, np.ndarray], goal_states : Tuple[np.ndarray, np.ndarray], 
                actions: Tuple[np.ndarray, np.ndarray],
                epochs: int, mb_size: int = 32, model: PolicyNetwork = None,
                jitter: Optional[float] = None,
                rng : np.random.RandomState = None):
    
    '''
    Tuples in params [0] for training and [1] for eval
    '''
    
    print('Training...')
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
                                     jitter=jitter, epoch = epoch
                                    )
            epoch_loss['train'] += loss

            loss = model.test_probe(init_images_embs_mb=init_embeds[1][ev_idx], 
                                    goal_images_embs_mb=goal_embeds[1][ev_idx], 
                                    actions_mb=actions[1][ev_idx],
                                    init_pos_mb=init_states[1][ev_idx],
                                    goal_pos_mb=goal_states[1][ev_idx],
                                   )
            epoch_loss['test'] += loss
        
        if epoch % 1 == 0:
            print(f'Epoch {epoch}. Train loss: {epoch_loss["train"] / 100}. Test loss: {epoch_loss["test"] / 100}')
    
    print(f'Final: Train loss: {epoch_loss["train"] / 100}. Test loss: {epoch_loss["test"] / 100}')
            
    return model, epoch_loss['train'].item()/100, epoch_loss['test'].item()/100

def train_images(model: PolicyNetwork, 
                 init_images: Tuple[np.ndarray, np.ndarray], goal_images:  Tuple[np.ndarray, np.ndarray], 
                 init_states:  Tuple[np.ndarray, np.ndarray], goal_states:  Tuple[np.ndarray, np.ndarray], 
                 actions: Tuple[np.ndarray, np.ndarray], 
                 epochs: int, mb_size: int, rng):
    print('Training...')
    for epoch in range(epochs):
        epoch_loss = {'train': 0.0, 'test': 0.0}
        start = time.time()
        for _ in range(100):
            if rng:
                tr_idx = torch.from_numpy(np.sort(rng.randint(len(actions[0]), size=mb_size))).to('cuda')
                ev_idx = torch.from_numpy(np.sort(rng.randint(len(actions[1]), size=mb_size))).to('cuda')
            else:
                tr_idx = np.random.randint(len(actions[0]), size=mb_size)
                ev_idx = np.random.randint(len(actions[1]), size=mb_size)
            loss = model.train_images(image_init_mb=torch.index_select(init_images[0], 0, tr_idx).to('cuda'), 
                                      image_goal_mb=torch.index_select(goal_images[0], 0, tr_idx).to('cuda'), 
                                      actions_mb=torch.index_select(actions[0], 0, tr_idx).to('cuda'),
                                      init_pos_mb=torch.index_select(init_states[0], 0, tr_idx).to('cuda'),
                                      goal_pos_mb=torch.index_select(goal_states[0], 0, tr_idx).to('cuda')
                                     )
            epoch_loss['train'] += loss
            
            loss = model.test_images(image_init_mb=torch.index_select(init_images[1], 0, ev_idx).to('cuda'), 
                                     image_goal_mb=torch.index_select(goal_images[1], 0, ev_idx).to('cuda'), 
                                     actions_mb=torch.index_select(actions[1], 0, ev_idx).to('cuda'),
                                     init_pos_mb=torch.index_select(init_states[1], 0, ev_idx).to('cuda'),
                                     goal_pos_mb=torch.index_select(goal_states[1], 0, ev_idx).to('cuda')
                                    )
            epoch_loss['test'] += loss
            
        end = time.time()
        if epoch % 1 == 0:
            print(f'Epoch {epoch}. Train loss: {epoch_loss["train"] / 100}. Test loss: {epoch_loss["test"] / 100}. Epoch took {(end - start)/60} mins')
    
    print(f'Final: Train loss: {epoch_loss["train"] / 100}, Test loss: {epoch_loss["test"] / 100}')      
    return model, epoch_loss['train'].item()/100, epoch_loss['test'].item()/100

def train_images_dataloader(model: PolicyNetwork, 
                 train_loader, test_loader,
                 epochs: int):
    
    for epoch in range(epochs):
        num_total_train, num_total_test = 0, 0
        epoch_loss = {'train': 0.0, 'test': 0.0}
        for batch, data in enumerate(train_loader):
            start = time.time()
            init_img, goal_img, action, init_state_space, goal_state_space = data 
            batch_size = action.shape[0]
            loss = model.train_images(image_init_mb=init_img, 
                                      image_goal_mb=goal_img, 
                                      init_pos_mb=init_state_space, 
                                      goal_pos_mb=goal_state_space, 
                                      actions_mb=action)
            num_total_train += batch_size
            epoch_loss['train'] += loss 
            end = time.time()
            print(batch, end-start)
        
        for batch, data in enumerate(test_loader):
            init_img, goal_img, action, init_state_space, goal_state_space = data 
            loss = model.test_images(image_init_mb=init_img, 
                                     image_goal_mb=goal_img, 
                                     init_pos_mb=init_state_space, 
                                     goal_pos_mb=goal_state_space, 
                                     actions_mb=action)
            num_total_test += batch_size
            epoch_loss['test'] += loss 
            
        
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}. Train loss: {epoch_loss["train"] / num_total_train}. Test loss: {epoch_loss["test"] / num_total_test}')
            
    return model, epoch_loss['train'].item()/num_total_train, epoch_loss['test'].item()/num_total_train

def make_gif(img_list, name):
    imageio.mimsave(f'{name}.gif', img_list)

def play_imitation_policy(the_policy: PolicyNetwork, env_name: str, 
                          steps_per_eps: int, eps: int, 
                          expert_path: str, 
                          state_keys: List[str],
                          device: torch.device):

    print(f'Playing {env_name}')
    
    size = 96 if the_policy.base_encoder_name == 'sam' else 224
    
    env = gym.make(env_name, render_mode='rgb_array', width=size, height=size, max_episode_steps=steps_per_eps)

    r_env = gym.make(env_name, render_mode='rgb_array', width=224, height=224, max_episode_steps=steps_per_eps)

    expert_env = AdroitPlayer(env_name=env_name,
                              expert_policy_path=expert_path,
                              max_steps_per_eps=steps_per_eps,
                              device=device,
                              size=size)
    successes = []
    success = 0
    obs_size = SIZES[env_name]
        
    for ep in range(eps):
        obs, _ = env.reset(seed=ep * 0x0badbeef)
        _, _ = r_env.reset()
        curr_state = env.get_env_state()
        expert_env.set_env_state(state_dict={k: curr_state[k] for k in state_keys})
        goal_obs, goal_img = expert_env.play()
        img_list = []
        for step in range(steps_per_eps):
            img = env.render()
            state_dict = env.get_env_state()
            state_dict.pop('target_pos', None)
            r_env.set_env_state(state_dict)
            rendered_img = r_env.render()
            img_list.append(rendered_img)
            img = img.transpose((2,0,1))
            action = the_policy.act(image=img, goal=goal_img, init_pos=obs[0:obs_size], goal_pos=goal_obs[0:obs_size])
            obs, reward, _, _, info = env.step(action)
            # if info['success']:
            #     break
        make_gif(img_list=img_list, name=f'gifs/{env_name}/{the_policy.base_encoder_name}/{ep}')
        if info['success']:
            successes.append(1)
            success += 1
            print(f'Episode {ep}. \t\t Success \t\t Current Success Rate {np.mean(successes)} \t\t {success}/{len(successes)}')
        else:
            successes.append(0)
            print(f'Episode {ep}. \t\t Took too long \t\t Current Success Rate {np.mean(successes)} \t\t {success}/{len(successes)}')

    return successes

        
        