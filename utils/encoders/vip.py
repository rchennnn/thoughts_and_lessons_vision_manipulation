import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
import copy
from utils.policy_net import PolicyNetwork
from vip import load_vip

# requires xformers==0.0.18

class VIPPolicy(PolicyNetwork):
    def __init__(self, device: str = 'cuda', state_size: int = 0, action_size: int = 0, layers: list = [256, 128, 64], pos_contrib: int = 16, seed: int = 0x0badbeef):
        super().__init__(device, state_size, action_size, layers, pos_contrib, seed)
        self.preprocess = T.Compose([
            # T.Lambda(lambd=lambda x : x/255.0),
            # T.Normalize(mean=_MEAN, std=_STD),
            # T.Resize((224, 224))
        ])
        # self.preprocess = T.Compose([T.Lambda(lambd=lambda x : x/255.0),])
        self.base_encoder_name = 'vip'
        self.seed = seed
    
    def _load_encoder(self):
        model : nn.Module = load_vip()
        model.eval()
        return model, 1024
    
    def act(self, image, goal, init_pos, goal_pos):
        return super().act(image, goal, init_pos, goal_pos)
    