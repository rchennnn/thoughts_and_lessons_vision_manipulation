import torch
import torch.nn as nn
import torchvision.transforms as T
from utils.policy_net import PolicyNetwork
import copy

import mvp

# Use the following normalizations
# transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

class MVPPolicy(PolicyNetwork):
    def __init__(self, device: str = 'cuda', state_size: int = 0, action_size: int = 0, layers: list = [256, 128, 64], pos_contrib: int = 16, seed: int = 0x0badbeef):
        super().__init__(device, state_size, action_size, layers, pos_contrib, seed)
        self.preprocess = T.Compose([
            T.Lambda(lambd=lambda x : x/255.0),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.Resize((224,224))
        ])
        # self.preprocess = T.Compose([T.Lambda(lambd=lambda x : x/255.0),])
        self.base_encoder_name = 'mvp'
    
    def _load_encoder(self):
        model = mvp.load('vitb-mae-egosoup').to('cuda')
        model.freeze()
        return model, 768
    
    def act(self, image, goal, init_pos, goal_pos):
        return super().act(image=image, goal=goal, init_pos=init_pos, goal_pos=goal_pos)
    