import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
import copy
from utils.policy_net import PolicyNetwork

# requires xformers==0.0.18

class DinoV2Policy(PolicyNetwork):
    def __init__(self, device: str = 'cuda', state_size: int = 0, action_size: int = 0, layers: list = [256, 128, 64], pos_contrib: int = 16, seed: int = 0x0badbeef):
        super().__init__(device, state_size, action_size, layers, pos_contrib, seed)
        _MEAN = (0.485, 0.456, 0.406)
        _STD = (0.229, 0.224, 0.225)
        self.preprocess = T.Compose([
            T.Lambda(lambd=lambda x : x/255.0),
            T.Normalize(mean=_MEAN, std=_STD),
            T.Resize((224, 224))
        ])
        # self.preprocess = T.Compose([T.Lambda(lambd=lambda x : x/255.0),])
        self.base_encoder_name = 'dino'
        self.seed = seed
    
    def _load_encoder(self):
        model : nn.Module = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg', pretrained=False).to('cuda')
        model.load_state_dict(torch.load('/home/ubuntu/encoders/dinov2_vitb14_reg4_pretrain.pth'))
        model.eval()
        return model, 768 # 1024
    
    def act(self, image, goal, init_pos, goal_pos):
        return super().act(image, goal, init_pos, goal_pos)