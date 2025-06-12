import torch
import torch.nn as nn
import copy
from typing import Optional

from utils.policy_net import PolicyNetwork

import torchvision.transforms as T
from segment_anything import sam_model_registry

class SAMPolicy(PolicyNetwork):
    def __init__(self, device: str = 'cuda', state_size: int = 0, action_size: int = 0, layers: list = [256, 128, 64], pos_contrib: int = 16, seed: int = 0x0badbeef):
        super().__init__(device, state_size, action_size, layers, pos_contrib, seed)
        self.preprocess = T.Compose([
            #T.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]), # NO normalization works better with SAM
        ])
        self.base_encoder_name = 'sam'
        self.embeddings = None
        self.encoder.blocks[11].register_forward_hook(self._register_hook())
        
    def _register_hook(self):
        def hook(module, input, output):
            self.embeddings = output
        return hook
        
    def _load_encoder(self):
        sam = sam_model_registry['vit_b'](checkpoint='/home/ubuntu/encoders/sam_vit_b_01ec64.pth')
        return sam.image_encoder, 50176 #9216 #50176 # 9216
    
    def act(self, image, goal, init_pos, goal_pos):
        return super().act(image=image, goal=goal, init_pos=init_pos, goal_pos=goal_pos)
