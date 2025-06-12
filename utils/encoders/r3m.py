import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
import copy
from utils.policy_net import PolicyNetwork
from r3m import load_r3m, load_r3m_local

# requires xformers==0.0.18

class R3MPolicy(PolicyNetwork):
    def __init__(self, device: str = 'cuda', state_size: int = 0, action_size: int = 0, layers: list = [256, 128, 64], pos_contrib: int = 16, seed: int = 0x0badbeef):
        super().__init__(device, state_size, action_size, layers, pos_contrib, seed)
        self.preprocess = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            # T.ToTensor() # ToTensor() divides by 255
        ])
        # self.preprocess = T.Compose([T.Lambda(lambd=lambda x : x/255.0),])
        self.base_encoder_name = 'r3m'
        self.seed = seed
    
    def _load_encoder(self):
        # r3m is now broken. need to modify to pull pt files locally
        try:
            r3m = load_r3m("resnet50") # resnet18, resnet34
            r3m.eval()
            return r3m, 2048
        except:
            print("R3M encoder not found. Loading from local path.")
            r3m = load_r3m_local("/home/ubuntu/encoders/r3m_model.pt", "/home/ubuntu/encoders/r3m_config.yaml")
            r3m.eval()
            return r3m, 2048
    
    def act(self, image, goal, init_pos, goal_pos):
        return super().act(image, goal, init_pos, goal_pos)
    