import torchvision.models as tv_models
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
import copy
from utils.policy_net import PolicyNetwork

class MoCoV3Policy(PolicyNetwork):
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
        model = tv_models.__dict__['resnet50']()
        checkpoint = torch.load('/home/ubuntu/encoders/r-50-1000ep.pth', map_location=self.device)
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only base_encoder up to before the embedding layer
            if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.fc'):
                # remove prefix
                state_dict[k[len("module.base_encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        model.load_state_dict(state_dict, strict=False)
        model.fc = nn.Identity()
        model.eval()
        
        for p in model.parameters():
            p.requires_grad = False
        
        return model, 2048
    
    def act(self, image, goal, init_pos, goal_pos):
        return super().act(image, goal, init_pos, goal_pos)