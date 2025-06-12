import torch
import torch.nn as nn
import torchvision.transforms as T
from utils.policy_net import PolicyNetwork
import copy

import torch.nn as nn
import torch.nn.parallel as parallel

import torch
import torch.nn as nn

import utils.encoders.rn.builder as builder

class ObjRNPolicy(PolicyNetwork):
    def __init__(self, device: str = 'cuda', state_size: int = 0, action_size: int = 0, layers: list = [256, 128, 64], pos_contrib: int = 16, seed: int = 0x0badbeef):
        super().__init__(device, state_size, action_size, layers, pos_contrib, seed)
        _OPENAI_MEAN = (0.48145466, 0.4578275, 0.40821073)
        _OPENAI_STD = (0.26862954, 0.26130258, 0.27577711)
        self.preprocess = T.Compose([
            T.Lambda(lambd=lambda x: x/255.0),
            T.Normalize(mean=_OPENAI_MEAN, std=_OPENAI_STD),
            T.Resize((224, 224))
        ])
        # self.preprocess = T.Compose([T.Lambda(lambd=lambda x : x/255.0),])
        self.base_encoder_name = 'obj_rn'
    
    def _load_encoder(self):
        model = builder.BuildAutoEncoder('resnet50')
        model.eval()
        
        checkpoint = torch.load('/home/ubuntu/encoders/objects365-resnet50.pth')
        model_dict = model.state_dict()
        model_dict.update(checkpoint['state_dict'])            
        model.load_state_dict(model_dict, strict=False)
        del checkpoint
        
        return model, 2048
    
    def act(self, image, goal, init_pos, goal_pos):
        return super().act(image, goal, init_pos, goal_pos)
    
if __name__ == '__main__':
    o = ObjRNPolicy()
    # img = torch.randn(8, 3, 224, 224).to('cuda')
    # print(o.encoder(img).shape)