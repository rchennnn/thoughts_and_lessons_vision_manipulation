import torch
import torch.nn as nn
import torchvision.transforms as T
from utils.policy_net import PolicyNetwork
import copy

import open_clip
from utils.encoders.clip_lora_adaptation import PlainMultiHeadAttention

class CLIPPolicy(PolicyNetwork):
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
        self.base_encoder_name = 'clip'
    
    def _load_encoder(self):
        # model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='/shares/bcs516/ryan/encoders/clip_vit_b.pth', device='cuda')
        model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='/home/ubuntu/encoders/clip_vit_b.pth', device='cuda')
        model.eval()
        # remember to self.preprocess batch images with clip
        return model.visual, 512
    
    def act(self, image, goal, init_pos, goal_pos):
        return super().act(image, goal, init_pos, goal_pos)