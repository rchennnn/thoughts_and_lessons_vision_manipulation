import torch
import torch.nn as nn
import torchvision.transforms as T
from utils.policy_net import PolicyNetwork
import copy

import utils.encoders.vit.vit as vit_models
from utils.encoders.vit.vit import interpolate_pos_embed

# Use the following normalizations
# transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

class MAEPolicy(PolicyNetwork):
    def __init__(self, device: str = 'cuda', state_size: int = 0, action_size: int = 0, layers: list = [256, 128, 64], pos_contrib: int = 16, seed: int = 0x0badbeef):
        super().__init__(device, state_size, action_size, layers, pos_contrib, seed)
        _MEAN = [0.485, 0.456, 0.406]
        _STD=[0.229, 0.224, 0.225]
        self.preprocess = T.Compose([
            T.Lambda(lambd=lambda x: x/255.0),
            T.Normalize(mean=_MEAN, std=_STD),
            T.Resize((224, 224))
        ])
        # self.preprocess = T.Compose([T.Lambda(lambd=lambda x : x/255.0),])
        self.base_encoder_name = 'mae'
    
    def _load_encoder(self):
        # load base model
        model = vit_models.vit_base_patch16()
        # checkpoint = torch.load('/shares/bcs516/ryan/encoders/mae_pretrain_vit_base.pth', map_location='cuda')
        checkpoint = torch.load('/home/ubuntu/encoders/mae_pretrain_vit_base.pth', map_location='cuda')
        # weights are under "model" key
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        # remove head weight and biases
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                del checkpoint_model[k]
        # process pos embed        
        interpolate_pos_embed(model, checkpoint_model)
        model.load_state_dict(checkpoint['model'], strict=False)
        # make head return embeddings
        model.head = torch.nn.Identity()
        return model, 768
    
    def act(self, image, goal, init_pos, goal_pos):
        return super().act(image, goal, init_pos, goal_pos)
    