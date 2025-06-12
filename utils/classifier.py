from typing import Dict, Optional
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.cuda.amp import autocast, grad_scaler

from utils.encoders import SAMPolicy, MVPPolicy, MAEPolicy, DinoV2Policy, CLIPPolicy
import numpy as np
from segment_anything import sam_model_registry
from segment_anything.modeling import ImageEncoderViT

from utils.net_utils import CustomMLP

MODELS = {
    'sam' : SAMPolicy,
    'mvp' : MVPPolicy,
    'mae' : MAEPolicy,
    'dino' : DinoV2Policy,
    'clip' : CLIPPolicy
}

class ClassifierNetwork(nn.Module):
    def __init__(self, device : str ='cuda', encoder: str = 'sam', num_classes: int = 0, layers: list = [128,64,32], seed: int= 0x0badbeef):
        super().__init__()
        
        self.device = device
        self.preprocess = None
        self.seed = seed
        self.encoder_name = encoder

        self.encoder, self.embed_size, self.preprocess = self._load_encoder(encoder)
        for p in self.encoder.parameters():
            if p.requires_grad:
                p.requires_grad = False

        torch.manual_seed(self.seed)
        
        head_layers = layers

        self.model = nn.ModuleDict({
            'ENCODE': self.encoder,
            'MLPE': CustomMLP(layers=[self.embed_size, 512, 256], response_size=128, seed=self.seed,), # [embed, 256, 128] and 64
            'MLPFIN': CustomMLP(layers=head_layers, response_size=num_classes, seed=self.seed, layer_norm=True)
        }).to(device)
        
        print(f'Head layers: {head_layers}')
        
        # orthogonal init for hand envs. use default for fetch envs
        self.model.MLPE.apply(self._initialize_weights)
        self.model.MLPFIN.apply(self._initialize_weights)
        

        self.criterion = nn.MSELoss()
        self.lr = 0.0008 # 0.0008, 0.0003
        self.optimizer = torch.optim.AdamW(
            [   
                # {'params': self.model.ENCODE.parameters()},
                {'params': self.model.MLPE.parameters()},
                {'params': self.model.MLPFIN.parameters()},
            ], lr =self.lr, # weight_decay=1e-3
        )

        self._clip_gradients(value=1) # 1 for fetch, 0.1 for hand
        
        self.jitter_gen = torch.Generator(device='cuda')
        self.jitter_gen.manual_seed(42)
        self.jitter_count = 1
                
        self.grad_accum = 0
        
        self.s = 0
        self.prev_scale = 1
        self.layer_std = dict()
        
    def _load_encoder(self, encoder: str):
        base_model = MODELS[encoder]()
        encoder, embedding_size = base_model._load_encoder()
        preprocess = base_model.preprocess
        return encoder, embedding_size, preprocess
    
    def eval_encoder(self, image: torch.Tensor) -> torch.Tensor:
        # image = self.preprocess(image)
        with torch.no_grad():
            embeddings = self.encoder(image)
        return embeddings

    def _toggle_train(self, mode):
        if mode == 'train':
            self.model.MLPE.train()
            self.model.MLPPOS.train()
            self.model.MLPFIN.train()
        elif mode == 'eval':
            self.model.MLPE.eval()
            self.model.MLPPOS.eval()
            self.model.MLPFIN.eval()

    def _kill_grad(self, module: nn.Module, name: str) -> None:
        counter = 0
        for p in module.parameters():
            if p.requires_grad:
                counter += 1
                p.requires_grad = False
        print(f'Killed {counter} parameter grads in {name}')
        
    def _load_pretrained(self, module: nn.Module, path: str, name: str) -> None:
        print(f'Loading {path} to {name}')
        module.load_state_dict(torch.load(path, map_location=self.device))

    def _clip_gradients(self, value: int = 0.1) -> None:
        if value == 0:
            print('Warning, no gradients clipped')
            return 
        if value < 0:
            print('Clipping value must be > 0')
            value = -value
            
        for p in self.model.parameters():
            if p.requires_grad:
                p.register_hook(lambda grad : torch.clamp(grad, -value, value))
    
    def _initialize_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            # torch.nn.init.orthogonal_(m.bias)
            # torch.nn.init.orthogonal_(m.weight, a=-0.25, b=0.25)
            # torch.nn.init.orthogonal_(m.bias, a=-0.25, b=0.25)
            # print('skip init weights')
            # pass  
            
    def get_heads(self) -> nn.ModuleDict:
        '''
        Returns a dict of the named MLP heads in self.model
        '''
        return nn.ModuleDict({'MLPI': self.model.MLPI,
                'MLPG': self.model.MLPG,
                'MLPFIN' : self.model.MLPFIN,
                'MLPPOS': self.model.MLPPOS
        })
    
        
    def load_lora(self) -> None:
        raise NotImplementedError

    def save_lora(self) -> None:
        raise NotImplementedError
    