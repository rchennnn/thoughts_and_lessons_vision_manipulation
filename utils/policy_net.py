from typing import Dict, Optional
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.cuda.amp import autocast, grad_scaler

import mvp
from segment_anything import sam_model_registry
from segment_anything.modeling import ImageEncoderViT

from utils.net_utils import CustomMLP

class PolicyNetwork(nn.Module):
    def __init__(self, device : str ='cuda', state_size: int = 0, action_size: int = 0, layers: list = [256,128,64], pos_contrib:int = 16, seed: int= 0x0badbeef):
        super().__init__()
        
        self.device = device
        self.preprocess = None
        self.seed = seed

        self.encoder, self.embed_size = self._load_encoder()

        for p in self.encoder.parameters():
            if p.requires_grad:
                p.requires_grad = False

        torch.manual_seed(self.seed)
        
        head_layers = layers
        layers[0] = layers[0] + pos_contrib

        self.model = nn.ModuleDict({
            'ENCODE': self.encoder, # 50176
            'MLPI': CustomMLP(layers=[self.embed_size, 256, 128], response_size=64, seed=self.seed,), # [embed, 256, 128] and 64
            'MLPG': CustomMLP(layers=[self.embed_size, 256, 128], response_size=64, seed=self.seed,),
            'MLPPOS': nn.Linear(in_features=state_size * 2, out_features=pos_contrib),
            'MLPFIN': CustomMLP(layers=head_layers, response_size=action_size, seed=self.seed, layer_norm=True)
        }).to(device)
        
        print(f'Head layers: {head_layers}')
        
        # self.model.MLPI.layers[0].weight.requires_grad = False
        # self.model.MLPI.layers[0].bias.requires_grad = False
        # self.model.MLPG.layers[0].weight.requires_grad = False
        # self.model.MLPG.layers[0].bias.requires_grad = False
        
        # orthogonal init for hand envs. use default for fetch envs
        self.model.MLPI.apply(self._initialize_weights)
        self.model.MLPG.apply(self._initialize_weights)
        self.model.MLPPOS.apply(self._initialize_weights)
        self.model.MLPFIN.apply(self._initialize_weights)
        
        # self._kill_grad(self.model.MLPI, 'MLPI')
        # self._kill_grad(self.model.MLPG, 'MLPG')

        self.criterion = nn.MSELoss()
        self.lr = 0.0008 # 0.0008, 0.0003
        self.optimizer = torch.optim.NAdam(
            [   
                # {'params': self.model.ENCODE.parameters()},
                {'params': self.model.MLPI.parameters()},
                {'params': self.model.MLPG.parameters()},
                {'params': self.model.MLPPOS.parameters()},
                {'params': self.model.MLPFIN.parameters()},
            ], lr =self.lr, # weight_decay=1e-3
        )

        self._clip_gradients(value=0.1) # 1 for fetch, 0.1 for hand
        
        self.jitter_gen = torch.Generator()
        self.jitter_gen.manual_seed(42)
        self.jitter_count = 1
        self.layer_std = dict()
        self._hook_layers()
        
        self.amp_scaler = torch.cuda.amp.GradScaler()
    
    def _hook_layers(self):
        def get_out(name):
            def hooks(module, input, output):
                self.layer_std[name] = torch.std(output.detach(), dim = 0)
                # print(module, self.layer_std[name])
            return hooks        
        self.model.MLPFIN.layers[0].register_forward_hook(get_out(0))
        self.model.MLPFIN.layers[1].register_forward_hook(get_out(1))
        
    def _load_encoder(self):
        raise NotImplemented
    
    def eval_encoder(self, image: torch.Tensor) -> torch.Tensor:
        image = self.preprocess(image)
        with torch.no_grad():
            embeddings = self.encoder(image)
        return embeddings

    def train_probe(self, init_images_embs_mb: torch.Tensor, goal_images_embs_mb: torch.Tensor, actions_mb: torch.Tensor,
              init_pos_mb: torch.Tensor, goal_pos_mb: torch.Tensor, jitter: Optional[float] = None, epoch: int = 0
              ) -> torch.Tensor:
        self._toggle_train('train')
        
        if jitter and 0 in self.layer_std:# and self.s > 0.04:
            with torch.no_grad():
                if not torch.any(self.layer_std[0] == 0):
                    self.model.MLPFIN.layers[0].weight.add_(torch.randn(self.model.MLPFIN.layers[0].weight.size(), device=self.device, generator=self.jitter_gen) / (jitter * self.layer_std[0].unsqueeze(1)))
                if not torch.any(self.layer_std[1] == 0):
                    self.model.MLPFIN.layers[1].weight.add_(torch.randn(self.model.MLPFIN.layers[1].weight.size(), device=self.device, generator=self.jitter_gen) / (jitter * self.layer_std[1].unsqueeze(1)))
        
        # flatten features and apply dropout
        
        features1_mb = torch.flatten(init_images_embs_mb, start_dim=1)
        features2_mb = torch.flatten(goal_images_embs_mb, start_dim=1)
        
        # if jitter:
        #     sd1 = torch.std(features1_mb, dim = -1).to('cuda')
        #     sd2 = torch.std(features1_mb, dim = -1).to('cuda')
        #     jitter1 = torch.rand((features1_mb.shape[1], features1_mb.shape[0]), generator = self.jitter_gen).to('cuda')
        #     jitter2 = torch.rand((features2_mb.shape[1], features2_mb.shape[0]), generator = self.jitter_gen).to('cuda')
        #     jitter1 = sd1 * (2 * jitter1 - 1) / jitter #(torch.sqrt(torch.Tensor([features1_mb.shape[1]]).to('cuda')))
        #     jitter2 = sd2 * (2 * jitter2 - 1) / jitter #(torch.sqrt(torch.Tensor([features2_mb.shape[1]]).to('cuda')))
        #     features1_mb += jitter1.permute((1,0))
        #     features2_mb += jitter2.permute((1,0))
        
        p = 0.3
        features1_mb, features2_mb = nn.Dropout(p=p)(features1_mb), nn.Dropout(p=p)(features2_mb) # 0.3

        # get image features
        image_features_mb = torch.concat((self.model.MLPI(features1_mb), self.model.MLPG(features2_mb)), dim = -1)
        image_features_mb = nn.SELU()(image_features_mb)

        # get state features
        state_features_mb = torch.concat((init_pos_mb, goal_pos_mb), dim = -1)
        pos_proj_mb = self.model.MLPPOS(state_features_mb)
        pos_proj_mb = nn.SELU()(pos_proj_mb)
        
        # make predictions
        pred_acts_mb = self.model.MLPFIN(torch.concat((image_features_mb, pos_proj_mb), dim = -1))

        # get losses
        loss = self.criterion(pred_acts_mb, actions_mb)
        
        # backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
    
    def test_probe(self, init_images_embs_mb: torch.Tensor, goal_images_embs_mb: torch.Tensor, actions_mb: torch.Tensor,
             init_pos_mb: torch.Tensor, goal_pos_mb: torch.Tensor
             ) -> torch.Tensor:
        self._toggle_train('eval')
        with torch.no_grad():
            # get image features and push forward network
            features1_mb = torch.flatten(init_images_embs_mb, start_dim=1)
            features2_mb = torch.flatten(goal_images_embs_mb, start_dim=1)
            image_features_mb = torch.concat((self.model.MLPI(features1_mb), self.model.MLPG(features2_mb)), dim = -1)
            image_features_mb = nn.SELU()(image_features_mb)

            # get state features and push forward network
            state_features_mb = torch.concat((init_pos_mb, goal_pos_mb), dim = -1)
            pos_proj_mb = self.model.MLPPOS(state_features_mb)
            pos_proj_mb = nn.SELU()(pos_proj_mb)
            
            # prediction actions and get losses
            pred_acts_mb = self.model.MLPFIN(torch.concat((image_features_mb, pos_proj_mb), dim = -1))
            loss = self.criterion(pred_acts_mb, actions_mb)
        return loss
        
    def act(self, image, goal, init_pos, goal_pos):
        torch.set_warn_always(False)    
        self.model.eval()
        img, gl, initpos, goalpos = image.copy(), goal.copy(), init_pos.copy(), goal_pos.copy()
        image_torch = torch.from_numpy(img).to(device=self.device, dtype=torch.float32).unsqueeze(0)
        goal_torch = torch.from_numpy(gl).to(device=self.device, dtype=torch.float32).unsqueeze(0)
        initpos_torch = torch.from_numpy(initpos).to(device=self.device, dtype=torch.float32).unsqueeze(0)
        goalpos_torch = torch.from_numpy(goalpos).to(device=self.device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            # preprocess
            image_torch = self.preprocess(image_torch)
            goal_torch = self.preprocess(goal_torch)
            
            # extract image features
            features1_mb = self.model.ENCODE(image_torch) 
            features2_mb = self.model.ENCODE(goal_torch) 

            features1_mb = torch.flatten(features1_mb, start_dim=1)
            features2_mb = torch.flatten(features2_mb, start_dim=1)

            image_features_mb = torch.concat((self.model.MLPI(features1_mb), self.model.MLPG(features2_mb)), dim = -1)
            image_features_mb = nn.SELU()(image_features_mb)
            
            # extract state features
            state_features_mb = torch.concat((initpos_torch, goalpos_torch), dim = -1)  
            pos_proj = self.model.MLPPOS(state_features_mb)
            pos_proj = nn.SELU()(pos_proj)

            # predict action
            pred_acts_mb = self.model.MLPFIN(torch.concat((image_features_mb, pos_proj), dim = -1))

        return pred_acts_mb.detach().cpu().numpy().squeeze()

    def _toggle_train(self, mode):
        if mode == 'train':
            self.model.MLPI.train()
            self.model.MLPG.train()
            self.model.MLPPOS.train()
            self.model.MLPFIN.train()
        elif mode == 'eval':
            self.model.MLPI.eval()
            self.model.MLPG.eval()
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
    
        
    def run_forward(self, image, goal, init_pos, goal_pos):
        image_torch = self.preprocess(image)
        goal_torch = self.preprocess(goal)
        
        # extract image features
        features1_mb = self.model.ENCODE(image_torch) 
        features2_mb = self.model.ENCODE(goal_torch) 

        features1_mb = torch.flatten(features1_mb, start_dim=1)
        features2_mb = torch.flatten(features2_mb, start_dim=1)

        image_features_mb = torch.concat((self.model.MLPI(features1_mb), self.model.MLPG(features2_mb)), dim = -1)
        image_features_mb = nn.SELU()(image_features_mb)
        
        # extract state features
        state_features_mb = torch.concat((init_pos, goal_pos), dim = -1)  
        pos_proj = self.model.MLPPOS(state_features_mb)
        pos_proj = nn.SELU()(pos_proj)

        # predict action
        pred_acts_mb = self.model.MLPFIN(torch.concat((image_features_mb, pos_proj), dim = -1))

        return pred_acts_mb
        
    def load_lora(self) -> None:
        raise NotImplementedError

    def save_lora(self) -> None:
        raise NotImplementedError
    
    def train_images(self, image_init_mb: torch.Tensor, image_goal_mb: torch.Tensor, init_pos_mb: torch.Tensor, goal_pos_mb: torch.Tensor, actions_mb: torch.Tensor):
        self._toggle_train('train')
        # with torch.cuda.amp.autocast():
        image_init_mb = self.preprocess(image_init_mb)
        image_goal_mb = self.preprocess(image_goal_mb)
        # get image features
        init_images_embs_mb = self.model.ENCODE(image_init_mb)
        goal_images_embs_mb = self.model.ENCODE(image_goal_mb)
        
        features1_mb = torch.flatten(init_images_embs_mb, start_dim=1)
        features2_mb = torch.flatten(goal_images_embs_mb, start_dim=1)
        p = 0.3
        features1_mb, features2_mb = nn.Dropout(p=p)(features1_mb), nn.Dropout(p=p)(features2_mb) # 0.3

        image_features_mb = torch.concat((self.model.MLPI(features1_mb), self.model.MLPG(features2_mb)), dim = -1)
        image_features_mb = nn.SELU()(image_features_mb)

        # get state features
        state_features_mb = torch.concat((init_pos_mb, goal_pos_mb), dim = -1)
        pos_proj_mb = self.model.MLPPOS(state_features_mb)
        pos_proj_mb = nn.SELU()(pos_proj_mb)
        
        # make predictions
        pred_acts_mb = self.model.MLPFIN(torch.concat((image_features_mb, pos_proj_mb), dim = -1))

        # get losses
        loss = self.criterion(pred_acts_mb, actions_mb)
        
        # backprop
        # self.optimizer.zero_grad()
        # self.amp_scaler.scale(loss).backward()
        # self.amp_scaler.step(self.optimizer)
        # self.amp_scaler.update()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss
    
    def test_images(self, image_init_mb: torch.Tensor, image_goal_mb: torch.Tensor, init_pos_mb: torch.Tensor, goal_pos_mb: torch.Tensor, actions_mb: torch.Tensor):
        self._toggle_train('eval')
        
        with torch.no_grad():
            # get image features
            with torch.cuda.amp.autocast():
                image_init_mb = self.preprocess(image_init_mb)
                image_goal_mb = self.preprocess(image_goal_mb)
                
                init_images_embs_mb = self.model.ENCODE(image_init_mb)
                goal_images_embs_mb = self.model.ENCODE(image_goal_mb)
                
                features1_mb = torch.flatten(init_images_embs_mb, start_dim=1)
                features2_mb = torch.flatten(goal_images_embs_mb, start_dim=1)

                image_features_mb = torch.concat((self.model.MLPI(features1_mb), self.model.MLPG(features2_mb)), dim = -1)
                image_features_mb = nn.SELU()(image_features_mb)

                # get state features
                state_features_mb = torch.concat((init_pos_mb, goal_pos_mb), dim = -1)
                pos_proj_mb = self.model.MLPPOS(state_features_mb)
                pos_proj_mb = nn.SELU()(pos_proj_mb)
                
                # make predictions
                pred_acts_mb = self.model.MLPFIN(torch.concat((image_features_mb, pos_proj_mb), dim = -1))

                # get losses
                loss = self.criterion(pred_acts_mb, actions_mb)
            
        return loss