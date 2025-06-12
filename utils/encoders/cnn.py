import torch
import torch.nn as nn
from utils.policy_net import PolicyNetwork

from utils.encoders.vqvae.auto_encoder import VQ_CVAE

class CNNPolicy(PolicyNetwork):
    def __init__(self, device: str = 'cuda', state_size: int = 0, action_size: int = 0, layers: list = [256, 128, 64], pos_contrib: int = 16):
        super().__init__(device, state_size, action_size, layers, pos_contrib)
    
    def _load_encoder(self):
        model = VQ_CVAE(d=256)
        model.encoder.load_state_dict(torch.load('./experiments/utils/cnn_extractor/cnn_encoder.pth', map_location=self.device))
        model.encoder.eval()
        return model.encoder, 9216
    
    def train(self, init_images_embs_mb: torch.Tensor, goal_images_embs_mb: torch.Tensor, actions_mb: torch.Tensor,
              init_pos_mb: torch.Tensor, goal_pos_mb: torch.Tensor
              ) -> torch.Tensor:
        self._toggle_train('train')
        init_images_embs_mb = init_images_embs_mb.permute((0,3,1,2))
        goal_images_embs_mb = goal_images_embs_mb.permute((0,3,1,2))
        
        init_images_embs_mb = self.model.ENCODE(init_images_embs_mb)
        goal_images_embs_mb = self.model.ENCODE(goal_images_embs_mb)
        
        features1_mb = torch.flatten(init_images_embs_mb, start_dim=1)
        features2_mb = torch.flatten(goal_images_embs_mb, start_dim=1)
        
        features1_mb, features2_mb = nn.Dropout(p=0.3)(features1_mb), nn.Dropout(p=0.3)(features2_mb)

        image_features_mb = torch.concat((self.model.MLPI(features1_mb), self.model.MLPG(features2_mb)), dim = -1)
        image_features_mb = nn.SELU()(image_features_mb)

        state_features_mb = torch.concat((init_pos_mb, goal_pos_mb), dim = -1)
        # state_features_mb = init_pos_mb
        pos_proj_mb = self.model.MLPPOS(state_features_mb)

        pred_acts_mb = self.model.MLPFIN(torch.concat((image_features_mb, pos_proj_mb), dim = -1))

        loss = self.criterion(pred_acts_mb, actions_mb)
    
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
    
    def test(self, init_images_embs_mb: torch.Tensor, goal_images_embs_mb: torch.Tensor, actions_mb: torch.Tensor,
             init_pos_mb: torch.Tensor, goal_pos_mb: torch.Tensor
             ) -> torch.Tensor:
        self._toggle_train('eval')
        with torch.no_grad():
            init_images_embs_mb = init_images_embs_mb.permute((0,3,1,2))
            goal_images_embs_mb = goal_images_embs_mb.permute((0,3,1,2))
        
            init_images_embs_mb = self.model.ENCODE(init_images_embs_mb)
            goal_images_embs_mb = self.model.ENCODE(goal_images_embs_mb)
            
            features1_mb = torch.flatten(init_images_embs_mb, start_dim=1)
            features2_mb = torch.flatten(goal_images_embs_mb, start_dim=1)

            image_features_mb = torch.concat((self.model.MLPI(features1_mb), self.model.MLPG(features2_mb)), dim = -1)
            image_features_mb = nn.SELU()(image_features_mb)

            state_features_mb = torch.concat((init_pos_mb, goal_pos_mb), dim = -1)
            # state_features_mb = init_pos_mb

            pos_proj_mb = self.model.MLPPOS(state_features_mb)
            pred_acts_mb = self.model.MLPFIN(torch.concat((image_features_mb, pos_proj_mb), dim = -1))

            loss = self.criterion(pred_acts_mb, actions_mb)
        return loss
    
    def act(self, image, goal, init_pos, goal_pos):
        self.model.eval()
        img, gl, initpos, goalpos = image.copy(), goal.copy(), init_pos.copy(), goal_pos.copy()
        image_torch = torch.from_numpy(img).to(device=self.device, dtype=torch.float32).unsqueeze(0)
        goal_torch = torch.from_numpy(gl).to(device=self.device, dtype=torch.float32).unsqueeze(0)
        initpos_torch = torch.from_numpy(initpos).to(device=self.device, dtype=torch.float32).unsqueeze(0)
        goalpos_torch = torch.from_numpy(goalpos).to(device=self.device, dtype=torch.float32).unsqueeze(0)
                
        with torch.no_grad():            
            features1_mb = self.model.ENCODE(image_torch) 
            features2_mb = self.model.ENCODE(goal_torch) 

            features1_mb = torch.flatten(features1_mb, start_dim=1)
            features2_mb = torch.flatten(features2_mb, start_dim=1)

            image_features_mb = torch.concat((self.model.MLPI(features1_mb), self.model.MLPG(features2_mb)), dim = -1)
            image_features_mb = nn.SELU()(image_features_mb)
            
            state_features_mb = torch.concat((initpos_torch, goalpos_torch), dim = -1)
            # state_features_mb = initpos_torch
  
            pos_proj = self.model.MLPPOS(state_features_mb)

            pred_acts_mb = self.model.MLPFIN(torch.concat((image_features_mb, pos_proj), dim = -1))

        return pred_acts_mb.detach().cpu().numpy().squeeze()