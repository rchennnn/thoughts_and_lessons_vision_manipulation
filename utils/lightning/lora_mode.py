from typing import Any
import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler

import torch
import torch.nn as nn
import time

from utils.policy_net import PolicyNetwork

class LightningLoRATraining(L.LightningModule):
    def __init__(self, policy: PolicyNetwork, save_lora_path: str = None) -> None:
        super().__init__()
        self.policy = policy
        
        self.policy.load_lora()
        self.policy.model.ENCODE.print_trainable_parameters()
        
        self.final_loss = 'NaN'
        self.preprocess = self.policy.preprocess
        self.metrics = {'train_loss': [], 'eval_loss': []}
        
        self.save_lora_path = save_lora_path
    
    '''
    Training step
    '''
    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:     
        init_image_or_embedding, goal_image_or_embedding, action, init_state_space, goal_state_space = batch 

        if len(init_image_or_embedding.shape) > 2 and init_image_or_embedding.shape[1] == 3:
            init_image, goal_image = self.preprocess(init_image_or_embedding), self.preprocess(goal_image_or_embedding)
        
            init_image_or_embedding = self.policy.model.ENCODE(init_image)
            goal_image_or_embedding = self.policy.model.ENCODE(goal_image)
        
        features1_mb = torch.flatten(init_image_or_embedding, start_dim=1)
        features2_mb = torch.flatten(goal_image_or_embedding, start_dim=1)
        
        p = 0.3
        features1_mb, features2_mb = nn.Dropout(p=p)(features1_mb), nn.Dropout(p=p)(features2_mb)
        image_features_mb = torch.concat((self.policy.model.MLPI(features1_mb), self.policy.model.MLPG(features2_mb)), dim = -1)
        image_features_mb = nn.SELU()(image_features_mb)
        
        state_features_mb = torch.concat((init_state_space, goal_state_space), dim = -1)
        pos_proj_mb = self.policy.model.MLPPOS(state_features_mb)
        pos_proj_mb = nn.SELU()(pos_proj_mb)
        
        # make predictions
        pred_acts_mb = self.policy.model.MLPFIN(torch.concat((image_features_mb, pos_proj_mb), dim = -1))

        # get losses
        loss = nn.functional.mse_loss(pred_acts_mb, action)
        
        self.log('train_loss', loss)
        self.metrics['train_loss'].append(loss)
        return loss 
    
    
        
    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.NAdam(self.parameters(), lr = 0.0008)
        return optimizer
    
    def on_train_epoch_start(self) -> None:
        self.metrics['train_loss']= []
        self.start = time.time()
    
    def on_train_epoch_end(self) -> None:
        train_loss = torch.mean(torch.Tensor(self.metrics['train_loss']))
        self.end = time.time()
        print('Epoch:', self.current_epoch, 'Loss :', train_loss.item(), 'Took', (self.end - self.start)/60, 'minutes' )
        self.metrics['train_loss'] = []
        self.final_loss = train_loss.item()
    
    '''
    Validation step
    '''
    
    def validation_step(self, batch, batch_idx):
        init_image_or_embedding, goal_image_or_embedding, action, init_state_space, goal_state_space = batch 
        if len(init_image_or_embedding.shape) > 2 and init_image_or_embedding.shape[1] == 3:
            init_image, goal_image = self.preprocess(init_image_or_embedding), self.preprocess(goal_image_or_embedding)
                
            init_image_or_embedding = self.policy.model.ENCODE(init_image)
            goal_image_or_embedding = self.policy.model.ENCODE(goal_image)
        
        features1_mb = torch.flatten(init_image_or_embedding, start_dim=1)
        features2_mb = torch.flatten(goal_image_or_embedding, start_dim=1)
                
        image_features_mb = torch.concat((self.policy.model.MLPI(features1_mb), self.policy.model.MLPG(features2_mb)), dim = -1)
        image_features_mb = nn.SELU()(image_features_mb)
                
        state_features_mb = torch.concat((init_state_space, goal_state_space), dim = -1)
        pos_proj_mb = self.policy.model.MLPPOS(state_features_mb)
        pos_proj_mb = nn.SELU()(pos_proj_mb)
                
        # make predictions
        pred_acts_mb = self.policy.model.MLPFIN(torch.concat((image_features_mb, pos_proj_mb), dim = -1))
        # get losses
        loss = nn.functional.mse_loss(pred_acts_mb, action)
                
        self.log('eval_loss', loss)
        self.metrics['eval_loss'].append(loss)
        return loss 
    
    def on_validation_epoch_end(self):
        self.metrics['eval_loss']= []
    
    def on_validation_epoch_end(self):
        eval_loss = torch.mean(torch.Tensor(self.metrics['eval_loss']))
        print('\t\tEpoch:', self.current_epoch, 'Eval Loss :', eval_loss.item())
        self.metrics['eval_loss']=[]
        self.final_loss = eval_loss.item()
    
    '''
    What to do when training ends
    '''
    
    def on_train_end(self) -> None:
        if self.save_lora_path:
            self.policy.save_lora(self.save_lora_path)
        return super().on_train_end()
    