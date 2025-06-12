from typing import Any
import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch
import torch.nn as nn
import time
import numpy as np

import matplotlib.pyplot as plt
from utils.classifier import ClassifierNetwork

class LightningClassifierMode(L.LightningModule):
    def __init__(self, classifier: ClassifierNetwork, topk: int, jitter: float, image_path: str) -> None:
        super().__init__()
        self.policy = classifier
        self.final_loss = 'NaN'
        self.preprocess = self.policy.preprocess
        self.jitter = jitter
        self.jitter_gen = torch.Generator(device='cpu')
        self.jitter_gen.manual_seed(42)
        self.topk = topk
        self.metrics = {'train_loss': [], 'eval_loss': [], f'train_top_{self.topk}_acc': [], f'eval_top_{self.topk}_acc': []}
        self.epoch = []
        self.train_loss = []
        self.eval_loss = []
        self.eval_acc = []
        self.plot_paths = image_path
        
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        image, label = batch 
        
        # if self.jitter and 0 in self.layer_std:# and self.s > 0.04:
        #     with torch.no_grad():
        #         if not torch.any(self.layer_std[0] == 0):
        #             self.model.MLPFIN.layers[0].weight.add_(torch.randn(self.model.MLPFIN.layers[0].weight.size(), device=self.device, generator=self.jitter_gen) / (self.jitter * self.layer_std[0].unsqueeze(1)))
        #         if not torch.any(self.layer_std[1] == 0):
        #             self.model.MLPFIN.layers[1].weight.add_(torch.randn(self.model.MLPFIN.layers[1].weight.size(), device=self.device, generator=self.jitter_gen) / (self.jitter *self.layer_std[1].unsqueeze(1)))
        
        features = self.policy.eval_encoder(image)
        features = torch.flatten(features, start_dim=1)
        
        # if self.jitter:
        #     sd1 = torch.std(features, dim = 1).to('cuda').unsqueeze(1)
        #     unif = torch.rand((features.shape[0], features.shape[1]), generator = self.jitter_gen).to('cuda')
        #     jitter_amp = sd1 * (2 * unif - 1) / self.jitter #(torch.sqrt(torch.Tensor([features1_mb.shape[1]]).to('cuda')))
        #     features += jitter_amp
            
        image_embeddings = self.policy.model.MLPE(features)
        predictions = self.policy.model.MLPFIN(image_embeddings)
        
        loss = nn.functional.cross_entropy(predictions, label)
        
        batch_topk = torch.argsort(predictions, dim=1, descending=True)[:, 0:self.topk]
        acc = np.mean([truth in topk_preds for topk_preds, truth in zip(batch_topk, label)])
        
        self.metrics[f'train_top_{self.topk}_acc'].append(acc)
        self.metrics['train_loss'].append(loss)
        return loss 
    
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        
        image, label = batch 
        
        features = self.policy.eval_encoder(image)
        features = torch.flatten(features, start_dim=1)
        image_embeddings = self.policy.model.MLPE(features)
        predictions = self.policy.model.MLPFIN(image_embeddings)
        
        loss = nn.functional.cross_entropy(predictions, label)
        
        batch_topk = torch.argsort(predictions, dim=1, descending=True)[:, 0:self.topk]
        acc = np.mean([truth in topk_preds for topk_preds, truth in zip(batch_topk, label)])
        
        self.metrics[f'eval_top_{self.topk}_acc'].append(acc)
        self.metrics['eval_loss'].append(loss)
        return loss 
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(self.parameters(), lr = 0.0003)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.125)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
    def on_train_epoch_start(self) -> None:
        self.metrics['train_loss']= []
        self.start = time.time()
    
    # def on_train_epoch_end(self) -> None:
    #     train_loss = torch.mean(torch.Tensor(self.metrics['train_loss']))
    #     train_acc = torch.mean(torch.Tensor(self.metrics[f'train_top_{self.topk}_acc']))
    #     print('Epoch:', self.current_epoch, '| Loss:', train_loss.item(), ' | Acc:', train_acc.item(), '| Took', (self.end - self.start)/60, 'minutes' )
    #     self.metrics['train_loss']= []
    #     self.metrics[f'train_top_{self.topk}_acc']=[]
    #     self.final_loss = train_loss.item()
    
    def on_validation_epoch_end(self):
        eval_loss = torch.mean(torch.Tensor(self.metrics['eval_loss']))
        eval_acc = torch.mean(torch.Tensor(self.metrics[f'eval_top_{self.topk}_acc']))
        train_loss = torch.mean(torch.Tensor(self.metrics['train_loss']))
        train_acc = torch.mean(torch.Tensor(self.metrics[f'train_top_{self.topk}_acc']))
        self.end = time.time()
        print('-' * 70)
        print('Epoch:', self.current_epoch, '\t\t', 'Train', '\t\t\t', 'Eval')
        print('Loss:\t\t\t', round(train_loss.item(), 5), '\t\t', round(eval_loss.item(), 5))
        print(f'Acc (Top {self.topk}):\t\t', round(train_acc.item(), 5), '\t\t', round(eval_acc.item(), 5))
        print('Epoch time:',  (self.end - self.start)/60, 'minutes')
        print('-' * 70)
        self.metrics['train_loss']= []
        self.metrics[f'train_top_{self.topk}_acc']=[]
        self.metrics['eval_loss']=[]
        self.metrics[f'eval_top_{self.topk}_acc']=[]
        self.epoch.append(self.current_epoch)
        self.train_loss.append(train_loss.item())
        self.eval_loss.append(eval_loss.item())
        self.eval_acc.append(eval_acc.item())
        self.final_loss = eval_loss.item()
        
    def plot(self):
        plt.plot(self.epoch, self.eval_loss, label = 'eval_loss')
        plt.plot(self.epoch, self.train_loss, label = 'train_loss')
        plt.title(self.policy.encoder_name)
        plt.legend()
        plt.savefig(f'{self.plot_paths}/{self.policy.encoder_name}.png')
        plt.cla()
        plt.clf()
        plt.plot(self.epoch, self.eval_acc, label = 'eval_acc')
        plt.title(self.policy.encoder_name)
        plt.legend()
        plt.savefig(f'{self.plot_paths}/{self.policy.encoder_name}_acc.png')
        