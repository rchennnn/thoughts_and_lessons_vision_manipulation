from typing import List
import torch.nn as nn
import torch

class CustomMLP(nn.Module):
    def __init__(self, layers: List[int], response_size, dropout=0.2, active: bool = True, layer_norm: bool = False, seed: int =None) -> None:
        super().__init__()
        if seed:
            torch.manual_seed(seed=seed) 
            
        module_list = []
        for k, layer_size in enumerate(layers):
            if k < len(layers) - 1:
                l = nn.Linear(in_features=layer_size, out_features=layers[k + 1])
                module_list.append(l)
            else:
                l = nn.Linear(in_features=layer_size, out_features=response_size)
                module_list.append(l)
        self.layers = nn.ModuleList(module_list)
        self.active = active
        self.lrelu = nn.LeakyReLU()
        self.selu = nn.SELU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout)
        self.use_ln = layer_norm
        self.bn = nn.BatchNorm1d(num_features=layers[0], track_running_stats=False)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(normalized_shape=k, device='cuda', elementwise_affine=False) for k in layers[1:]])

    def forward(self, x):
        total_layers=len(self.layers)
        for k, fc_layer in enumerate(self.layers):
            if k < total_layers - 1:
                x = fc_layer(x)
                if self.active:
                    x = self.selu(x)
                if self.use_ln:
                    x = self.layer_norms[k](x)
            else:
                out = fc_layer(x)
        return out

class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=9, stride=2, padding=1),
            # nn.BatchNorm2d(32, track_running_stats=False),
            nn.SELU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=9, stride=2, padding=1),
            # nn.BatchNorm2d(64, track_running_stats=False),
            nn.SELU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=9, stride=2, padding=1),
            # nn.BatchNorm2d(128, track_running_stats=False),
            nn.SELU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=9, stride=2, padding=1),
            # nn.BatchNorm2d(256, track_running_stats=False),
            nn.SELU())
        # self.layer5 = nn.Sequential(
        #     nn.Conv2d(512, 1024, kernel_size=5, stride=1, padding=2),
        #     nn.BatchNorm2d(1024, track_running_stats=False),
        #     nn.ReLU())
        # self.layer6 = nn.Sequential(
        #     nn.Conv2d(1024, 2048, kernel_size=5, stride=1, padding=2),
        #     nn.BatchNorm2d(2048, track_running_stats=False),
        #     nn.ReLU())
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x