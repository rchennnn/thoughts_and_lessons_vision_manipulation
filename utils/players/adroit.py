from typing import List
import torch.nn as nn
import torch 

class CustomMLP(nn.Module):
    def __init__(self, layers: List[int], response_size: int, active: bool) -> None:
        super().__init__()
        self.layers = []
        n = len(layers)
        for k, _ in enumerate(layers):
            if k < len(layers) - 1:
                self.layers.append(nn.Linear(in_features=layers[k], out_features=layers[k+1]))
            else:
                self.layers.append(nn.Linear(in_features=layers[n-1], out_features=response_size))
        
        self.activ = nn.SELU()
        self.active = active

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        n = len(self.layers)
        for k, layer in enumerate(self.layers):
            if k < n - 1:
                x = layer(x)
                if self.active:
                    x = self.activ(x)

        out = self.layers[-1](x)
        return out

class AdroitImitationPolicy:
    def __init__(self, init_layers_features: List[int], init_feature_extractor_out: int,
                #  goal_layers_features: List[int], goal_feature_extractor_out: int, 
                 layers_cloner: List[int], cloner_out: int, device: torch.device) -> None:
        
        self.device = device
                        
        self.model = nn.ModuleDict(
            {
                'MLPO': CustomMLP(layers=init_layers_features, response_size=init_feature_extractor_out, active=True),
                'MLPF': CustomMLP(layers=layers_cloner, response_size=cloner_out, active=True),
            }
        ).to(device)

        self.optimizer = torch.optim.NAdam(params=self.model.parameters(), lr=0.0001)
        self.criterion = nn.MSELoss()
        
        for p in self.model.parameters():
            if p.requires_grad:
                p.register_hook(lambda grad: torch.clamp(grad, -0.1, 0.1))

    def train(self, init_obs, goal_obs= None, actions = None):
        init_features = self.model.MLPO(init_obs)
        pred_action = self.model.MLPF(init_features)

        loss = self.criterion(pred_action, actions)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def test(self, init_obs, goal_obs = None, actions = None):
        with torch.no_grad():
            init_features = self.model.MLPO(init_obs)
            pred_action = self.model.MLPF(init_features)

            loss = self.criterion(pred_action, actions)
        
        return loss.item()
    
    def act(self, init_ob):
        init_ob = torch.from_numpy(init_ob).to(dtype=torch.float32, device=self.device)
        with torch.no_grad():
            init_features = self.model.MLPO(init_ob)
            pred_action = self.model.MLPF(init_features)
        return pred_action.detach().cpu().numpy()
    
    def save_policy(self, path: str = '/projects/p31777/ryan/.minari/cloning/policy/', name: str = 'relocate_cloned_agent.pth'):
        torch.save(self.model.state_dict(), f'{path}/{name}')
        
    def load_policy(self, path: str, device= str):
        self.model.load_state_dict(torch.load(path, map_location=device))
        
        
class FetchImitationPolicy:
    def __init__(self, observation_size:int, action_size: int, device: torch.device) -> None:
        
        self.device = device
                        
        self.model = nn.Sequential(
            nn.Linear(observation_size, 128),
            nn.SELU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 256),
            nn.SELU(),
            nn.Linear(256, 256),
            nn.SELU(),
            nn.Linear(256, 256),
            nn.SELU(),
            nn.Linear(256, 256),
            nn.SELU(),
            nn.Linear(256, 128),
            nn.SELU(),
            nn.Linear(128, action_size)            
        )

        self.optimizer = torch.optim.NAdam(params=self.model.parameters(), lr=0.0003)
        self.criterion = nn.MSELoss()
        
        for p in self.model.parameters():
            if p.requires_grad:
                p.register_hook(lambda grad: torch.clamp(grad, -0.1, 0.1))

    def train(self, obs, dgs, acts):
        self.model.train()
        pred = self.model(torch.concat((obs, dgs), dim=-1))
        loss = self.criterion(pred, acts)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def test(self, obs, dgs, acts):
        self.model.eval()
        with torch.no_grad():
            pred = self.model(torch.concat((obs, dgs), dim=-1))
            loss = self.criterion(pred, acts)
        return loss.item()
    
    def act(self, obs, dgs):
        self.model.eval()
        obs = torch.from_numpy(obs).to(torch.float32)
        dgs = torch.from_numpy(dgs).to(torch.float32)
        with torch.no_grad():
            return self.model(torch.concat((obs, dgs), dim = -1)).detach().cpu().numpy()
    
    def save_policy(self, path: str = '/projects/p31777/ryan/fetch_data/fetch_data/experts/', name: str = 'expert-fetchplace-distract.pth'):
        torch.save(self.model.state_dict(), f'{path}/{name}')
        
    def load_policy(self, path: str, device= str):
        self.model.load_state_dict(torch.load(path, map_location=device), strict=False)