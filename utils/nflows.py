import torch
import torch.optim as optim
import normflows as nf
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR


class NormFlowEstimator():
    def __init__(self, sample_data):
        super(NormFlowEstimator, self).__init__()
        self.X = sample_data.to('cuda')
        mean_per_dim = torch.mean(self.X, dim=0)
        dim = self.X.shape[1]
        self.base_distribution = nf.distributions.base.DiagGaussian(self.X.shape[1])
        self.flows = []
        self.model = None
        self.loss_hist = np.array([])
        self.optimizer = None
        self.scheduler = None
        num_layers = 32
        flows = []
        for i in range(num_layers):
            # Neural network with two hidden layers having 64 units each
            # Last layer is initialized by zeros making training more stable
            param_map = nf.nets.MLP([dim, 64, 64, 2], init_zeros=True)
            # Add flow layer
            # flows.append(nf.flows.AffineCouplingBlock(param_map))
            # Swap dimensions
            flows.append(nf.flows.Permute(2, mode='swap'))

          # Construct flow model
        self.model = nf.NormalizingFlow(self.base_distribution, flows)
        self.model.to('cuda')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4,)# weight_decay=1e-5)
          
    def train(self, iterations, batch_size):
      
        # Train model
        max_iter = iterations
        num_samples = batch_size
        model_prev = self.model
        loss_hist = []
        for it in range(max_iter):
            self.optimizer.zero_grad()

            # Get training samples
            idx = np.random.randint(0, 50, num_samples)
            x = self.X[idx]

            # Compute loss
            loss = self.model.forward_kld(x)
            if loss.item() < 0:
                print('Negative loss')
                self.model = model_prev
                break
            # Do backprop and optimizer step
            if ~(torch.isnan(loss) | torch.isinf(loss)):
                loss.backward()
                self.optimizer.step()
                loss_hist.append(loss.item())
                # print(loss.item())
                # scheduler.step()
            # Log loss
            if it % 100 == 0:
                print('Iteration: {}, Loss: {:.3f}'.format(it, loss.item()))
            model_prev = self.model
        # print(self.model.log_prob(torch.ones((10, 512)).to('cuda')))
        # -726.4965
        return self.model
    
    def __call__(self, x):
        return self.model.log_prob(x)

if __name__ == '__main__':
    sample_data = torch.rand((50, 512))

    nfe = NormFlowEstimator(sample_data)
    nfe.train(8000, 32)
