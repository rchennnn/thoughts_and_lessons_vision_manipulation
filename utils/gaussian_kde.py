# https://discuss.pytorch.org/t/kernel-density-estimation-as-loss-function/62261/8

import torch
from torch.distributions import MultivariateNormal, Normal
from torch.distributions.distribution import Distribution
import numpy as np
import torch.nn as nn

class GaussianKDE(nn.Module):
    def __init__(self, X, bw = 0.01):
        """
        X : tensor (n, d)
          `n` points with `d` dimensions to which KDE will be fit
        bw : numeric
          bandwidth for Gaussian kernel
        """
        super(GaussianKDE, self).__init__()
        self.X = X
        self.bw = bw
        self.dims = X.shape[-1]
        self.n = X.shape[0]
        self.mvn = MultivariateNormal(loc=torch.zeros(self.dims),
                                      covariance_matrix=torch.eye(self.dims))

    def forward(self, samples):
        return self.score_samples(samples)

    def score_samples(self, Y, X=None, logged=True):
        """Returns the kernel density estimates of each point in `Y`.

        Parameters
        ----------
        Y : tensor (m, d)
          `m` points with `d` dimensions for which the probability density will
          be calculated
        X : tensor (n, d), optional
          `n` points with `d` dimensions to which KDE will be fit. Provided to
          allow batch calculations in `log_prob`. By default, `X` is None and
          all points used to initialize KernelDensityEstimator are included.


        Returns
        -------
        log_probs : tensor (m)
          log probability densities for each of the queried points in `Y`
        """
        if X == None:
            X = self.X

        density = torch.exp(self.mvn.log_prob(X.unsqueeze(1)-Y)).sum(dim=1) / self.n
        return density