#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 13:25:37 2024

@author: tang.1856
"""

from botorch.acquisition import AcquisitionFunction, AnalyticAcquisitionFunction
import torch
from botorch.models.model import Model
from typing import Dict, Optional, Tuple, Union
from torch import Tensor
from botorch.acquisition.objective import PosteriorTransform
from botorch.utils.transforms import convert_to_target_pre_hook, t_batch_mode_transform
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
import numpy as np
from scipy.spatial.distance import cdist, jensenshannon
from botorch.models.transforms.outcome import Standardize
from botorch.test_functions import Rosenbrock, Ackley, Hartmann, StyblinskiTang
from torch.quasirandom import SobolEngine
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import Interval
from gpytorch.kernels import RBFKernel, ScaleKernel
 
def reachability_uniformity(behavior, n_bins = 25, obj_lb = -5, obj_ub = 5):
    behavior = behavior.squeeze(1).numpy()
    num = len(behavior)
    cum_hist, _ = np.histogram(behavior, np.linspace(obj_lb, obj_ub, n_bins + 1))
    cum_hist = cum_hist[np.nonzero(cum_hist)] / (num) # discrete distribution
    cum_coverage = len(cum_hist) / n_bins    
    return cum_coverage

class MaxVariance(AnalyticAcquisitionFunction):
    r"""Single-outcome Max Variance Acq.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> MaxVar = MaxVariance(model)
        >>> maxvar = MaxVar(test_X)
    """

    def __init__(
        self,
        model: Model,
        posterior_transform: Optional[PosteriorTransform] = None,
        maximize: bool = True,
    ) -> None:
        r"""Single-outcome Max Variance.

        Args:
            model: A fitted single-outcome GP model (must be in batch mode if
                candidate sets X will be)
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, posterior_transform=posterior_transform)
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the posterior variance on the candidate set X.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.

        Returns:
            A `(b1 x ... bk)`-dim tensor of posterior variance values at the
            given design points `X`.
        """
        mean, sigma = self._mean_and_sigma(X)
        
        return sigma**2

if __name__ == '__main__':    
    
    dim = 20
    N_init = 40
    BO_iter = 1000
    lb = -2
    ub = 2
    n_bins = 50
   
    obj_lb = -7.79
    obj_ub =  0
   
    # Specify the synthetic function we want to study
    fun = Ackley(dim=dim, negate=True)
    
    replicate = 10
    RandomSearch = False # Random search or MaxVar

    coverage_list = [[] for _ in range(replicate)]

    for seed in range(replicate):

        sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
        train_X = sobol.draw(n=N_init).to(torch.float64)
        train_Y = fun(lb + (ub-lb)*train_X).unsqueeze(1)
        
        coverage = reachability_uniformity(train_Y, n_bins, obj_lb, obj_ub)
        coverage_list[seed].append(coverage)
        
        for bo_iter in range(BO_iter):
            likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
            covar_module = ScaleKernel(RBFKernel(ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0))) # select the RBF kernel
            gp = SingleTaskGP(train_X, train_Y, outcome_transform=Standardize(m=1), covar_module=covar_module, likelihood=likelihood)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            
            acquisition_function = MaxVariance(gp)
                      
            bounds = torch.tensor([[0.0]*dim, [1.0]*dim], dtype=torch.float)
                        
            if RandomSearch:
                candidate = torch.rand(1,dim)
            else:
                candidate, acq = optimize_acqf(
                    acquisition_function, bounds=bounds, q=1, num_restarts=10, raw_samples=512,
                )
                
            train_X = torch.cat((train_X, candidate))
            Y_next = fun(lb+(ub-lb)*candidate).unsqueeze(1)
            train_Y = torch.cat((train_Y, Y_next))
            
            coverage = reachability_uniformity(train_Y, n_bins, obj_lb, obj_ub)
            coverage_list[seed].append(coverage)
            
    coverage_tensor = torch.tensor(coverage_list, dtype=torch.float32) 

