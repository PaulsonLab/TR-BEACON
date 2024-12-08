#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:31:13 2024

@author: tang.1856
"""

import torch
import gpytorch
from botorch.models import SingleTaskGP
from gpytorch.kernels import MaternKernel, RFFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine
from botorch.fit import fit_gpytorch_mll
from botorch.utils import standardize
import botorch
from typing import Tuple
from botorch.utils.transforms import t_batch_mode_transform
from botorch.acquisition import AcquisitionFunction, AnalyticAcquisitionFunction
from botorch.optim import optimize_acqf
from scipy.spatial.distance import cdist, jensenshannon
import numpy as np
from torch.quasirandom import SobolEngine
from botorch.test_functions import Rosenbrock, Ackley, Hartmann, StyblinskiTang
import pickle
from botorch.models.transforms.outcome import Standardize
import matplotlib.pyplot as plt
import time

def reachability_uniformity(behavior, n_bins = 25, obj_lb = -5, obj_ub = 5):
    behavior = behavior.squeeze(1).numpy()
    num = len(behavior)
    cum_hist, _ = np.histogram(behavior, np.linspace(obj_lb, obj_ub, n_bins + 1))
    cum_hist = cum_hist[np.nonzero(cum_hist)] / (num) # discrete distribution
    cum_coverage = len(cum_hist) / n_bins   
    return cum_coverage
    
class CustomAcquisitionFunction(AnalyticAcquisitionFunction):
    def __init__(self, model, sampled_X, k=10):
        '''Inits acquisition function with model.'''
        super().__init__(model=model)
        
        self.model = model
        self.k = k
        self.sampled_X = sampled_X
        
    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        """Compute the acquisition function value at X."""
        
        dist = torch.norm(X - self.sampled_X, dim=2)
        dist, _ = torch.sort(dist, dim = 1) # sort the distance 
        n = dist.size()[1]

        E = torch.cat((torch.ones(self.k), torch.zeros(n-self.k)), dim = 0) # find the k-nearest neighbor
        dist = dist*E
        acquisition_values = torch.sum(dist,dim=1)
        
        return acquisition_values.flatten()

if __name__ == '__main__':
    
    lb = -2
    ub = 2
    dim = 20
    N_init = 40
    replicate = 20
    BO_iter = 1000
    n_bins = 50
    TS = 1
    k = 10
    
    obj_lb = -7.79
    obj_ub =  0
    
    function = Ackley(dim=dim, negate=True)
    
    coverage_tensor = []

    for seed in range(replicate):
       
        sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
        train_x = sobol.draw(n=N_init).to(torch.float64)
        train_y = function((lb+(ub-lb)*train_x)).unsqueeze(1)    
        coverage = reachability_uniformity(train_y, n_bins, obj_lb, obj_ub) # Calculate the initial reachability and uniformity
        
        coverage_list = [coverage]
     
        for i in range(BO_iter):        

            # We actually don't need the GP
            model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)

            custom_acq_function = CustomAcquisitionFunction(model, train_x, k=k)
                
            bounds = torch.tensor([[0.0]*dim, [1.0]*dim], dtype=torch.float)  # Define bounds of the feature space
            # Optimize the acquisition function (continuous)
            candidate, acq_value = optimize_acqf(
                acq_function=custom_acq_function,
                bounds=bounds,
                q=1,  # Number of candidates to generate
                num_restarts=10,  # Number of restarts for the optimizer
                raw_samples=512,  # Number of initial raw samples to consider
            )
                
              
            train_x = torch.cat((train_x, candidate))
            y_new = function(lb+(ub-lb)*candidate).unsqueeze(1)
            train_y = torch.cat((train_y, y_new))
            
            coverage = reachability_uniformity(train_y, n_bins, obj_lb, obj_ub)
            coverage_list.append(coverage)         
         
        coverage_tensor.append(coverage_list)
    
    coverage_tensor = torch.tensor(coverage_tensor, dtype=torch.float32)   