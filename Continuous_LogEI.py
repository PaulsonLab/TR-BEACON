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
import gpytorch
import time
from botorch.acquisition import ExpectedImprovement, LogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.constraints import Interval
from gpytorch.likelihoods import GaussianLikelihood
from torch.quasirandom import SobolEngine

def reachability_uniformity(behavior, n_bins = 25, obj_lb = -5, obj_ub = 5):
    behavior = behavior.squeeze(1).numpy()
    num = len(behavior)
    cum_hist, _ = np.histogram(behavior, np.linspace(obj_lb, obj_ub, n_bins + 1))
    cum_hist = cum_hist[np.nonzero(cum_hist)] / (num) # discrete distribution
    cum_hist_uni = np.mean(cum_hist) * np.ones_like(cum_hist) # theoretical uniform distribution

    cum_coverage = len(cum_hist) / n_bins
    cum_uniformity = 1 - jensenshannon(cum_hist, cum_hist_uni, base=2)
    
    return cum_coverage, cum_uniformity



if __name__ == '__main__':    
    
    dim = 20
    N_init = 40
    BO_iter = 1000
    lb = -2
    ub = 2
    n_bins = 50
    replicate = 10
   
    # Specify the minimum/maximum value for each synthetic function as to calculate reachability
    obj_lb = -7.79
    obj_ub =  0# maximum obj value for Ackley
   
    # Specify the synthetic function we want to study
    
    # fun = Rosenbrock(dim=dim)
    fun = Ackley(dim=dim, negate=True)
    # fun = StyblinskiTang(dim=dim)
    
    
    RandomSearch =False# Random search or MaxVar
      
    cost_list = [[] for _ in range(replicate)]
    coverage_list = [[] for _ in range(replicate)]

    for seed in range(replicate):
      
        print('seed:',seed)
       
        # train_X = torch.tensor(np.random.rand(N_init, dim))
        sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
        train_X = sobol.draw(n=N_init).to(torch.float64)
        train_Y = fun(lb + (ub-lb)*train_X).unsqueeze(1).to(torch.float64)
        
        cost_list[seed].append(0)
        coverage, uniformity = reachability_uniformity(train_Y, n_bins, obj_lb, obj_ub)
        # variance = float(train_Y.var())
        coverage_list[seed].append(coverage)
        
        for bo_iter in range(BO_iter):
            likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
            covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0))) # Choose the RBF kernel
            gp = SingleTaskGP(train_X, train_Y, covar_module = covar_module, outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            
            try:
                fit_gpytorch_mll(mll)
            except:
                print('cant fit GP')
            
            # acquisition_function = MaxVariance(gp)
            acquisition_function = LogExpectedImprovement(model=gp, best_f=max(train_Y))            
                      
            bounds = torch.tensor([[0.0]*dim, [1.0]*dim], dtype=torch.float)
                        
            if RandomSearch:
                candidate = torch.rand(1,dim)
            else:
                candidate, acq = optimize_acqf(
                    acquisition_function, bounds=bounds, q=1, num_restarts=10, raw_samples=20,
                )
                
            train_X = torch.cat((train_X, candidate))
            Y_next = fun(lb+(ub-lb)*candidate).unsqueeze(1)
            train_Y = torch.cat((train_Y, Y_next))
            
            coverage, uniformity = reachability_uniformity(train_Y, n_bins, obj_lb, obj_ub)
            # variance = float(train_Y.var())
            coverage_list[seed].append(coverage)
            cost_list[seed].append(cost_list[seed][-1]+1)
            
        
        
    cost_tensor = torch.tensor(cost_list, dtype=torch.float32) 
    coverage_tensor = torch.tensor(coverage_list, dtype=torch.float32) 
    torch.save(coverage_tensor, '20DAckley_coverage_LogEI.pt')
    # torch.save(cost_tensor, '8DAckley_cost_list_EI.pt')  
    # torch.save(time_tensor, '20DRosen_time_list_RS.pt')
