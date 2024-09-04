#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:25:40 2024

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

def reachability_uniformity(behavior, n_bins = 25, obj_lb = -5, obj_ub = 5):
    behavior = behavior.squeeze(1).numpy()
    num = len(behavior)
    cum_hist, _ = np.histogram(behavior, np.linspace(obj_lb, obj_ub, n_bins + 1))
    cum_hist = cum_hist[np.nonzero(cum_hist)] / (num) 
    cum_hist_uni = np.mean(cum_hist) * np.ones_like(cum_hist) 

    cum_coverage = len(cum_hist) / n_bins
    cum_uniformity = 1 - jensenshannon(cum_hist, cum_hist_uni, base=2)
    
    return cum_coverage, cum_uniformity
    

if __name__ == '__main__':
    
    lb = -2
    ub = 2
    dim = 20
    N_init = 40
    replicate = 10
    BO_iter = 1000
    n_bins = 50
    TS = 1 
    k = 10
    
    obj_lb = -7.79
    obj_ub =  0 # maximum obj value for Ackley
    # Specify the minimum/maximum value for each synthetic function as to calculate reachability
       
    # Specify the synthetic function we want to study    
    # function = Rosenbrock(dim=dim)
    function = Ackley(dim=dim, negate=True)
    # function = StyblinskiTang(dim=dim)
    
    cost_tensor = []
    coverage_tensor = []
      
    for seed in range(replicate):
        print('seed:',seed)
        np.random.seed(seed)
        # train_x = torch.tensor(np.random.rand(N_init, dim))
        sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
        train_x = sobol.draw(n=N_init).to(torch.float64)
        train_y = function((lb+(ub-lb)*train_x)).unsqueeze(1)
    
        coverage, uniformity = reachability_uniformity(train_y, n_bins, obj_lb, obj_ub) 
        
        coverage_list = [coverage]
        cost_list = [0]
                
        # Start BO loop
        for i in range(BO_iter):        
            
            sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
            train_x = sobol.draw(n=int(N_init+i+1)).to(torch.float64)
            train_y = function((lb+(ub-lb)*train_x)).unsqueeze(1)
                      
            coverage, uniformity = reachability_uniformity(train_y, n_bins, obj_lb, obj_ub)
            coverage_list.append(coverage)
            cost_list.append(cost_list[-1]+1)
            
        cost_tensor.append(cost_list)
        coverage_tensor.append(coverage_list)
       
    cost_tensor = torch.tensor(cost_tensor, dtype=torch.float32) 
    coverage_tensor = torch.tensor(coverage_tensor, dtype=torch.float32) 
    torch.save(coverage_tensor, '20DAckley_coverage_sobol.pt')
    # torch.save(cost_tensor, '20DRosen_cost_list_sobol.pt')  
