#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:25:40 2024

@author: tang.1856
"""


import torch
from torch.quasirandom import SobolEngine
from scipy.spatial.distance import cdist, jensenshannon
import numpy as np
from torch.quasirandom import SobolEngine
from botorch.test_functions import Rosenbrock, Ackley, Hartmann, StyblinskiTang


def reachability_uniformity(behavior, n_bins = 25, obj_lb = -5, obj_ub = 5):
    behavior = behavior.squeeze(1).numpy()
    num = len(behavior)
    cum_hist, _ = np.histogram(behavior, np.linspace(obj_lb, obj_ub, n_bins + 1))
    cum_hist = cum_hist[np.nonzero(cum_hist)] / (num) 
    cum_coverage = len(cum_hist) / n_bins
    return cum_coverage
    

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
    obj_ub =  0 

    function = Ackley(dim=dim, negate=True)
    
    coverage_tensor = []
      
    for seed in range(replicate):
       
        sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
        train_x = sobol.draw(n=N_init).to(torch.float64)
        train_y = function((lb+(ub-lb)*train_x)).unsqueeze(1)
    
        coverage = reachability_uniformity(train_y, n_bins, obj_lb, obj_ub) 
        
        coverage_list = [coverage]
                
        for i in range(BO_iter):        
            
            sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
            train_x = sobol.draw(n=int(N_init+i+1)).to(torch.float64)
            train_y = function((lb+(ub-lb)*train_x)).unsqueeze(1)
                      
            coverage = reachability_uniformity(train_y, n_bins, obj_lb, obj_ub)
            coverage_list.append(coverage)
            
        coverage_tensor.append(coverage_list)
      
    coverage_tensor = torch.tensor(coverage_tensor, dtype=torch.float32) 
