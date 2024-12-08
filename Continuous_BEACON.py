#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 13:26:09 2024

@author: tang.1856
"""
import torch
from botorch.models import SingleTaskGP
from gpytorch.kernels import MaternKernel, RFFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine
from botorch.fit import fit_gpytorch_mll
from botorch.utils.transforms import t_batch_mode_transform
from botorch.acquisition import AcquisitionFunction, AnalyticAcquisitionFunction
from botorch.optim import optimize_acqf
from scipy.spatial.distance import cdist, jensenshannon
import numpy as np
from torch.quasirandom import SobolEngine
from botorch.test_functions import Rosenbrock, Ackley, Hartmann, StyblinskiTang
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import RBFKernel, ScaleKernel
from ThompsonSampling import EfficientThompsonSampler
import time
from gpytorch.constraints import Interval

def reachability_uniformity(behavior, n_bins = 25, obj_lb = -5, obj_ub = 5):
    behavior = behavior.squeeze(1).numpy()
    num = len(behavior)
    cum_hist, _ = np.histogram(behavior, np.linspace(obj_lb, obj_ub, n_bins + 1))
    cum_hist = cum_hist[np.nonzero(cum_hist)] / (num) # discrete distribution
    cum_coverage = len(cum_hist) / n_bins  
    return cum_coverage
    
class CustomAcquisitionFunction(AnalyticAcquisitionFunction):
    def __init__(self, model, sampled_behavior, k=10):
        '''Inits acquisition function with model.'''
        super().__init__(model=model)
        
        self.model = model
        self.k = k
        self.sampled_behavior = sampled_behavior
       
        self.ts_sampler = EfficientThompsonSampler(model)
        self.ts_sampler.create_sample()
        
    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        """Compute the acquisition function value at X."""
     
        samples = self.ts_sampler.query_sample(X) # Thompson sampling
        dist = torch.cdist(samples, self.sampled_behavior) # Calculate Euclidean distance between TS and all sampled point
        dist, _ = torch.sort(dist, dim = 1) # sort the distance 
        n = dist.size()[1]
        E = torch.cat((torch.ones(self.k), torch.zeros(n-self.k)), dim = 0) # find the k-nearest neighbor
        dist = dist*E
        acquisition_values = torch.sum(dist,dim=1)
              
        return acquisition_values.flatten()


if __name__ == '__main__':
    
    lb = -2 # lower bound for feature
    ub = 2 # upper bound for feature
    dim = 20 # feature dimension
    N_init = 40 # number of initial training data
    replicate = 10# number of replicates for experiment
    BO_iter = 1000 # number of evaluations
    n_bins = 50 # grid number for calculating reachability
    TS = 1 # number of TS (posterior sample)
    k = 10 # k-nearest neighbor
    
    obj_lb = -7.79
    obj_ub = 0

    function = Ackley(dim=dim, negate=True)
    coverage_tensor = [] # list containing reachability for every itertation
    
    for seed in range(replicate): 
              
        sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
        train_x = sobol.draw(n=N_init).to(torch.float64)              
        train_y = function((lb+(ub-lb)*train_x)).unsqueeze(1)
    
        coverage = reachability_uniformity(train_y, n_bins, obj_lb, obj_ub) # Calculate the initial reachability
        coverage_list = [coverage]    
             
        # Start BEACON loop
        for i in range(BO_iter):        
            
            likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
            covar_module = ScaleKernel(RBFKernel(ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0))) 
            model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=1), covar_module=covar_module, likelihood=likelihood)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            
            fit_gpytorch_mll(mll)
            
            model.train_x = train_x
            model.train_y = train_y
           
            custom_acq_function = CustomAcquisitionFunction(model, train_y, k=k)                
            bounds = torch.tensor([[0.0]*dim, [1.0]*dim], dtype=torch.float64)  # Define bounds of the feature space (always operate within [0,1]^d)
            # Optimize the acquisition function
            candidate, acq_value = optimize_acqf(
                acq_function=custom_acq_function,
                bounds=bounds,
                q=1,  
                num_restarts=10,  
                raw_samples=512,  
            )
                               
            train_x = torch.cat((train_x, candidate))
            y_new = function(lb+(ub-lb)*candidate).unsqueeze(1)
            train_y = torch.cat((train_y, y_new))
            
            coverage = reachability_uniformity(train_y, n_bins, obj_lb, obj_ub)
            coverage_list.append(coverage)      
            
        coverage_tensor.append(coverage_list)
 
    coverage_tensor = torch.tensor(coverage_tensor, dtype=torch.float32)   

