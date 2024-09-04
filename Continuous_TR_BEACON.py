#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 19:29:19 2024

@author: tang.1856
"""

import os
import math
import warnings
from dataclasses import dataclass

import torch
from botorch.acquisition import qExpectedImprovement, qLogExpectedImprovement
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.test_functions import Ackley, Rosenbrock, StyblinskiTang
from botorch.utils.transforms import unnormalize
from torch.quasirandom import SobolEngine
import numpy as np
import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import AcquisitionFunction, AnalyticAcquisitionFunction
from ThompsonSampling import EfficientThompsonSampler
from botorch.utils.transforms import t_batch_mode_transform
from botorch.models.transforms.outcome import Standardize

device = torch.device("cpu")
dtype = torch.float64

fun = Ackley(dim=20, negate=True).to(dtype=dtype, device=device)
fun.bounds[0, :].fill_(-5)
fun.bounds[1, :].fill_(5)
dim = fun.dim
lb, ub = fun.bounds
lb = lb.to(dtype)
ub = ub.to(dtype)

batch_size = 1
n_init = 2 * dim
k_NN = 10
max_cholesky_size = float("inf")  # Always use Cholesky
n_bins = 50
obj_lb = -14.30 # minimum obj value for Ackley
# obj_lb = -10.3947
# obj_lb = -7.79
obj_ub =  0 # maximum obj value for Ackley
# obj_lb = -1710685.71069 
# obj_lb = -68571 # minimum obj value for Rosenbrock
# obj_ub = 0 # obj maximum for Rosenbrock
# obj_ub = 39.16599*dim # minimum obj val for 4D SkyTang
# obj_lb = -125*dim # maximum obj val for 4D SkyTang

def reachability_uniformity(behavior, n_bins = 25, obj_lb = -5, obj_ub = 5):
    behavior = behavior.squeeze(1).numpy()
    num = len(behavior)
    cum_hist, _ = np.histogram(behavior, np.linspace(obj_lb, obj_ub, n_bins + 1))
    cum_hist = cum_hist[np.nonzero(cum_hist)] / (num) # discrete distribution
    cum_hist_uni = np.mean(cum_hist) * np.ones_like(cum_hist) # theoretical uniform distribution

    cum_coverage = len(cum_hist) / n_bins
    # cum_uniformity = 1 - jensenshannon(cum_hist, cum_hist_uni, base=2)
    
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
    
def eval_objective(x):
    """This is a helper function we use to unnormalize and evalaute a point"""
    return fun(unnormalize(x, fun.bounds))


@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10  # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False
    

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )


def update_state(state, Y_next, Y_sampled, x_next):
    # sum_of_distance = torch.sum(torch.cdist(Y_next, Y_sampled)**2)**0.5
    # sum_of_distance = torch.sum(torch.sort(torch.cdist(Y_next, Y_sampled), dim=-1).values[:,0:40]**2, dim=-1)**0.5
    
    sum_of_distance = torch.cat((Y_sampled, Y_next)).var()
    
    # if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
    #     state.success_counter += 1
    #     state.failure_counter = 0
    
    if sum_of_distance > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
        # state.x_center = x_next[0].clone()
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    # state.best_value = max(state.best_value, max(Y_next).item())
    state.best_value = max(state.best_value, sum_of_distance.item())
    if state.length < state.length_min:
        state.restart_triggered = True
    return state






def get_initial_points(dim, n_pts, seed):
    sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
    X_init = sobol.draw(n=n_pts).to(dtype=dtype, device=device)
    return X_init

def generate_batch(
    state,
    model,  # GP model
    X,  # Evaluated points on the domain [0, 1]^d
    Y,  # Function values
    batch_size,
    n_candidates=None,  # Number of candidates for Thompson sampling
    num_restarts=10,
    raw_samples=512,
    acqf="ts",  # "ei" or "ts"
):
    assert acqf in ("ts", "ei")
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

    # Scale the TR to be proportional to the lengthscales
    x_center = X[torch.argmax(torch.sum(torch.cdist(Y, Y), dim=-1))].clone() # select point that has the largest sum of distance from all other sampled points
    # x_center = X[Y.argmax(), :].clone()
   
    weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

    
    # acq_fun = qLogExpectedImprovement(model, Y_turbo.max())
    acq_fun = CustomAcquisitionFunction(model, Y, k_NN)
    X_next, acq_value = optimize_acqf(
        acq_fun,
        bounds=torch.stack([tr_lb, tr_ub]),
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )

    return X_next

BO_iter = 1500
replicate = 10
coverage_list = [[] for _ in range(replicate)]
for seed in range(replicate):

    state = TurboState(dim=dim, batch_size=batch_size)
    # print(state)
    
    X_turbo = get_initial_points(dim, n_init, seed)
    Y_turbo = torch.tensor(
        [eval_objective(x) for x in X_turbo], dtype=dtype, device=device
    ).unsqueeze(-1)
    coverage = reachability_uniformity(Y_turbo, n_bins, obj_lb, obj_ub)
    coverage_list[seed].append(coverage)
    # best_value = torch.max(torch.sum(torch.cdist(Y_turbo, Y_turbo)**2, dim=-1)**0.5)
    # best_value = torch.sum(torch.sort(torch.cdist(Y_turbo, Y_turbo), dim=-1).values[:,0:40]**2, dim=-1)**0.5
    # best_value = torch.max(best_value)
    
    best_value = Y_turbo.var()
    state = TurboState(dim, batch_size=batch_size, best_value=best_value.item())
    
    NUM_RESTARTS = 10 
    RAW_SAMPLES = 512
    N_CANDIDATES = min(5000, max(2000, 200 * dim))
    
    torch.manual_seed(seed)
    
    # while not state.restart_triggered:  # Run until TuRBO converges
    for _ in range(BO_iter):
        # Fit a GP model
        # train_Y = (Y_turbo - Y_turbo.mean()) / Y_turbo.std()
        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        # covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
        #     MaternKernel(
        #         nu=2.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0)
        #     )
        # )
        covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
            RBFKernel(
                ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0)
            )
        )
        model = SingleTaskGP(
            X_turbo, Y_turbo, covar_module=covar_module, likelihood=likelihood, outcome_transform=Standardize(1)
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
    
        # Do the fitting and acquisition function optimization inside the Cholesky context
        with gpytorch.settings.max_cholesky_size(max_cholesky_size):
            # Fit the model
            try:
                fit_gpytorch_mll(mll)
            except:
                print('fail to fit GP!')
            
            model.train_x = X_turbo
            model.train_y = Y_turbo
            
            # Create a batch
            X_next = generate_batch(
                state=state,
                model=model,
                X=X_turbo,
                Y=Y_turbo,
                batch_size=batch_size,
                n_candidates=N_CANDIDATES,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,
                acqf="ts",
            )
    
        Y_next = torch.tensor(
            [eval_objective(x) for x in X_next], dtype=dtype, device=device
        ).unsqueeze(-1)
    
        # Update state
        state = update_state(state=state, Y_next=Y_next, Y_sampled = Y_turbo, x_next=X_next)
    
        # Append data
        X_turbo = torch.cat((X_turbo, X_next), dim=0)
        Y_turbo = torch.cat((Y_turbo, Y_next), dim=0)
        coverage = reachability_uniformity(Y_turbo, n_bins, obj_lb, obj_ub)
        coverage_list[seed].append(coverage)
        print('reachability=', coverage)
        # Print current status
        print(
            f"{len(X_turbo)}) Best value: {state.best_value:.2e}, TR length: {state.length:.2e}"
        )

coverage_list = torch.tensor(coverage_list)
torch.save(coverage_list, '20DAckley_coverage_TR_BEACON.pt')

















