# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 09:32:29 2019

@author: Andri
"""

"""
Variational Inference with Normalizing Flows
arXiv:1505.05770v6
"""

import torch
import torch.nn as nn
import torch.distributions as D

import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import argparse


parser = argparse.ArgumentParser()
# action
parser.add_argument('--train', default = True, action='store_true', help='Train a flow.')
# target potential
parser.add_argument('--target_potential', default = 'u_z1', choices=['u_z0', 'u_z5', 'u_z1', 'u_z2', 'u_z3', 'u_z4'], help='Which potential function to approximate.')
# flow params
parser.add_argument('--base_sigma', type=float, default=1, help='Std of the base isotropic 0-mean Gaussian distribution.')
parser.add_argument('--learn_base', default=True, action='store_true', help='Whether to learn a mu-sigma affine transform of the base distribution.')
parser.add_argument('--flow_length', type=int, default=2, help='Length of the flow.')
# training params
parser.add_argument('--init_sigma', type=float, default=1, help='Initialization std for the trainable flow parameters.')
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--start_step', type=int, default=0, help='Starting step (if resuming training will be overwrite from filename).')
parser.add_argument('--n_steps', type=int, default=1000000, help='Optimization steps.')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay.')
parser.add_argument('--beta', type=float, default=1, help='Multiplier for the target potential loss.')
parser.add_argument('--seed', type=int, default=2, help='Random seed.')


# --------------------
# Flow
# --------------------


class Planar_flow:
    
    class PlanarTransform(nn.Module):
        def __init__(self, init_sigma=0.01):
            super().__init__()
            self.u = nn.Parameter(torch.randn(1, 2).normal_(0, init_sigma))
            self.w = nn.Parameter(torch.randn(1, 2).normal_(0, init_sigma))
            self.b = nn.Parameter(torch.randn(1).fill_(0))
    
        def forward(self, x, normalize_u=True):
            # allow for a single forward pass over all the transforms in the flows with a Sequential container
            if isinstance(x, tuple):
                z, sum_log_abs_det_jacobians = x
            else:
                z, sum_log_abs_det_jacobians = x, 0
    
            # normalize u s.t. w @ u >= -1; sufficient condition for invertibility
            u_hat = self.u
            if normalize_u:
                wtu = (self.w @ self.u.t()).squeeze()
                m_wtu = - 1 + torch.log1p(wtu.exp())
                u_hat = self.u + (m_wtu - wtu) * self.w / (self.w @ self.w.t())
    
            # compute transform
            f_z = z + u_hat * torch.tanh(z @ self.w.t() + self.b)
            # compute log_abs_det_jacobian
            psi = (1 - torch.tanh(z @ self.w.t() + self.b)**2) @ self.w
            det = 1 + psi @ u_hat.t()
            log_abs_det_jacobian = torch.log(torch.abs(det) + 1e-6).squeeze()
            sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian
    
            return f_z, sum_log_abs_det_jacobians
    
    class AffineTransform(nn.Module):
        def __init__(self, learnable=True):
            super().__init__()
            self.mu = nn.Parameter(torch.zeros(2)).requires_grad_(learnable)
            self.logsigma = nn.Parameter(torch.zeros(2)).requires_grad_(learnable)
    
        def forward(self, x):
            z = self.mu + self.logsigma.exp() * x
            sum_log_abs_det_jacobians = self.logsigma.sum()
            return z, sum_log_abs_det_jacobians
        

    def optimize_flow(base_dist, flow, target_energy_potential, optimizer, args):
    
        # anneal rate for free energy
        temp = lambda i: min(1, 0.01 + i/10000)
    
        for i in range(args.start_step, args.n_steps):
    
            # sample base dist
            z = base_dist.sample((args.batch_size, ))
    
            # pass through flow:
            # 1. compute expected log_prob of data under base dist -- nothing tied to parameters here so irrelevant to grads
            base_log_prob = base_dist.log_prob(z)
            # 2. compute sum of log_abs_det_jacobian through the flow
            zk, sum_log_abs_det_jacobians = flow(z)
            # 3. compute expected log_prob of z_k the target_energy potential
            p_log_prob = - temp(i) * target_energy_potential(zk)  # p = exp(-potential) ==> p_log_prob = - potential
    
            loss = base_log_prob - sum_log_abs_det_jacobians - args.beta * p_log_prob
            loss = loss.mean(0)
    
            # compute loss and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



# --------------------
# Run
# --------------------

if __name__ == '__main__':

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # setup flow
    flow = nn.Sequential(AffineTransform(args.learn_base), *[PlanarTransform() for _ in range(args.flow_length)])

    # setup target potential to approx
    u_z = vars()[args.target_potential]

    # setup base distribution
    base_dist = D.MultivariateNormal(torch.zeros(2), args.base_sigma * torch.eye(2))


    if args.train:
        optimizer = torch.optim.RMSprop(flow.parameters(), lr=args.lr, momentum=0.9, alpha=0.90, eps=1e-6, weight_decay=args.weight_decay)
#        if optimizer_state: optimizer.load_state_dict(optimizer_state)
        args.n_steps = args.start_step + args.n_steps
        optimize_flow(base_dist, flow, u_z, optimizer, args)

