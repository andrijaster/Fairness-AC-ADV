# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 15:04:00 2019

@author: Andri
"""

import torch
import torch.utils.data
import numpy as np
from torch import nn, optim
from torch.nn import functional as F

class Adversarial_weight_soft_class():
    class Fair_classifier(nn.Module):
        def __init__(self):
            super(Adversarial_weight_soft_class.Fair_classifier, self).__init__()
            
            self.fc1 = nn.Sequential(nn.Linear(56,20),
            nn.BatchNorm1d(num_features=20),
            nn.ReLU(),
            nn.Linear(20,1))    
            
            self.fc2 = nn.Sequential(nn.Linear(56,20),
            nn.BatchNorm1d(num_features=20),
            nn.ReLU(),
            nn.Linear(20,1))  
    
            self.fc3 = nn.Sequential(nn.Linear(56,20),
            nn.BatchNorm1d(num_features=20),
            nn.ReLU(),
            nn.Linear(20,1))        
            
            self.fc4 = nn.Sequential(nn.Linear(56,20),
            nn.BatchNorm1d(num_features=20),
            nn.ReLU(),
            nn.Linear(20,1))        
    
    
        def forward(self, x):
            output_y = torch.sigmoid(self.fc1(x))
            output_A = torch.sigmoid(self.fc2(x))
            output_alfa = torch.max(torch.ones(len(x),1)*1e-5, torch.exp(self.fc3(x))) 
            output_beta = torch.max(torch.ones(len(x),1)*1e-5, torch.exp(self.fc4(x)))
            return output_y, output_A, output_alfa, output_beta

    def __init__(self):
        self.model = Adversarial_weight_soft_class.Fair_classifier()


    def fit(self, x_train, y_train, A_train, max_epoch = 300, mini_batch_size = 50, 
            alpha = 1, beta = 0, log_epoch = 10, log = 1):
        
        def loss(output, target, weights):
            output = torch.clamp(output, 1e-5, 1 - 1e-5)
            weights = torch.clamp(weights, 1e-5, 1 - 1e-5)
            ML =  weights*(target*torch.log(output) + (1-target)*torch.log(1-output))
            return torch.neg(torch.mean(ML))

        def loss_w(output_A, output_Y, target_A, target_Y, alpha, beta, dist, sam):
            output_A = torch.clamp(output_A, 1e-5, 1 - 1e-5)
            output_Y = torch.clamp(output_Y, 1e-5, 1 - 1e-5)
            ML = dist.log_prob(sam)*(loss(output_Y, target_Y, sam) - alpha*loss(output_A, target_A, sam) - beta * torch.mean(sam))
            return torch.neg(torch.sum(ML))
        
        
        self.model.train()
        nll_criterion =F.binary_cross_entropy
        list_0 = list(self.model.fc1.parameters())
        list_1 = list(self.model.fc3.parameters()) + list(self.model.fc4.parameters())
        list_2 = list(self.model.fc2.parameters())
        
        optimizer_0 = torch.optim.Adam(list_0, lr = 0.0001)
        optimizer_1 = torch.optim.Adam(list_1, lr = 0.0001)
        optimizer_2 = torch.optim.Adam(list_2, lr = 0.0001)
    
        for e in range(max_epoch):
            for i in range(0,x_train.size()[0], mini_batch_size):     
                batch_x, batch_y, batch_A = (x_train[i:i+mini_batch_size], y_train[i:i+mini_batch_size], 
                                            A_train[i:i+mini_batch_size])
                y, A, alfa, beta = self.model(batch_x)
                
                dist = torch.distributions.Beta(alfa, beta)
                w = dist.sample()
                
                optimizer_0.zero_grad()
                loss0 = loss(y, batch_y, w) 
                loss0.backward(retain_graph = True)
                optimizer_0.step()
                
                optimizer_2.zero_grad()           
                loss2 = loss(A, batch_A, w)
                loss2.backward(retain_graph = True)
                optimizer_2.step()
                
                optimizer_1.zero_grad()
                loss1 = loss_w(A, y, batch_A, batch_y, alpha, beta, dist, w)
                loss1.backward()
                optimizer_1.step()
    
                
            if e%log_epoch == 0 and log == 1:
                y, A, alfa, beta = self.model(x_train)
                dist = torch.distributions.Beta(alfa, beta)
                w = dist.sample()
                print(min(w.data),max(w.data),torch.mean(w).data,torch.sum(w).data)
                print(nll_criterion(A, A_train).data, nll_criterion(y, y_train).data)
    

    def predict(self, x_test):
        self.model.eval()
        y,A,_,_= self.model(x_test)
        y = np.round(y.data)
        A = np.round(A.data)
        return y,A

    def predict_proba(self, x_test):
        self.model.eval()
        y,A,_,_= self.model(x_test)
        y = y.data
        A = A.data
        return y,A