# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 08:15:22 2019

@author: Andri
"""

import torch
import torch.utils.data
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from sklearn.model_selection import KFold
from aif360.datasets import GermanDataset
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import StepLR

class Adversarial_weight_class:
    
    class Fair_classifier(nn.Module):
        def __init__(self, input_size):
            super(Adversarial_weight_class.Fair_classifier, self).__init__()
            
            self.fc1 = nn.Sequential(nn.Linear(input_size,20),
            nn.BatchNorm1d(num_features=20),
            nn.ReLU(),
            nn.Linear(20,1))    
            
            self.fc2 = nn.Sequential(nn.Linear(input_size,20),
            nn.BatchNorm1d(num_features=20),
            nn.ReLU(),
            nn.Linear(20,1))  
    
            self.fc3 = nn.Sequential(nn.Linear(input_size,20),
            nn.BatchNorm1d(num_features=20),
            nn.ReLU(),
            nn.Linear(20,1))        
    
        def forward(self, x):
            output_y = torch.sigmoid(self.fc1(x))
            output_A = torch.sigmoid(self.fc2(x))
            output_w = torch.sigmoid(self.fc3(x))
            return output_y, output_A, output_w

    def __init__(self, input_size):
        self.model = Adversarial_weight_class.Fair_classifier(input_size)


    def fit(self, x_train, y_train, A_train, max_epoch = 300, mini_batch_size = 50, 
            alpha = 1.0, beta = 0, log_epoch = 10, log = 1):
        
        def loss(output, target, weights):
            output = torch.clamp(output, 1e-5, 1 - 1e-5)
            weights = torch.clamp(weights, 1e-5, 1 - 1e-5)
            ML =  weights*(target*torch.log(output) + (1-target)*torch.log(1-output))
            return torch.neg(torch.mean(ML))

        self.model.train()
        nll_criterion =F.binary_cross_entropy
        list_0 = list(self.model.fc1.parameters())
        list_1 = list(self.model.fc3.parameters())
        list_2 = list(self.model.fc2.parameters())
        
        optimizer_0 = torch.optim.Adam(list_0, lr = 0.0001)
        optimizer_1 = torch.optim.Adam(list_1, lr = 0.0001)
        optimizer_2 = torch.optim.Adam(list_2, lr = 0.0001)
        
    
        for e in range(max_epoch):
            for i in range(0,x_train.size()[0], mini_batch_size):     
                batch_x, batch_y, batch_A = (x_train[i:i+mini_batch_size], y_train[i:i+mini_batch_size], 
                                            A_train[i:i+mini_batch_size])
                y, A, w = self.model(batch_x)
                loss0 = loss(y, batch_y, w) 
                optimizer_0.zero_grad()
                loss0.backward(retain_graph = True)
                optimizer_0.step()
                if e%1 == 0:
                    loss2 = loss(A, batch_A, w)
                    optimizer_2.zero_grad()
                    loss2.backward(retain_graph = True)
                    optimizer_2.step()
                    loss1 = loss(y, batch_y, w) - alpha*loss(A, batch_A, w) - beta*torch.norm(w,1)
                    optimizer_1.zero_grad()
                    loss1.backward()
                    optimizer_1.step()
    
                
            if e%log_epoch == 0 and log == 1:
                y, A, w = self.model(x_train)
                print(min(w.data),max(w.data),torch.mean(w).data,torch.sum(w).data)
                print(nll_criterion(A, A_train).data, nll_criterion(y, y_train).data)


    def predict(self, x_test):
        self.model.eval()
        y, A , w= self.model(x_test)
        y = np.round(y.data)
        A = np.round(A.data)
        w = w.data
        return y, A

    def predict_proba(self, x_test):
        self.model.eval()
        y, A , w= self.model(x_test)
        y = y.data
        A = A.data
        w = w.data
        return y, A


