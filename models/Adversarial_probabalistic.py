# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 10:33:01 2019

@author: Andri
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 09:49:48 2019

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




class Adversarial_prob_class():
    
    class FairClass(nn.Module):
        def __init__(self):
            super(Adversarial_prob_class.FairClass, self).__init__()
            self.fc1 = nn.Sequential(nn.Linear(56,20),
                    nn.BatchNorm1d(num_features=20),
                    nn.ReLU())
            
            self.fc21 = nn.Linear(20,10)
            self.fc22 = nn.Linear(20,10)
            
            self.fc31 = nn.Sequential(nn.Linear(10,5),
                                      nn.ReLU(),
                                      nn.Linear(5,1))
            self.fc41 = nn.Sequential(nn.Linear(10,5),
                                      nn.ReLU(),
                                      nn.Linear(5,1))        
        def P_zx(self, x):
            h1 = self.fc1(x)
            return self.fc21(h1), self.fc22(h1)
        
        def reparameterize(self, mu, logvar, no_samples):
            std = torch.exp(0.5*logvar)
            number = list(std.size())
            number.insert(0,no_samples)
            eps = torch.randn(number)
            return mu + eps*std
    
        def P_yz(self, z):
            y = torch.sigmoid(self.fc31(z))
            return torch.mean(y, dim=0)
        
        def P_Az(self, z):
            A = torch.sigmoid(self.fc41(z))
            return torch.mean(A, dim=0)
            
    
        def forward(self, x, no_samples = 100):
            mu, logvar = self.P_zx(x)
            z = self.reparameterize(mu, logvar, no_samples)
            output_y = self.P_yz(z)
            output_A = self.P_Az(z)
            return output_y, output_A
    
    def __init__(self):
        self.model = Adversarial_prob_class.FairClass()

    def fit(self, x_train, y_train, A_train, max_epoch = 100, mini_batch_size = 50, alpha = 1,
            log_epoch = 1, log = 1, no_sample = 1):
        self.model.train()
        
        nll_criterion = F.binary_cross_entropy
        list_z = list(self.model.fc1.parameters())+list(self.model.fc21.parameters())+list(self.model.fc22.parameters())
        list_1 = list(self.model.fc31.parameters())+list_z
        list_2 = list(self.model.fc41.parameters())
        optimizer_1 = torch.optim.Adam(list_1, lr = 0.001)
        optimizer_2 = torch.optim.Adam(list_2, lr = 0.001)
        
        for e in range(max_epoch):
            for i in range(0,x_train.size()[0], mini_batch_size):     
                batch_x, batch_y, batch_A = (x_train[i:i+mini_batch_size], y_train[i:i+mini_batch_size], 
                                            A_train[i:i+mini_batch_size])
                output_1, output_2 = self.model(batch_x, no_samples = no_sample)
                
                loss2 = nll_criterion(output_2, batch_A)
                optimizer_2.zero_grad()
                loss2.backward(retain_graph=True)
                optimizer_2.step()
                
                loss1 = nll_criterion(output_1, batch_y) - alpha * nll_criterion(output_2,batch_A)
                optimizer_1.zero_grad()
                loss1.backward()
                optimizer_1.step()
                
            if e%log_epoch == 0 and  log == 1:
                out_1, out_2 = self.model(x_train)
                print(nll_criterion(out_2, A_train).data, nll_criterion(out_1, y_train).data)

    def predict(self, x_test, no_samples = 100):
        self.model.eval()
        y, A = self.model(x_test, no_samples)
        y = np.round(y.data)
        A = np.round(A.data)
        return y, A
    
    def predict_proba(self, x_test):
        self.model.eval()
        y, A = self.model(x_test, no_samples = 100)
        y = y.data
        A = A.data
        return y, A
    

