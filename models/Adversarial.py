# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 09:49:48 2019

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



class Adversarial_class():
    
    def __init__(self):
        self.model = Adversarial_class.FairClass()
        
    
    class FairClass(nn.Module):
        def __init__(self):
            super(Adversarial_class.FairClass, self).__init__()
            self.fc1 = nn.Sequential(nn.Linear(56,20),
                            nn.BatchNorm1d(num_features=20),
                            nn.ReLU())
            self.fc2 = nn.Linear(20,1)
            self.fc3 = nn.Linear(20,1)
            
        def forward(self, x):
            z = self.fc1(x)
            y = torch.sigmoid(self.fc2(z))
            A = torch.sigmoid(self.fc3(z))
            return y, A
        
    def fit(self, x_train, y_train, A_train, max_epoch = 300, mini_batch_size = 50, 
            alpha = 1, log = 1, log_epoch = 10):
        
        self.model.train()
        nll_criterion =F.binary_cross_entropy
        list_1 = list(self.model.fc1.parameters())+list(self.model.fc2.parameters())
        list_2 = list(self.model.fc3.parameters())
        optimizer_1 = torch.optim.Adam(list_1, lr = 0.001)
        optimizer_2 = torch.optim.Adam(list_2, lr = 0.001)
        for e in range(max_epoch):
            for i in range(0,x_train.size()[0], mini_batch_size):     
                batch_x, batch_y, batch_A = (x_train[i:i+mini_batch_size], y_train[i:i+mini_batch_size], 
                                            A_train[i:i+mini_batch_size])
                output_1, output_2 = self.model(batch_x)
                loss2 = nll_criterion(output_2, batch_A)
                optimizer_2.zero_grad()
                loss2.backward(retain_graph=True)
                optimizer_2.step()
                loss1 = nll_criterion(output_1, batch_y) - alpha * nll_criterion(output_2,batch_A)
                optimizer_1.zero_grad()
                loss1.backward()
                optimizer_1.step()
            if e%log_epoch==0 and log == 1:
                out_1, out_2 = self.model(x_train)
                print(nll_criterion(out_2, A_train).data, nll_criterion(out_1, y_train).data)
                
    def predict(self, x_test):
        self.model.eval()
        y, A = self.model(x_test)
        y = np.round(y.data)
        A = np.round(A.data)
        return y, A
    
    def predict_proba(self, x_test):
        self.model.eval()
        y, A = self.model(x_test)
        y, A = y.data, A.data
        return y, A





        
    
    
    
    