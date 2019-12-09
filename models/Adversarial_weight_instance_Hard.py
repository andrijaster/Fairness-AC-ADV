# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 10:32:12 2019

@author: Andri
"""

import torch
import torch.utils.data
import numpy as np
from torch.nn import functional as F
from torch import nn

class Adversarial_weight_hard_class():

    class Output_class(nn.Module):
        def __init__(self, input_size):
            super(Adversarial_weight_hard_class.Output_class, self).__init__()
            
            self.fc1 = nn.Sequential(nn.Linear(input_size,20),
            nn.BatchNorm1d(num_features=20),
            nn.ReLU(),
            nn.Linear(20,1))        
    
        def forward(self, x):
            output_y = torch.sigmoid(self.fc1(x))
            return output_y
        
        
    class Atribute_class(nn.Module):
        def __init__(self, input_size):
            super(Adversarial_weight_hard_class.Atribute_class, self).__init__()
            
            self.fc2 = nn.Sequential(nn.Linear(input_size,20),
            nn.BatchNorm1d(num_features=20),
            nn.ReLU(),
            nn.Linear(20,1))        
    
        def forward(self, x):
            u = self.fc2(x)
            output_A = torch.sigmoid(u)
            return output_A    
        
    class weight_class(nn.Module):
        def __init__(self, input_size):
            super(Adversarial_weight_hard_class.weight_class, self).__init__()      
            self.fc3 = nn.Sequential(nn.Linear(input_size,20),
            nn.BatchNorm1d(num_features=20),
            nn.ReLU(),
            nn.Linear(20,1))       
            
        def forward(self, x):
            output_w = torch.sigmoid(self.fc3(x))
            return output_w    
    

    def __init__(self, input_size):
        self.model_y = Adversarial_weight_hard_class.Output_class(input_size)
        self.model_A = Adversarial_weight_hard_class.Atribute_class(input_size)
        self.model_w = Adversarial_weight_hard_class.weight_class(input_size)


    def fit(self, x_train, y_train, A_train, max_epoch = 200, 
            mini_batch_size = 50, alpha = 1, beta = 1, log_epoch = 10, log = 1):
        
        def loss_ML_sam(output, target, sam):
            output = torch.clamp(output, 1e-5, 1 - 1e-5)
            ML =  sam*(target*torch.log(output) + (1-target)*torch.log(1-output))
            return torch.neg(torch.sum(ML))
    
        def loss_w(output_A, output_Y, target_A, target_Y, alpha, beta, weights, sam):
            output_A = torch.clamp(output_A, 1e-5, 1 - 1e-5)
            output_Y = torch.clamp(output_Y, 1e-5, 1 - 1e-5)
            weights = torch.clamp(weights, 1e-5, 1 - 1e-5)
            ML = torch.log(weights)*(loss_ML_sam(output_Y, target_Y, sam) - alpha*loss_ML_sam(output_A, target_A, sam) - beta * torch.mean(sam))
            return torch.neg(torch.sum(ML))
        
        self.model_y.train()
        self.model_A.train()
        self.model_w.train()
        nll_criterion =F.binary_cross_entropy
        list_1 = list(self.model_y.parameters()) 
        list_2 = list(self.model_A.parameters())
        list_3 = list(self.model_w.parameters())
        
        optimizer_1 = torch.optim.Adam(list_1, lr = 0.0001)
        optimizer_2 = torch.optim.Adam(list_2, lr = 0.0001)
        optimizer_3 = torch.optim.Adam(list_3, lr = 0.0001)
        
        for e in range(max_epoch):
            for i in range(0,x_train.size()[0], mini_batch_size):     
                batch_x, batch_y, batch_A = (x_train[i:i+mini_batch_size], y_train[i:i+mini_batch_size], 
                                            A_train[i:i+mini_batch_size])
                A = self.model_A(batch_x)
                y = self.model_y(batch_x)
                w = self.model_w(batch_x)
                
                dist = torch.distributions.Bernoulli(w)
                sam = dist.sample()
                
                optimizer_2.zero_grad()
                loss2 = loss_ML_sam(A, batch_A, sam)
                loss2.backward(retain_graph = True)
                optimizer_2.step()
    
                optimizer_1.zero_grad()        
                loss1 = loss_ML_sam(y, batch_y, sam)
                loss1.backward(retain_graph = True)
                optimizer_1.step()
                
                optimizer_3.zero_grad()
                loss3 = loss_w(A, y, batch_A, batch_y, alpha, beta, w, sam)
                loss3.backward()
                optimizer_3.step()
                
            if e%log_epoch == 0 and log == 1:
                y = self.model_y(x_train)
                A = self.model_A(x_train)
                print(nll_criterion(A, A_train).data, nll_criterion(y, y_train).data)
                print(min(w.data),max(w.data),torch.mean(w).data,torch.sum(w).data)



    def predict(self, x_test):
        self.model_y.eval()
        self.model_A.eval()
        y = self.model_y(x_test)
        A = self.model_A(x_test)
        y = np.round(y.data)
        A = np.round(A.data)
        return y, A
    
    def predict_proba(self, x_test):
        self.model_y.eval()
        self.model_A.eval()
        y = self.model_y(x_test)
        A = self.model_A(x_test)
        y = y.data
        A = A.data
        return y, A



