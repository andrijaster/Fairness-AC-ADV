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
from torch import nn
from torch.nn import functional as F





class Adversarial_prob_nf_class():
    
    
    class Flow_transform(nn.Module): 
        def __init__(self, no_sample = 1):    
            super(Adversarial_prob_nf_class.Flow_transform, self).__init__()
            self.block = nn.Sequential(nn.Linear(10,10), nn.Tanh())
        
        def forward(self, x):
            y = self.block(x)
            return y
    
    class Affine_transform(nn.Module):    
        def __init__(self, no_sample):
            super(Adversarial_prob_nf_class.Affine_transform, self).__init__()
            self.fc1 = nn.Sequential(nn.Linear(56,20),
                    nn.BatchNorm1d(num_features=20),
                    nn.ReLU())
            self.fc21 = nn.Linear(20,10)
            self.fc22 = nn.Linear(20,10)
            self.no_samples = no_sample

            
        def P_zx(self, x):
            h1 = self.fc1(x)
            return self.fc21(h1), self.fc22(h1)  
        
        def reparameterize(self, mu, logvar, no_samples):
            std = torch.exp(0.5*logvar)
            number = list(std.size())
            number.insert(0,no_samples)
            eps = torch.randn(number)
            return mu + eps*std
        
        def forward(self, x):
             mu, logvar = self.P_zx(x)
             z = self.reparameterize(mu, logvar, self.no_samples)
             return z
             
  
    class Y_output(nn.Module):      
        def __init__(self):
            super(Adversarial_prob_nf_class.Y_output, self).__init__()            
            self.fc31 = nn.Sequential(nn.Linear(10,5),
                              nn.ReLU(),
                              nn.Linear(5,1))
        
        def forward(self, z):
            y = torch.sigmoid(self.fc31(z))
            return torch.mean(y, dim=0)   
       
        
 
    class A_output(nn.Module):
        def __init__(self):
            super(Adversarial_prob_nf_class.A_output, self).__init__()            
            self.fc41 = nn.Sequential(nn.Linear(10,5),
                              nn.ReLU(),
                              nn.Linear(5,1))

        def forward(self, z):
            A = torch.sigmoid(self.fc41(z))
            return torch.mean(A, dim=0)
                        
    
    def __init__(self, flow_length, no_sample):
        self.model_y = Adversarial_prob_nf_class.Y_output()
        self.model_A = Adversarial_prob_nf_class.A_output()
        self.Transform = nn.Sequential(Adversarial_prob_nf_class.Affine_transform(no_sample), 
                                       *[Adversarial_prob_nf_class.Flow_transform() 
                                       for _ in range(flow_length)])


    def fit(self, x_train, y_train, A_train, max_epoch = 100, mini_batch_size = 50, alpha = 1,
            log_epoch = 1, log = 1):
        
        self.model_y.train()
        self.model_A.train()
        self.Transform.train()
               
        nll_criterion = F.binary_cross_entropy
        
        list_1 = list(self.model_y.parameters())+list(self.Transform.parameters())
        list_2 = list(self.model_A.parameters())
        
        optimizer_1 = torch.optim.Adam(list_1, lr = 0.001)
        optimizer_2 = torch.optim.Adam(list_2, lr = 0.001)
        
        for e in range(max_epoch):
            for i in range(0,x_train.size()[0], mini_batch_size):     
               
                batch_x, batch_y, batch_A = (x_train[i:i+mini_batch_size], y_train[i:i+mini_batch_size], 
                                            A_train[i:i+mini_batch_size])
                
                z = self.Transform(batch_x)
                output_1 = self.model_y(z)
                output_2 = self.model_A(z)
                                
                loss2 = nll_criterion(output_2, batch_A)
                optimizer_2.zero_grad()
                loss2.backward(retain_graph=True)
                optimizer_2.step()
                
                loss1 = nll_criterion(output_1, batch_y) - alpha * nll_criterion(output_2,batch_A)
                optimizer_1.zero_grad()
                loss1.backward()
                optimizer_1.step()
                
            if e%log_epoch == 0 and  log == 1:
                z = self.Transform(x_train)
                out_1 = self.model_y(z)
                out_2 = self.model_A(z)
                print(nll_criterion(out_2, A_train).data, nll_criterion(out_1, y_train).data)

   
    def predict(self, x_test, no_sample = 100):
        self.model_y.eval()
        self.model_A.eval()
        self.Transform.eval()
        z = self.Transform(x_test, no_sample)
        y = self.model_y(z)
        A = self.model_A(z)
        y = np.round(y.data)
        A = np.round(A.data)
        return y, A
    
    def predict_proba(self, x_test, no_sample = 100):
        self.model_y.eval()
        self.model_A.eval()
        self.Transform.eval()
        z = self.Transform(x_test, no_sample)
        y = self.model_y(z)
        A = self.model_A(z)
        y = y.data
        A = A.data
        return y, A
    

