# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 13:00:12 2019

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

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Sequential(nn.Linear(56, 36),
                                 nn.BatchNorm1d(36),
                                 nn.ReLU())
                                 
        self.fc21 = nn.Linear(20, 10)
        self.fc22 = nn.Linear(20, 10)
        
        self.fc3 =  nn.Sequential(nn.Linear(20, 36),
                                  nn.BatchNorm1d(36),
                                  nn.Relu())
        
        self.fc4 = nn.Linear(36,3)
        self.fc5 = nn.Linear(36,51)
        self.fc3 = nn.Linear(36,2)
        
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = self.fc3(z)
        return torch.sigmoid(self.fc5(h3)), torch.softmax(self.fc4(h3)), self.fc3(h3)
    
    def predict_A(self,z):
        return torch.sigmoid(self.fc5)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    
def german_dataset(name_prot=['age']):
    dataset_orig = GermanDataset(
        protected_attribute_names = name_prot,       # this dataset also contains protected
                                                     # attribute for "sex" which we do not
                                                     # consider in this evaluation
        privileged_classes=[lambda x: x >= 25],      # age >=25 is considered privileged
        features_to_drop=['personal_status', 'sex']  # ignore sex-related attributes
    )
    
    data, _ = dataset_orig.convert_to_dataframe()
    sensitive = data[name_prot]
    output = data['credit']
    output.replace((1,2),(0,1),inplace = True)
    data.drop('credit', axis = 1, inplace = True)
    data.drop(name_prot, axis = 1, inplace = True)    
    return data, sensitive, output

def loss_function(recon_x_cat,recon_x_bin,recon_x_cont, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
#    MSE = 
    
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

atribute, sensitive, output = german_dataset()
atribute = atribute.values
sensitive = sensitive.values
output = output.values
atribute[:,5] = np.where(atribute[:,5]==1,0,atribute[:,5])
atribute[:,5] = np.where(atribute[:,5]==2,1,atribute[:,5])
x_num = atribute[:,0:2]
x_cat = atribute[:,2:5]
x_bin = atribute[:,5:]

