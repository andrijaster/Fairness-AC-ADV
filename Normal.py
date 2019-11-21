# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 19:00:06 2019

@author: Andri
"""

import torch
import torch.utils.data
import pandas as pd
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from sklearn.model_selection import KFold
from aif360.datasets import GermanDataset
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


class NetNLL(nn.Module):
    def __init__(self):
        super(NetNLL, self).__init__()
        self.fc1 = nn.Linear(56,1)
        self.bn1 = nn.BatchNorm1d(num_features=1)

        
    def forward(self, x):
        x = self.bn1(self.fc1(x))
        x = F.sigmoid(x)
        return x

def german_dataset(name_prot=['age']):
    dataset_orig = GermanDataset(
        protected_attribute_names = name_prot,           # this dataset also contains protected
                                                     # attribute for "sex" which we do not
                                                     # consider in this evaluation
        privileged_classes=[lambda x: x >= 25],      # age >=25 is considered privileged
        features_to_drop=['personal_status', 'sex'] # ignore sex-related attributes
    )
    
    data, _ = dataset_orig.convert_to_dataframe()
    sensitive = data[name_prot]
    output = data['credit']
    output.replace((1,2),(0,1),inplace = True)
    data.drop('credit', axis = 1, inplace = True)
    data.drop(name_prot, axis = 1, inplace = True)    
    return data, sensitive, output


atribute, sensitive, output = german_dataset()

skf = KFold(n_splits = 10)
skf.get_n_splits(atribute, output)
nll_criterion = torch.nn.BCELoss()

for train_index,test_index in skf.split(atribute, output):
    model = NetNLL()
    x_train, x_test = atribute.iloc[train_index,:], atribute.iloc[test_index,:]
    y_train, y_test = output.iloc[train_index], output.iloc[test_index] 
#    y_test = y_test.values.reshape(-1,1)
#    y_train = y_train.values.reshape(-1,1)
    x_train = torch.tensor(x_train.values).type('torch.FloatTensor')
    x_test = torch.tensor(x_test.values).type('torch.FloatTensor')
    y_train = torch.tensor(y_train.values).type('torch.FloatTensor')
    y_test = torch.tensor(y_test.values).type('torch.FloatTensor')

    model.train()
    mini_batch_size=36
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01 )
    for e in range(100):
        for i in range(0,x_train.size()[0], mini_batch_size):     
            batch_x, batch_y = x_train[i:i+mini_batch_size], y_train[i:i+mini_batch_size]
            output_1 = model(batch_x)
            loss = nll_criterion(output_1, batch_y)
#            print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if e%10 == 0:
                print(loss.item())