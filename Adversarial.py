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




class FairClass(nn.Module):
    def __init__(self):
        super(FairClass, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(56,20),
                        nn.BatchNorm1d(num_features=20),
                        nn.ReLU())
        self.fc2 = nn.Linear(20,1)
        self.fc3 = nn.Linear(20,1)
        
    def forward(self, x):
        z = self.fc1(x)
        y = F.sigmoid(self.fc2(z))
        A = F.sigmoid(self.fc3(z))
        return y, A


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

def evaluate(model, x_test, y_test, A_test):
    model.eval()
    y_calc, A_calc = model(x_test)
    ACC_1 = accuracy_score(y_test, np.round(y_calc.data))
    ACC_2 = accuracy_score(A_test, np.round(A_calc.data))
    return ACC_1, ACC_2

def training(model, x_train, y_train, A_train, max_epoch = 300, mini_batch_size = 50, alpha = 1):
    model.train()
    nll_criterion =F.binary_cross_entropy
    list_1 = list(model.fc1.parameters())+list(model.fc2.parameters())
    list_2 = list(model.fc3.parameters())
    optimizer_1 = torch.optim.Adam(list_1, lr = 0.001)
    optimizer_2 = torch.optim.Adam(list_2, lr = 0.001)
    for e in range(max_epoch):
        for i in range(0,x_train.size()[0], mini_batch_size):     
            batch_x, batch_y, batch_A = (x_train[i:i+mini_batch_size], y_train[i:i+mini_batch_size], 
                                        A_train[i:i+mini_batch_size])
            output_1, output_2 = model(batch_x)
            loss2 = nll_criterion(output_2, batch_A)
            optimizer_2.zero_grad()
            loss2.backward(retain_graph=True)
            optimizer_2.step()
            loss1 = nll_criterion(output_1, batch_y) - alpha * nll_criterion(output_2,batch_A)
            optimizer_1.zero_grad()
            loss1.backward()
            optimizer_1.step()
        if e%10 == 0:
            out_1, out_2 = model(x_train)
            print(nll_criterion(out_2, A_train).data, nll_criterion(out_1, y_train).data)

                         
    return model


skf = KFold(n_splits = 10)
skf.get_n_splits(atribute, output)
model = FairClass()



for train_index,test_index in skf.split(atribute, output):
    x_train, x_test = atribute.iloc[train_index,:], atribute.iloc[test_index,:]
    y_train, y_test = output.iloc[train_index], output.iloc[test_index] 
    A_train, A_test = sensitive.iloc[train_index], sensitive.iloc[test_index]  
    std_scl = StandardScaler()
    std_scl.fit(x_train)
    x_train = std_scl.transform(x_train)
    x_test = std_scl.transform(x_test)
    
    
    x_train = torch.tensor(x_train).type('torch.FloatTensor')
    x_test = torch.tensor(x_test).type('torch.FloatTensor')
    y_train = torch.tensor(y_train.values).type('torch.FloatTensor').reshape(-1,1)
    y_test = torch.tensor(y_test.values).type('torch.FloatTensor').reshape(-1,1)
    A_train = torch.tensor(A_train.values).type('torch.FloatTensor').reshape(-1,1)
    A_test = torch.tensor(A_test.values).type('torch.FloatTensor').reshape(-1,1)
    
    model = training(model, x_train, y_train, A_train)
    print(evaluate(model,x_test, y_test, A_test))



        
    
    
    
    