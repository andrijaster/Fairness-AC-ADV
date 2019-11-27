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


class Fair_classifier(nn.Module):
    def __init__(self):
        super(Fair_classifier, self).__init__()
        
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

    def forward(self, x):
        output_y = torch.sigmoid(self.fc1(x))
        output_A = torch.sigmoid(self.fc2(x))
        output_w = torch.sigmoid(self.fc3(x))
        return output_y, output_A, output_w


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


def loss(output, target, weights):
    output = torch.clamp(output, 1e-5, 1 - 1e-5)
    weights = torch.clamp(weights, 1e-5, 1 - 1e-5)
    ML =  weights*(target*torch.log(output) + (1-target)*torch.log(1-output))
    return torch.neg(torch.mean(ML))


def training(model, x_train, y_train, A_train, max_epoch = 300, mini_batch_size = 50, alpha = 1.1, beta = 0):
    
    model.train()
    nll_criterion =F.binary_cross_entropy
    list_0 = list(model.fc1.parameters())
    list_1 = list(model.fc3.parameters())
    list_2 = list(model.fc2.parameters())
    
    optimizer_0 = torch.optim.Adam(list_0, lr = 0.0001)
    optimizer_1 = torch.optim.Adam(list_1, lr = 0.0001)
    optimizer_2 = torch.optim.Adam(list_2, lr = 0.0001)
    
#    scheduler_1 = StepLR(optimizer_1, step_size=1, gamma=1)
#    scheduler_2 = StepLR(optimizer_1, step_size=1, gamma=1)

    for e in range(max_epoch):
        for i in range(0,x_train.size()[0], mini_batch_size):     
            batch_x, batch_y, batch_A = (x_train[i:i+mini_batch_size], y_train[i:i+mini_batch_size], 
                                        A_train[i:i+mini_batch_size])
            y, A, w = model(batch_x)
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

            
        if e%10 == 0:
            y, A, w = model(x_train)
            print(min(w.data),max(w.data),torch.mean(w).data,torch.sum(w).data)
            print(nll_criterion(A, A_train).data, nll_criterion(y, y_train).data)
#            print(loss_2(A, A_train, w).data, (loss_1(y, y_train, w) - alpha*loss_2(A, A_train, w)).data)
#        scheduler_1.step()
#        scheduler_2.step()
    return model


def evaluate(model, x_test, y_test):
    model.eval()
    y,_,_= model(x_test)
    ACC_1 = accuracy_score(y_test, np.round(y.data))
    return ACC_1



atribute, sensitive, output = german_dataset()


skf = KFold(n_splits = 10)
skf.get_n_splits(atribute, output)



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
    
    model = Fair_classifier()
    model = training(model, x_train, y_train, A_train)
    print(evaluate(model, x_test, y_test))
