# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 10:32:12 2019

@author: Andri
"""

import torch
import torch.utils.data
import numpy as np
from torch.nn import functional as F
from torch import nn, optim
from sklearn.model_selection import KFold
from aif360.datasets import GermanDataset
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


class Output_class(nn.Module):
    def __init__(self):
        super(Output_class, self).__init__()
        
        self.fc1 = nn.Sequential(nn.Linear(56,20),
        nn.BatchNorm1d(num_features=20),
        nn.ReLU(),
        nn.Linear(20,1))        

    def forward(self, x):
        output_y = torch.sigmoid(self.fc1(x))
        return output_y
    
    
class Atribute_class(nn.Module):
    def __init__(self):
        super(Atribute_class, self).__init__()
        
        self.fc2 = nn.Sequential(nn.Linear(56,20),
        nn.BatchNorm1d(num_features=20),
        nn.ReLU(),
        nn.Linear(20,1))        

    def forward(self, x):
        u = self.fc2(x)
        output_A = torch.sigmoid(u)
        return output_A    
    
class weight_class(nn.Module):
    def __init__(self):
        super(weight_class, self).__init__()      
        self.fc3 = nn.Sequential(nn.Linear(56,20),
        nn.BatchNorm1d(num_features=20),
        nn.ReLU(),
        nn.Linear(20,1))       
        
    def forward(self, x):
        output_w = torch.sigmoid(self.fc3(x))
        return output_w    
    




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


def loss_ML_sam(output, target, sam):
    output = torch.clamp(output, 1e-5, 1 - 1e-5)
    ML =  sam*(target*torch.log(output) + (1-target)*torch.log(1-output))
    return torch.neg(torch.sum(ML))

def loss_w(output_A, output_Y, target_A, target_Y, alpha, beta, weights, sam):
    weights = torch.clamp(weights, 1e-5, 1 - 1e-5)
    ML = torch.log(weights)*(loss_ML_sam(output_Y, target_Y, sam) - alpha*loss_ML_sam(output_A, target_A, sam) - beta * torch.mean(sam))
    return torch.neg(torch.sum(ML))


def training(model_y, model_A, model_w, x_train, y_train, A_train, max_epoch = 200, mini_batch_size = 50, alpha = 1, beta = 1):
    
    model_y.train()
    model_A.train()
    model_w.train()
    nll_criterion =F.binary_cross_entropy
    list_1 = list(model_y.parameters()) 
    list_2 = list(model_A.fc2.parameters())
    list_3 = list(model_w.parameters())
    
    optimizer_1 = torch.optim.Adam(list_1, lr = 0.0001)
    optimizer_2 = torch.optim.Adam(list_2, lr = 0.0001)
    optimizer_3 = torch.optim.Adam(list_3, lr = 0.0001)
    
    for e in range(max_epoch):
        for i in range(0,x_train.size()[0], mini_batch_size):     
            batch_x, batch_y, batch_A = (x_train[i:i+mini_batch_size], y_train[i:i+mini_batch_size], 
                                        A_train[i:i+mini_batch_size])
            A = model_A(batch_x)
            y = model_y(batch_x)
            w = model_w(batch_x)
            
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
            
#            for p in model_w.parameters(): print(p.data)
#            print(w)
            
        if e%1 == 0:
            y = model_y(x_train)
            A = model_A(x_train)
            print(nll_criterion(A, A_train).data, nll_criterion(y, y_train).data)
            print(min(w.data),max(w.data),torch.mean(w).data,torch.sum(w).data)


                         
    return model_y


def evaluate(model_y, x_test, y_test):
    model_y.eval()
    y_calc = model_y(x_test)
    ACC_1 = accuracy_score(y_test, np.round(y_calc.data))
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
    
    model_y = Output_class()
    model_A = Atribute_class()
    model_w = weight_class()
    
    model_y = training(model_y, model_A, model_w, x_train, y_train, A_train)
    print(evaluate(model_y, x_test, y_test))