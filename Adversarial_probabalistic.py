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




class FairClass(nn.Module):
    def __init__(self):
        super(FairClass, self).__init__()
        self.fc1 = nn.Linear(56,20)
        self.bn1 = nn.BatchNorm1d(num_features=20)
        self.fc21 = nn.Linear(20,10)
        self.fc22 = nn.Linear(20,10)
        self.fc31 = nn.Linear(10,5)
        self.fc32 = nn.Linear(5,1)
        self.fc41 = nn.Linear(10,5)
        self.fc42 = nn.Linear(5,1)
        
    def P_zx(self, x):
        h1 = F.relu(self.bn1(self.fc1(x)))
        return self.fc21(h1), self.fc22(h1)
    
    def reparameterize(self, mu, logvar, no_samples):
        std = torch.exp(0.5*logvar)
        number = list(std.size())
        number.insert(0,no_samples)
        eps = torch.randn(number)
        return mu + eps*std

    def P_yz(self, z):
        h3 = F.relu(self.fc31(z))
        y = torch.sigmoid(self.fc32(h3))
        return torch.mean(y, dim=0)
    
    def P_Az(self, z):
        h4 = F.relu(self.fc41(z))
        A = torch.sigmoid(self.fc42(h4))
        return torch.mean(A, dim=0)
        

    def forward(self, x, no_samples = 100):
        mu, logvar = self.P_zx(x)
        z = self.reparameterize(mu, logvar, no_samples)
        output_y = self.P_yz(z)
        output_A = self.P_Az(z)
        return output_y, output_A
    


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

def evaluate(model, x_test, y_test, A_test, no_samples = 100):
    model.eval()
    y_calc, A_calc = model(x_test, no_samples)
    ACC_1 = accuracy_score(y_test, np.round(y_calc.data))
    ACC_2 = accuracy_score(A_test, np.round(A_calc.data))
    return ACC_1, ACC_2

def training(model, x_train, y_train, A_train, max_epoch = 1000, mini_batch_size = 50, alpha = 1):
    model.train()
    nll_criterion =F.binary_cross_entropy
    list_z = list(model.fc1.parameters())+list(model.bn1.parameters())+list(model.fc21.parameters())+list(model.fc22.parameters())
    list_1 = list(model.fc31.parameters())+list(model.fc32.parameters())+list_z
    list_2 = list(model.fc41.parameters())+list(model.fc42.parameters())
    optimizer_1 = torch.optim.Adam(list_1, lr = 0.0005)
    optimizer_2 = torch.optim.Adam(list_2, lr = 0.0005)
    for e in range(max_epoch):
        for i in range(0,x_train.size()[0], mini_batch_size):     
            batch_x, batch_y, batch_A = (x_train[i:i+mini_batch_size], y_train[i:i+mini_batch_size], 
                                        A_train[i:i+mini_batch_size])
            output_1, output_2 = model(batch_x, no_samples = 100)
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
    y_train = torch.tensor(y_train.values).type('torch.FloatTensor')
    y_test = torch.tensor(y_test.values).type('torch.FloatTensor')
    A_train = torch.tensor(A_train.values).type('torch.FloatTensor')
    A_test = torch.tensor(A_test.values).type('torch.FloatTensor') 
    
    model = training(model, x_train, y_train, A_train)
    print(evaluate(model,x_test, y_test, A_test))

