# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 11:31:36 2019

@author: Andri
"""

import torch

from sklearn.preprocessing import StandardScaler
from aif360.datasets import GermanDataset
from sklearn.model_selection import KFold
import models



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

for train_index,test_index in skf.split(atribute, output):
    
    model_ADV = models.Adversarial_class()
    model_ADV_prob = models.Adversarial_prob_class()
    model_weight_class = models.Adversarial_weight_class()
    model_weight_beta = models.Adversarial_weight_soft_class()
    model_weight_hard = models.Adversarial_weight_hard_class()
    model_weight_beta_reparam = models.Adversarial_weight_soft_r_class()
    
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
    
    model_weight_beta_reparam.fit(x_train, y_train, A_train)
    model_weight_hard.fit(x_train, y_train, A_train)
    model_weight_beta.fit(x_train, y_train, A_train)
    model_weight_class.fit(x_train, y_train, A_train)
    model_ADV_prob.fit(x_train, y_train, A_train, max_epoch = 100, no_sample = 100)
    model_ADV.fit(x_train, y_train, A_train, max_epoch = 100)
    print(model_weight_class.predict(x_test))
    print(model_ADV.predict(x_test))
    print(model_ADV.predict_proba(x_test))
    
