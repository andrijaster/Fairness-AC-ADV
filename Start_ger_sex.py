# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 11:31:36 2019

@author: Andri
"""

import torch
import os
import xlwt 
import numpy as np
import models

from sklearn.preprocessing import StandardScaler
from aif360.metrics import ClassificationMetric
from aif360.datasets import BinaryLabelDataset
from aif360.datasets import GermanDataset
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from xlwt import Workbook 


def german_dataset(name_prot=['sex']):
    dataset_orig = GermanDataset(
        protected_attribute_names = name_prot,           # this dataset also contains protected
                                                     # attribute for "sex" which we do not
                                                     # consider in this evaluation      # age >=25 is considered privileged
        features_to_drop=['personal_status', 'age'] # ignore sex-related attributes
    )
    
    privileged_groups = [{'sex': 1}]
    unprivileged_groups = [{'sex': 0}]
    
    data, _ = dataset_orig.convert_to_dataframe()
    data.rename(columns={'credit':'labels'}, inplace = True)
    sensitive = data[name_prot]
    output = data['labels']
    output.replace((1,2),(0,1),inplace = True)
    atribute = data.drop('labels', axis = 1, inplace = False)
    atribute.drop(name_prot, axis = 1, inplace = True)    
    return data, atribute, sensitive, output, privileged_groups, unprivileged_groups



def test(dataset, model, x_test, thresh_arr, unprivileged_groups, privileged_groups):

    bld = BinaryLabelDataset(df = dataset, label_names = ['labels'], 
                             protected_attribute_names=['sex'])
   
    y_val_pred_prob, A_val_pred_prob = model.predict_proba(x_test)
    
    metric_arrs = np.empty([0,8])
    for thresh in thresh_arr:
        y_val_pred = (y_val_pred_prob.numpy() > thresh).astype(np.float64)
        A_val_pred = (A_val_pred_prob.numpy() > thresh).astype(np.float64)
        
        metric_arrs = np.append(metric_arrs, roc_auc_score(y_test, y_val_pred_prob))
        metric_arrs = np.append(metric_arrs, roc_auc_score(A_test, A_val_pred_prob))

        dataset_pred = dataset.copy()
        dataset_pred.labels = y_val_pred
        bld2 = BinaryLabelDataset(df = dataset_pred, label_names = ['labels'], 
                             protected_attribute_names=['sex'])
        
        metric = ClassificationMetric(
                bld, bld2,
                unprivileged_groups = unprivileged_groups,
                privileged_groups = privileged_groups)

        metric_arrs = np.append(metric_arrs, ((metric.true_positive_rate()
                                     + metric.true_negative_rate()) / 2))
        metric_arrs = np.append(metric_arrs, metric.average_odds_difference())
        metric_arrs = np.append(metric_arrs, metric.disparate_impact())
        metric_arrs = np.append(metric_arrs, metric.statistical_parity_difference())
        metric_arrs = np.append(metric_arrs, metric.equal_opportunity_difference())
        metric_arrs = np.append(metric_arrs, metric.theil_index())
    
    return metric_arrs


saver_dir_res = 'Results'
file_name = os.path.join(saver_dir_res, 'Results_german_sex.xls')
model_no = 7
metrics = np.zeros([model_no,8])
lst = []

if not os.path.exists(saver_dir_res):
    os.mkdir(saver_dir_res)


wb = Workbook()
sheet1 = wb.add_sheet('Results') 

for i in range(model_no):
    sheet1.write(i+1,0,'model_{}'.format(i))


columns = ["AUC_y", "AUC_A", 'bal_acc', 'avg_odds_diff', 
           'disp_imp','stat_par_diff', 'eq_opp_diff', 'theil_ind']

k = 1
for i in columns:
    sheet1.write(0,k,i)
    k+=1

data, atribute, sensitive, output, pr_gr, un_gr = german_dataset()
skf = KFold(n_splits = 10)
skf.get_n_splits(atribute, output)

std_scl = 1
AUC_y = np.zeros(model_no)
AUC_A = np.zeros(model_no)
#threshold = np.linspace(0.01, 1, 100)
threshold = [0.5]
inp = 56
epochs = 400

for train_index,test_index in skf.split(atribute, output):
    
 
    lst = [models.Adversarial_prob_class(input_size = inp),
        models.Adversarial_class(input_size = inp),
        models.Adversarial_weight_class(input_size = inp),
        models.Adversarial_weight_soft_class(input_size = inp),
        models.Adversarial_weight_hard_class(input_size = inp),
        models.Adversarial_weight_soft_r_class(input_size = inp),
        models.Adversarial_prob_nf_class(flow_length = 3, no_sample = 50,
                                         input_size = inp)]
    
    x_train, x_test = atribute.iloc[train_index,:], atribute.iloc[test_index,:]
    y_train, y_test = output.iloc[train_index], output.iloc[test_index] 
    A_train, A_test = sensitive.iloc[train_index], sensitive.iloc[test_index]  
    data_train, data_test = data.iloc[train_index,:], data.iloc[test_index,:]
    
    if std_scl == 1:
        std_scl = StandardScaler()
        std_scl.fit(x_train)
        x_train_x = std_scl.transform(x_train)
        x_test_x = std_scl.transform(x_test)
    else:
        x_train_x = x_train.values
        x_test_x = x_test.values
            
        
    x_train_t = torch.tensor(x_train_x).type('torch.FloatTensor')
    x_test_t = torch.tensor(x_test_x).type('torch.FloatTensor')
    y_train_t = torch.tensor(y_train.values).type('torch.FloatTensor').reshape(-1,1)
    y_test_t = torch.tensor(y_test.values).type('torch.FloatTensor').reshape(-1,1)
    A_train_t = torch.tensor(A_train.values).type('torch.FloatTensor').reshape(-1,1)
    A_test_t = torch.tensor(A_test.values).type('torch.FloatTensor').reshape(-1,1)
    
    k = 0
    for i in lst: 
        if k == 0:
            i.fit(x_train_t, y_train_t, A_train_t, max_epoch= epochs, log = 0, no_sample = 50)   
        else:
            i.fit(x_train_t, y_train_t, A_train_t, max_epoch= epochs, log = 0)

        metrics[k,:] += test(data_test, i, x_test_t, threshold, un_gr, pr_gr)
        k+=1

metrics = np.round(metrics/10,4)
for row in range(model_no):    
    for column,_ in enumerate(columns):  
        sheet1.write(row+1, column+1 , metrics[row,column])

wb.save(file_name) 

