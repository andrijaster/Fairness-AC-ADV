import torch
import os
import numpy as np
import pandas as pd
import models_H
import models
import pickle

from sklearn.preprocessing import StandardScaler
from aif360.metrics import ClassificationMetric
from aif360.datasets import BinaryLabelDataset
from aif360.datasets import AdultDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xlwt import Workbook 


def medical_dataset(name_prot = 'sex'):
    
    dataset_orig = AdultDataset(protected_attribute_names=['sex'],
            privileged_classes= [['Male']],
            features_to_keep=['age', 'education-num'])
    
    privileged_groups = [{'sex': 1}]
    unprivileged_groups = [{'sex': 0}]
    
    data, _ = dataset_orig.convert_to_dataframe()
    data.rename(columns={'income-per-year':'labels'}, inplace = True)
    data.reset_index(inplace = True, drop = True)
    sensitive = data[name_prot]
    output = dataset_orig.labels
    atribute = data.drop('labels', axis = 1, inplace = False)
    atribute.drop(name_prot, axis = 1, inplace = True)  
    return data, atribute, sensitive, output, privileged_groups, unprivileged_groups



def test(dataset, model, x_test, thresh_arr, unprivileged_groups, privileged_groups):

    bld = BinaryLabelDataset(df = dataset, label_names = ['labels'], 
                             protected_attribute_names=['sex'])
   
    if np.isin(k ,model_AIF):
        y_val_pred_prob = model.predict_proba(bld)
    else:
        y_val_pred_prob, A_val_pred_prob = model.predict_proba(x_test)
        
    metric_arrs = np.empty([0,8])
    for thresh in thresh_arr:
        if np.isin(k ,model_AIF):
            y_val_pred = (y_val_pred_prob > thresh).astype(np.float64)
        else:
            y_val_pred = (y_val_pred_prob.numpy() > thresh).astype(np.float64)
            
        metric_arrs = np.append(metric_arrs, roc_auc_score(y_test, y_val_pred_prob))
        
        if np.isin(k ,model_AIF):
            metric_arrs = np.append(metric_arrs, 0)
        else:
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


""" INPUT DATA """
model_no = 7
epochs = 1
threshold = [0.5]
model_AIF = [0]
std_scl = 1
alpha = np.linspace(2.42, 3, 2)
""" """

saver_dir_res = 'Results'
file_name = os.path.join(saver_dir_res, 'Results_Adult_epoch_H_{}_model_no_{}.xls'.format(epochs, model_no))

if not os.path.exists(saver_dir_res):
    os.mkdir(saver_dir_res)
    
saver_dir_models = 'Trained_models/Adult_H'    
if not os.path.exists(saver_dir_models):
    os.mkdir(saver_dir_models)
    
data, atribute, sensitive, output, pr_gr, un_gr = medical_dataset()

inp = atribute.shape[1]

wb = Workbook()


columns = ["AUC_y", "AUC_A", 'bal_acc', 'avg_odds_diff', 
           'disp_imp','stat_par_diff', 'eq_opp_diff', 'theil_ind']


alpha = np.linspace(1.259925, 3, 4)

sheets = [wb.add_sheet('{}'.format(i)) for i in alpha]

ind = 0
for a in alpha:
    
    metrics = np.zeros([model_no,8])
    k = 1
    for i in columns:
        sheets[ind].write(0,k,i)
        k+=1   
        
    for mod in range(model_no):
        sheets[ind].write(mod+1,0,'model_{}'.format(mod))

        
        lst = [models.Fair_rew_RF(un_gr, pr_gr),
            models_H.FAD_H_class(input_size = inp, num_layers_z = 3, num_layers_y = 3, 
                                      step_z = 1.5, step_y = 1.5),
            models_H.FAIR_scalar_H_class(input_size = inp, num_layers_w = 3, step_w = 1.5, 
                     num_layers_A = 2, step_A = 1.5, num_layers_y = 4, step_y = 1.5),
            models_H.FAIR_betaSF_H_class(input_size = inp, num_layers_w = 3, step_w = 1.5, 
                     num_layers_A = 2, step_A = 1.5, num_layers_y = 4, step_y = 1.5),
            models_H.FAIR_Bernoulli_H_class(input_size = inp, num_layers_w = 3, step_w = 1.5, 
                     num_layers_A = 2, step_A = 1.5, num_layers_y = 4, step_y = 1.5),
            models_H.FAIR_betaREP_H_class(input_size = inp, num_layers_w = 3, step_w = 1.5, 
                     num_layers_A = 2, step_A = 1.5, num_layers_y = 4, step_y = 1.5),
            models_H.FAD_prob_H_class(flow_length = 2, no_sample = 1,
                                             input_size = inp, num_layers_y = 2, 
                                             step_y = 2, step_z = 2)]
    
    x_train, x_test, y_train, y_test, A_train, A_test, data_train, data_test = train_test_split(
                atribute, output, sensitive, data, test_size=0.2, random_state=42)
   
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
    y_train_t = torch.tensor(y_train).type('torch.FloatTensor').reshape(-1,1)
    y_test_t = torch.tensor(y_test).type('torch.FloatTensor').reshape(-1,1)
    A_train_t = torch.tensor(A_train.values).type('torch.FloatTensor').reshape(-1,1)
    A_test_t = torch.tensor(A_test.values).type('torch.FloatTensor').reshape(-1,1)

    k = 0
    for i in lst: 
        if np.isin(k ,model_AIF):
            i.fit(data_train, ['labels'], ['sex'])
        else:
            i.fit(x_train_t, y_train_t, A_train_t, max_epoch= epochs, log = 0, alpha = a)
        saver_path = os.path.join(saver_dir_models, 'checkpoint_{}_epochs_{}_alpha_{}'.format(type(i).__name__, epochs, a))
        f = open(saver_path,"wb")
        pickle.dump(i,f)
        f.close
        metrics[k,:] += test(data_test, i, x_test_t, threshold, un_gr, pr_gr)
        k+=1
        
    for row in range(model_no):    
        for column,_ in enumerate(columns):  
            sheets[ind].write(row+1, column+1 , metrics[row,column])
    
    wb.save(file_name) 
    ind += 1
    

