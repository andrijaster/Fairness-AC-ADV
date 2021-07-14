import torch
import os
import numpy as np
import pandas as pd
import models
import pickle 


from utilities import Adult_dataset
from utilities import medical_dataset
from utilities import german_dataset_age
from utilities import german_dataset_sex
from utilities import Readmission_dataset

from utilities import format_datasets
from utilities import Dataset_format
from utilities import test


from torch.utils.data import Dataset, DataLoader


from sklearn.preprocessing import StandardScaler
from aif360.metrics import ClassificationMetric
from aif360.datasets import BinaryLabelDataset
from sklearn.metrics import roc_auc_score
from xlwt import Workbook 

if __name__ == "__main__":


    """ INPUT DATA """
    epochs = 3000
    threshold = 0.5
    model_AIF = [0, 1, 2, 3 ,4]
    alpha = [0, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
    eta = [0, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]

    saver_dir_models = r'Trained_models/MEPS19'    
    saver_dir_results= r'Results/MEPS19.xls' 


    data, atribute, sensitive, output, pr_gr, un_gr = medical_dataset()
    prot = list(pr_gr[0].keys())[0]
    
    data_train, data_test, data_val, atribute_train, atribute_val, atribute_test, sensitive_train, sensitive_val, sensitive_test, output_train, output_val, output_test = format_datasets(data, atribute, sensitive, output, sens_name=prot)

    dataset_train = Dataset_format(atribute_train, sensitive_train, output_train)
    dataset_val = Dataset_format(atribute_val, sensitive_val, output_val)
    dataset_test = Dataset_format(atribute_test, sensitive_test, output_test)

    dataloader_train = DataLoader(dataset_train, batch_size=len(dataset_train), shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=len(dataset_val), shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=True)

    inp = dataset_train[0][0].shape[0]

    iteracija = 0

    wb = Workbook()

    columns = ["model","AUC_y_val", "Accuracy_y_val", "AUC_A_val", 'bal_acc_val', 'avg_odds_diff_val', 
               'disp_imp_val','stat_par_diff_val', 'eq_opp_diff_val', 
               "AUC_y_test", "Accuracy_y_test", "AUC_A_test", 'bal_acc_test', 'avg_odds_diff_test', 
               'disp_imp_test','stat_par_diff_test', 'eq_opp_diff_test', "alpha"]

    sheets = wb.add_sheet('{}'.format("sheet_1"))

    k = 0
    for i in columns:
        sheets.write(0,k,i)
        k+=1   

    row = 1

    for a in alpha:

        ind = eta[iteracija]
        iteracija +=1

        lst = [
            # models.Fair_PR(sensitive = prot, class_attr = 'labels', eta = ind),
            # models.Fair_DI_NN(sensitive = prot, inp_size = inp, num_layers_y = 3, step_y = 1.5, repair_level = ind),
            # models.Fair_DI_RF(sensitive = prot, repair_level = ind),
            # models.Fair_rew_NN(un_gr, pr_gr, inp_size = inp, num_layers_y = 3, step_y = 1.5),
            # models.Fair_rew_RF(un_gr, pr_gr),

            models.FAD_class(input_size = inp, num_layers_z = 3, num_layers_y = 3, 
                                        step_z = 1.5, step_y = 1.5, name = f"FAD_{a}"),
            # models.FAIR_scalar_class(input_size = inp, num_layers_w = 3, step_w = 1.5, 
                        # num_layers_A = 2, step_A = 1.5, num_layers_y = 4, step_y = 1.5),
            # models.FAIR_betaSF_class(input_size = inp, num_layers_w = 3, step_w = 1.5, 
                        # num_layers_A = 2, step_A = 1.5, num_layers_y = 4, step_y = 1.5),
            # models.FAIR_Bernoulli_class(input_size = inp, num_layers_w = 3, step_w = 1.5, 
                        # num_layers_A = 2, step_A = 1.5, num_layers_y = 4, step_y = 1.5),
            # models.FAIR_betaREP_class(input_size = inp, num_layers_w = 3, step_w = 1.5, 
                        # num_layers_A = 2, step_A = 1.5, num_layers_y = 4, step_y = 1.5),
            # models.FAD_prob_class(flow_length = 2, no_sample = 1,
                                                # input_size = inp, num_layers_y = 2, 
                                                # step_y = 2, step_z = 2)
            models.CLFR_class(input_size = inp, num_layers_z = 3, num_layers_y = 3, 
                                        step_z = 1.5, step_y = 1.5, name = f"CLFR_{a}"),
            models.LURMI_class(input_size = inp, num_layers_z = 3, num_layers_y = 3, 
                                        step_z = 1.5, step_y = 1.5, name = f"LURMI_{a}")] 

        k = 0
        for i in lst:  
            
            if np.isin(k ,model_AIF):
                i.fit(data_train, ['labels'], [prot])
                saver_path = os.path.join(saver_dir_models, 'checkpoint_{}_epochs_{}_eta_{}'.format(type(i).__name__, epochs, ind))
            else:
                i.fit(dataloader_train, dataloader_val, max_epoch= epochs, log = 1, alpha = a, log_epoch = 2, early_stopping_no= 1)
                saver_path = os.path.join(saver_dir_models, 'checkpoint_{}_epochs_{}_alpha_{}'.format(type(i).__name__, epochs, a))

            torch.cuda.empty_cache()

            f = open(saver_path,"wb")
            pickle.dump(i,f)
            f.close

            metric_val, metric_test = test(data_val, data_test, i, output_val, output_test, 
                    sensitive_val, sensitive_test, threshold,
                model_AIF, k, dataloader_val, dataloader_test, prot, un_gr, pr_gr)
    
            for column, _ in enumerate(columns):
                if column == 0:
                    name = type(i).__name__
                    sheets.write(row, column, name)
                elif column > 0 and column < 9:
                    sheets.write(row, column , metric_val[column-1])
                elif column == len(columns) - 1:
                    sheets.write(row, column, a)
                else:
                    sheets.write(row, column , metric_test[column-9])
            
            wb.save(saver_dir_results) 
            
            row += 1
            k += 1