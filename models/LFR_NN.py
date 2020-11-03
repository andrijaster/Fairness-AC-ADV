from aif360.datasets import BinaryLabelDataset
from sklearn.ensemble import RandomForestClassifier
from aif360.algorithms.preprocessing import LFR

import torch
import torch.utils.data
import numpy as np
from torch import nn, optim
from torch.nn import functional as F

class FairClass():

    class Model_NN(nn.Module):
        def __init__(self, inp_size, num_layers_y, step_y):                
            super(FairClass.Model_NN, self).__init__()
            
            lst_1 = nn.ModuleList()
            out_size = inp_size
            
            for i in range(num_layers_y):
                inp_size = out_size
                out_size = int(inp_size//step_y)
                if i == num_layers_y-1:
                    block = nn.Linear(inp_size, 1)
                else:
                    block = nn.Sequential(nn.Linear(inp_size, out_size), 
                                      nn.BatchNorm1d(num_features = out_size),nn.ReLU())
                lst_1.append(block)
            
            self.fc1 = nn.Sequential(*lst_1)
            
        def forward(self, x):
            y = torch.sigmoid(self.fc1(x))
            return y
    
    def __init__(self, inp_size, num_layers_y, step_y):
        self.model = FairClass.Model_NN(inp_size, num_layers_y, step_y)


    def fit(self, x_train, y_train, max_epoch = 300, mini_batch_size = 50, 
        alpha = 1, log = 1, log_epoch = 10):
    
        self.model.train()
        nll_criterion =F.binary_cross_entropy
        list_1 = list(self.model.fc1.parameters())
        optimizer_1 = torch.optim.Adam(list_1, lr = 0.001)
        
        for e in range(max_epoch):

            for i in range(0,x_train.size()[0], mini_batch_size):     
                batch_x, batch_y = (x_train[i:i+mini_batch_size], y_train[i:i+mini_batch_size])
                output = self.model(batch_x)
                loss = nll_criterion(output, batch_y)
                optimizer_1.zero_grad()
                loss.backward()
                optimizer_1.step()
                    
    def predict(self, x_test):
        self.model.eval()
        y = self.model(x_test)
        y = np.round(y.data)
        return y
        
    def predict_proba(self, x_test):
        self.model.eval()
        y = self.model(x_test)
        y = y.data
        return y

class Fair_LFR_NN():
    
    def __init__(self, un_gr, pr_gr, inp_size, num_layers_y, step_y, Az):
        
        self.model_reweight = LFR(un_gr,pr_gr, Az = Az)
        self.model = FairClass(inp_size, num_layers_y, step_y)
        
    def fit(self, data, labels, prot):
        ds = BinaryLabelDataset(df = data, label_names = labels, 
                             protected_attribute_names= prot)
        self.prot = prot
        x = self.model_reweight.fit_transform(ds)
        index = x.feature_names.index(prot[0])
        x_train = np.delete(x.features,index,1)
        y_train = x.labels
        x_train = torch.tensor(x_train).type('torch.FloatTensor')
        y_train = torch.tensor(y_train).type('torch.FloatTensor')
        self.model.fit(x_train, y_train)

    def predict_proba(self, data_test):
        x = self.model_reweight.transform(data_test)
        index = x.feature_names.index(self.prot[0])
        x_test = np.delete(x.features,index,1)
        x_test = torch.tensor(x_test).type('torch.FloatTensor')
        y = self.model.predict_proba(x_test)
        return y.data.numpy()
        
