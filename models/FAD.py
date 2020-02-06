import torch
import torch.utils.data
import numpy as np
from torch import nn, optim
from torch.nn import functional as F



class FAD_class():
    
    def __init__(self, input_size, num_layers_z, num_layers_y, step_z, step_y):
        self.model = FAD_class.FairClass(input_size, 
                                                 num_layers_z, num_layers_y, 
                                                 step_z, step_y)
        
    
    class FairClass(nn.Module):
        def __init__(self, inp_size, num_layers_z, num_layers_y,  step_z, step_y):                
            super(FAD_class.FairClass, self).__init__()
            
            lst_z = nn.ModuleList()
            lst_1 = nn.ModuleList()
            out_size = inp_size
            
            for i in range(num_layers_z):
                inp_size = out_size
                out_size = int(inp_size//step_z)
                block = nn.Sequential(nn.Linear(inp_size, out_size), 
                                  nn.BatchNorm1d(num_features = out_size),
                                  nn.ReLU())
                lst_z.append(block)
            
            for i in range(num_layers_y):
                inp_size = out_size
                out_size = int(inp_size//step_y)
                if i == num_layers_y-1:
                    block = nn.Linear(inp_size, 1)
                else:
                    block = nn.Sequential(nn.Linear(inp_size, out_size), 
                                      nn.BatchNorm1d(num_features = out_size),nn.ReLU())
                lst_1.append(block)
            
            self.fc1 = nn.Sequential(*lst_z)
            self.fc2 = nn.Sequential(*lst_1)
            self.fc3 = nn.Sequential(*lst_1)
            
        def forward(self, x):
            z = self.fc1(x)
            y = torch.sigmoid(self.fc2(z))
            A = torch.sigmoid(self.fc3(z))
            return y, A
        
    def fit(self, x_train, y_train, A_train, max_epoch = 300, mini_batch_size = 50, 
            alpha = 1, log = 1, log_epoch = 10):
        
        self.model.train()
        nll_criterion =F.binary_cross_entropy
        list_1 = list(self.model.fc1.parameters())+list(self.model.fc2.parameters())
        list_2 = list(self.model.fc3.parameters())
        optimizer_1 = torch.optim.Adam(list_1, lr = 0.001)
        optimizer_2 = torch.optim.Adam(list_2, lr = 0.001)
        for e in range(max_epoch):
            for i in range(0,x_train.size()[0], mini_batch_size):     
                batch_x, batch_y, batch_A = (x_train[i:i+mini_batch_size], y_train[i:i+mini_batch_size], 
                                            A_train[i:i+mini_batch_size])
                output_1, output_2 = self.model(batch_x)
                loss2 = nll_criterion(output_2, batch_A)
                optimizer_2.zero_grad()
                loss2.backward(retain_graph=True)
                optimizer_2.step()
                loss1 = nll_criterion(output_1, batch_y) - alpha * nll_criterion(output_2,batch_A)
                optimizer_1.zero_grad()
                loss1.backward()
                optimizer_1.step()
            if e%log_epoch==0 and log == 1:
                out_1, out_2 = self.model(x_train)
                print(nll_criterion(out_2, A_train).data, nll_criterion(out_1, y_train).data)
                
    def predict(self, x_test):
        self.model.eval()
        y, A = self.model(x_test)
        y = np.round(y.data)
        A = np.round(A.data)
        return y, A
    
    def predict_proba(self, x_test):
        self.model.eval()
        y, A = self.model(x_test)
        y, A = y.data, A.data
        return y, A





        
    
    
    
    