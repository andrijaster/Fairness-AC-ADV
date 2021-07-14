import torch
import torch.utils.data
import numpy as np
import os
from torch import nn, optim
from torch.nn import functional as F


from sklearn.metrics import roc_auc_score

class LURMI_class():
    
    def __init__(self, input_size, num_layers_z, num_layers_y, step_z, step_y, name = "LURMI"):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LURMI_class.FairClass(input_size, 
                                                 num_layers_z, num_layers_y, 
                                                 step_z, step_y)
        self.model.to(self.device)
        self.path = os.path.join(r"C:\Users\Andri\Documents\GitHub\Fairness-AC-ADV\models_save",name)

    def loss_min(self, loss_y, T, T_sampled, alpha):
        return loss_y + alpha*(torch.mean(T) - torch.log(torch.mean(torch.exp(T_sampled))))
        

    class FairClass(nn.Module):

        def __init__(self, inp_size, num_layers_z, num_layers_y,  step_z, step_y):                
            super(LURMI_class.FairClass, self).__init__()
            
            num_layers_A = num_layers_y
            lst_z = nn.ModuleList()
            lst_1 = nn.ModuleList()
            lst_2 = nn.ModuleList()
            out_size = inp_size

            
            for i in range(num_layers_z):
                inp_size = out_size
                out_size = int(inp_size//step_z)
                block = nn.Sequential(nn.Linear(inp_size, out_size), 
                                  nn.BatchNorm1d(num_features = out_size),
                                  nn.ReLU())
                lst_z.append(block)
            
            out_size_old = out_size
            for i in range(num_layers_y):
                inp_size = out_size
                out_size = int(inp_size//step_y)
                if i == num_layers_y-1:
                    block = nn.Linear(inp_size, 1)
                else:
                    block = nn.Sequential(nn.Linear(inp_size, out_size), 
                                      nn.BatchNorm1d(num_features = out_size),nn.ReLU())
                lst_1.append(block)

            out_size = out_size_old + 1
            for i in range(num_layers_A):
                inp_size = out_size
                out_size = int(inp_size//step_y)
                if i == num_layers_y-1:
                    block = nn.Linear(inp_size, 1)
                else:
                    block = nn.Sequential(nn.Linear(inp_size, out_size), 
                                      nn.BatchNorm1d(num_features = out_size),nn.ReLU())
                lst_2.append(block)
            
            self.fc1 = nn.Sequential(*lst_z)
            self.fc2 = nn.Sequential(*lst_1)
            self.fc3 = nn.Sequential(*lst_2)
            
        def forward(self, x, sensitive = None, sensitive_sampled = None):
            z = self.fc1(x)
            y = torch.sigmoid(self.fc2(z))

            if sensitive != None and sensitive_sampled != None:
                input_z = torch.cat([sensitive, z],dim = 1)
                input_z_sampled = torch.cat([sensitive_sampled, z],dim = 1) 
                T_sampled = self.fc3(input_z_sampled)
                T = self.fc3(input_z)
                return y, T, T_sampled
            else:
                return y
        

    def fit(self, dataloader, dataloader_val, early_stopping_no = 3, max_epoch = 300, mini_batch_size = 50, 
            alpha = 1, log = 1, log_epoch = 1):
        
        self.model.train()
        nll_criterion = F.binary_cross_entropy
        list_1 = list(self.model.fc1.parameters())+list(self.model.fc2.parameters())
        list_2 = list(self.model.fc3.parameters())
        optimizer_1 = torch.optim.Adam(list_1, lr = 0.0001)
        optimizer_2 = torch.optim.Adam(list_2, lr = 0.0001)
        
        prev_loss_y, prev_loss_A = 9e10,9e10
        no_val = 0

        for e in range(max_epoch):

            self.model.train()
            for batch_x, batch_y, batch_A in dataloader: 

                batch_x = batch_x.to(self.device, dtype = torch.float)  
                batch_y = batch_y.unsqueeze(dim = 1).to(self.device, dtype = torch.float)
                batch_A = batch_A.unsqueeze(dim = 1).to(self.device, dtype = torch.float)
                w = torch.ones(batch_A.shape).to(self.device, dtype = torch.float)*0.5
                dist = torch.distributions.Bernoulli(w)
                batch_A_random = dist.sample()
               
                y_predict, T_predict, T_sample_predict = self.model(batch_x, batch_A, batch_A_random)
                loss2 = -(torch.mean(T_predict) - torch.log(torch.mean(torch.exp(T_sample_predict))))
                optimizer_2.zero_grad()
                loss2.backward()
                optimizer_2.step()

                y_predict, T_predict, T_sample_predict = self.model(batch_x, batch_A, batch_A_random)
                loss_1_y = nll_criterion(y_predict, batch_y)
                loss1 = self.loss_min(loss_1_y, T_predict, T_sample_predict, alpha)                
                optimizer_1.zero_grad()           
                loss1.backward()
                optimizer_1.step()

            if e%log_epoch==0 and log == 1:
                
                for x_val, y_val, A_val in dataloader_val:


                    x_val = x_val.to(self.device, dtype = torch.float) 
                    y_val = y_val.to(self.device, dtype = torch.float).reshape(-1,1)
                    A_val = A_val.to(self.device, dtype = torch.float).reshape(-1,1)

                    self.model.eval()
                    w = torch.ones(A_val.shape).to(self.device, dtype = torch.float)*0.5
                    dist = torch.distributions.Bernoulli(w)
                    batch_A_random = dist.sample()

                    y_predict, T_predict, T_sample_predict = self.model(x_val, A_val, batch_A_random)
                    loss_1_y = nll_criterion(y_predict, y_val)

                    loss2 = -(torch.mean(T_predict) - torch.log(torch.mean(torch.exp(T_sample_predict))))
                    loss1 = self.loss_min(loss_1_y, T_predict, T_sample_predict, alpha)   

                    # print("LOSS 1", loss1)
                    # print("LOSS 2", loss2)

                    if loss2 > prev_loss_y and loss1 > prev_loss_A:
                        no_val +=1
                    else:
                        prev_loss_y, prev_loss_A = loss2, loss1
                        torch.save(self.model.state_dict(), self.path)
                        no_val = 0
                
                if no_val == early_stopping_no:
                    break
                                    
    def predict(self, x_test):
        self.model.load_state_dict(torch.load(self.path))
        self.model.to(self.device).eval()
        y = self.model(x_test)
        y = np.round(y.data)
        return y
    
    def predict_proba(self, dataloader):
        self.model.load_state_dict(torch.load(self.path))
        self.model.to(self.device).eval()
        for batch_x, _ , _ in dataloader: 
            y = self.model(batch_x.to(self.device, dtype=torch.float))
            y = y.data.cpu().numpy()
        return y, _





        
    
    
    
    