import torch
import torch.utils.data
import numpy as np
from torch import nn
from torch.nn import functional as F


class FAIR_betaREP_class():
    
    class Fair_classifier(nn.Module):
        
        def __init__(self, input_size, num_layers_w,
                     step_w, num_layers_A, step_A,
                      num_layers_y, step_y):
            super(FAIR_betaREP_class.Fair_classifier, self).__init__()
            
            out_size_y = input_size
            lst_y = nn.ModuleList()
            
            for i in range(num_layers_y):
                inp_size = out_size_y
                out_size_y = int(inp_size//step_y)
                if i == num_layers_y-1:
                    block = nn.Linear(inp_size, 1)
                else:
                    block = nn.Sequential(nn.Linear(inp_size, out_size_y), 
                                      nn.BatchNorm1d(num_features = out_size_y),nn.ReLU())
                lst_y.append(block)

            out_size_A = input_size
            lst_A = nn.ModuleList()
            
            for i in range(num_layers_A):
                inp_size = out_size_A
                out_size_A = int(inp_size//step_A)
                if i == num_layers_A-1:
                    block = nn.Linear(inp_size, 1)
                else:
                    block = nn.Sequential(nn.Linear(inp_size, out_size_A), 
                                      nn.BatchNorm1d(num_features = out_size_A),nn.ReLU())
                lst_A.append(block)
                
            out_size_w = input_size
            lst_w = nn.ModuleList()
            
            for i in range(num_layers_w):
                inp_size = out_size_w
                out_size_w = int(inp_size//step_w)
                if i == num_layers_w-1:
                    block = nn.Linear(inp_size, 1)
                else:
                    block = nn.Sequential(nn.Linear(inp_size, out_size_w), 
                                      nn.BatchNorm1d(num_features = out_size_w),nn.ReLU())
                lst_w.append(block)
                
            self.fc1 = nn.Sequential(*lst_y)
            self.fc2 = nn.Sequential(*lst_A)
            self.fc3 = nn.Sequential(*lst_w)
            self.fc4 = nn.Sequential(*lst_w)
    
    
        def forward(self, x):
            output_y = torch.sigmoid(self.fc1(x))
            output_A = torch.sigmoid(self.fc2(x))
            output_alfa = torch.max(torch.ones(len(x),1)*1e-5, torch.exp(self.fc3(x))) 
            output_beta = torch.max(torch.ones(len(x),1)*1e-5, torch.exp(self.fc4(x)))
            return output_y, output_A, output_alfa, output_beta
    
    def __init__(self, input_size, num_layers_w,
                     step_w, num_layers_A, step_A,
                      num_layers_y, step_y):
        self.model = FAIR_betaREP_class.Fair_classifier(input_size, num_layers_w,
                     step_w, num_layers_A, step_A,
                      num_layers_y, step_y)
    
    def fit(self, x_train, y_train, A_train, max_epoch = 500, mini_batch_size = 50, 
            alpha = 1.0, beta = 0, log_epoch = 10, log = 1):
        
        def loss(output, target, weights):
            output = torch.clamp(output, 1e-5, 1 - 1e-5)
            weights = torch.clamp(weights, 1e-5, 1 - 1e-5)
            ML =  weights*(target*torch.log(output) + (1-target)*torch.log(1-output))
            return torch.neg(torch.mean(ML))
    
        def loss_w(output_A, output_Y, target_A, target_Y, alpha, beta, w):
            output_A = torch.clamp(output_A, 1e-5, 1 - 1e-5)
            output_Y = torch.clamp(output_Y, 1e-5, 1 - 1e-5)
            ML = loss(output_Y, target_Y, w) - alpha*loss(output_A, target_A, w) - beta * torch.mean(w)
            return torch.neg(torch.sum(ML))
        
        self.model.train()
        nll_criterion =F.binary_cross_entropy
        list_0 = list(self.model.fc1.parameters())
        list_1 = list(self.model.fc3.parameters()) + list(self.model.fc4.parameters())
        list_2 = list(self.model.fc2.parameters())
        
        optimizer_0 = torch.optim.Adam(list_0, lr = 0.0001)
        optimizer_1 = torch.optim.Adam(list_1, lr = 0.0001)
        optimizer_2 = torch.optim.Adam(list_2, lr = 0.0001)
    
        for e in range(max_epoch):
            for i in range(0,x_train.size()[0], mini_batch_size):     
                batch_x, batch_y, batch_A = (x_train[i:i+mini_batch_size], y_train[i:i+mini_batch_size], 
                                            A_train[i:i+mini_batch_size])
                y, A, alfa, beta = self.model(batch_x)
                
                dist = torch.distributions.Beta(alfa, beta)
                w = dist.rsample()
                
                optimizer_0.zero_grad()
                loss0 = loss(y, batch_y, w) 
                loss0.backward(retain_graph = True)
                optimizer_0.step()
                
                optimizer_2.zero_grad()           
                loss2 = loss(A, batch_A, w)
                loss2.backward(retain_graph = True)
                optimizer_2.step()
                
                optimizer_1.zero_grad()
                loss1 = loss_w(A, y, batch_A, batch_y, alpha, beta, w)
                loss1.backward()
                optimizer_1.step()
    
                
            if e%log_epoch == 0 and log == 1:
                y, A, alfa, beta = self.model(x_train)
                dist = torch.distributions.Beta(alfa, beta)
                w = dist.sample()
                print(min(w.data),max(w.data),torch.mean(w).data,torch.sum(w).data)
                print(nll_criterion(A, A_train).data, nll_criterion(y, y_train).data)
    

    def predict(self, x_test):
        self.model.eval()
        y,A,_,_= self.model(x_test)
        y = np.round(y.data)
        A = np.round(A.data)
        return y, A

    def predict_proba(self, x_test):
        self.model.eval()
        y,A,_,_= self.model(x_test)
        y = y.data
        A = A.data
        return y, A
