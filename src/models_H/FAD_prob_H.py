import torch
import torch.utils.data
import numpy as np
from torch import nn
from torch.nn import functional as F

class FAD_prob_H_class():
    
    
    class Flow_transform(nn.Module): 
        def __init__(self, step_z, no_sample, input_size):    
            super(FAD_prob_H_class.Flow_transform, self).__init__()
            self.block = nn.Sequential(nn.Linear(input_size, input_size), nn.ReLU())
        
        def forward(self, x):
            y = self.block(x)
            return y
    
    class Affine_transform(nn.Module):    
        def __init__(self, input_size, step_z, no_sample):
            super(FAD_prob_H_class.Affine_transform, self).__init__()
            output = int(input_size//step_z)
            self.fc1 = nn.Sequential(nn.Linear(input_size, output),
                    nn.BatchNorm1d(num_features=output),
                    nn.ReLU())
            self.fc21 = nn.Linear(output, int(input_size//step_z**2))
            self.fc22 = nn.Linear(output, int(input_size//step_z**2))
            self.no_samples = no_sample
    
            
        def P_zx(self, x):
            h1 = self.fc1(x)
            return self.fc21(h1), self.fc22(h1)  
        
        def reparameterize(self, mu, logvar, no_samples):
            std = torch.exp(0.5*logvar)
            number = list(std.size())
            number.insert(0,no_samples)
            eps = torch.randn(number)
            return mu + eps*std
        
        def forward(self, x):
             mu, logvar = self.P_zx(x)
             z = self.reparameterize(mu, logvar, self.no_samples)
             return z
             
  
    class Y_output(nn.Module):      
        def __init__(self, num_layers_y, step_y, out_size):
            super(FAD_prob_H_class.Y_output, self).__init__()            
            
            lst_y = nn.ModuleList()
            
            for i in range(num_layers_y):
                inp_size = out_size
                out_size = int(inp_size//step_y)
                if i == num_layers_y-1:
                    block = nn.Linear(inp_size, 1)
                else:
                    block = nn.Sequential(nn.Linear(inp_size, out_size), nn.ReLU())
                lst_y.append(block)            
            
            self.fc31 =  nn.Sequential(*lst_y)
        
        def forward(self, z):
            y = torch.sigmoid(self.fc31(z))
            return torch.mean(y, dim=0)   
       

    class A_output(nn.Module):
        def __init__(self, out_size, num_layers_A, step_A):
            super(FAD_prob_H_class.A_output, self).__init__()            
            lst_A= nn.ModuleList()
            
            for i in range(num_layers_A):
                inp_size = out_size
                out_size = int(inp_size//step_A)
                if i == num_layers_A-1:
                    block = nn.Linear(inp_size, 1)
                else:
                    block = nn.Sequential(nn.Linear(inp_size, out_size), nn.ReLU())
                lst_A.append(block)            
            
            self.fc41 =  nn.Sequential(*lst_A)
            
        def forward(self, z):
            A = torch.sigmoid(self.fc41(z))
            return torch.mean(A, dim=0)
                        
    
    def __init__(self, flow_length, no_sample, num_layers_y, input_size, step_z, step_y):
        
        output_size = int(input_size//step_z**2)
        
        self.model_y = FAD_prob_H_class.Y_output(num_layers_y, step_y,
                                                          out_size = output_size)
        self.model_A = FAD_prob_H_class.A_output(num_layers_A = num_layers_y, 
                                                          step_A = step_y, out_size=output_size)
        self.Transform = nn.Sequential(FAD_prob_H_class.Affine_transform(input_size,
                                                                                  step_z, no_sample), 
                                       *[FAD_prob_H_class.Flow_transform(step_z, no_sample, input_size=output_size) 
                                       for _ in range(flow_length)])


    def fit(self, x_train, y_train, A_train, max_epoch = 100, mini_batch_size = 50, alpha = 1,
            log_epoch = 1, log = 1):
        
        
        def entropy(output):
            output = torch.clamp(output, 1e-5, 1 - 1e-5)
            entropy = -output*torch.log(output)
            return torch.mean(entropy)
            
        self.model_y.train()
        self.model_A.train()
        self.Transform.train()
               
        nll_criterion = F.binary_cross_entropy
        
        list_1 = list(self.model_y.parameters())+list(self.Transform.parameters())
        list_2 = list(self.model_A.parameters())
        
        optimizer_1 = torch.optim.Adam(list_1, lr = 0.001)
        optimizer_2 = torch.optim.Adam(list_2, lr = 0.001)
        
        for e in range(max_epoch):
            for i in range(0,x_train.size()[0], mini_batch_size):     
               
                batch_x, batch_y, batch_A = (x_train[i:i+mini_batch_size], y_train[i:i+mini_batch_size], 
                                            A_train[i:i+mini_batch_size])
                
                z = self.Transform(batch_x)
                output_1 = self.model_y(z)
                output_2 = self.model_A(z)
                                
                loss2 = nll_criterion(output_2, batch_A)
                optimizer_2.zero_grad()
                loss2.backward(retain_graph=True)
                optimizer_2.step()
                
                loss1 = nll_criterion(output_1, batch_y) - alpha * entropy(output_2)
                optimizer_1.zero_grad()
                loss1.backward()
                optimizer_1.step()
                
            if e%log_epoch == 0 and  log == 1:
                z = self.Transform(x_train)
                out_1 = self.model_y(z)
                out_2 = self.model_A(z)
                print(nll_criterion(out_2, A_train).data, nll_criterion(out_1, y_train).data)

   
    def predict(self, x_test):
        self.model_y.eval()
        self.model_A.eval()
        self.Transform.eval()
        z = self.Transform(x_test)
        y = self.model_y(z)
        A = self.model_A(z)
        y = np.round(y.data)
        A = np.round(A.data)
        return y, A
    
    def predict_proba(self, x_test):
        self.model_y.eval()
        self.model_A.eval()
        self.Transform.eval()
        z = self.Transform(x_test)
        y = self.model_y(z)
        A = self.model_A(z)
        y = y.data
        A = A.data
        return y, A
    

