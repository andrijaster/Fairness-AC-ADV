import torch
import torch.utils.data
import numpy as np
from torch import nn
from torch.nn import functional as F

class FAD_prob_class():
    
    
    class Flow_transform(nn.Module): 
        def __init__(self, step_z, no_sample, input_size):    
            super(FAD_prob_class.Flow_transform, self).__init__()
            self.block = nn.Sequential(nn.Linear(input_size, input_size), nn.ReLU())
        
        def forward(self, x):
            y = self.block(x)
            return y
    
    class Affine_transform(nn.Module):    
        def __init__(self, input_size, step_z, no_sample):
            super(FAD_prob_class.Affine_transform, self).__init__()
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
            eps = torch.randn(number).cuda()
            return mu + eps*std
        
        def forward(self, x):
             mu, logvar = self.P_zx(x)
             z = self.reparameterize(mu, logvar, self.no_samples)
             return z
             
  
    class Y_output(nn.Module):      
        def __init__(self, num_layers_y, step_y, out_size):
            super(FAD_prob_class.Y_output, self).__init__()            
            
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
            super(FAD_prob_class.A_output, self).__init__()            
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

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        output_size = int(input_size//step_z**2)
        
        self.model_y = FAD_prob_class.Y_output(num_layers_y, step_y,
                                                          out_size = output_size)

        self.model_A = FAD_prob_class.A_output(num_layers_A = num_layers_y, 
                                                          step_A = step_y, out_size=output_size)
        self.Transform = nn.Sequential(FAD_prob_class.Affine_transform(input_size, step_z, no_sample), 
                                       *[FAD_prob_class.Flow_transform(step_z, no_sample, input_size=output_size) 
                                       for _ in range(flow_length)])

        self.model_y.to(self.device)
        self.model_A.to(self.device)
        self.Transform.to(self.device)


    def fit(self, dataloader, dataloader_val, early_stopping_no = 3, max_epoch = 100, mini_batch_size = 50, alpha = 1,
            log_epoch = 1, log = 1):
        
        self.model_y.train()
        self.model_A.train()
        self.Transform.train()
               
        nll_criterion = F.binary_cross_entropy
        
        list_1 = list(self.model_y.parameters())+list(self.Transform.parameters())
        list_2 = list(self.model_A.parameters())
        
        optimizer_1 = torch.optim.Adam(list_1, lr = 0.0001)
        optimizer_2 = torch.optim.Adam(list_2, lr = 0.0001)
        prev_loss_y, prev_loss_A = 9e10,9e10
        no_val = 0
        
        for e in range(max_epoch):
            for batch_x, batch_y, batch_A in dataloader:  

                batch_x = batch_x.to(self.device, dtype = torch.float)  
                batch_y = batch_y.unsqueeze(dim = 1).to(self.device, dtype = torch.float)
                batch_A = batch_A.unsqueeze(dim = 1).to(self.device, dtype = torch.float)
                
                z = self.Transform(batch_x)
                output_1 = self.model_y(z)
                output_2 = self.model_A(z)
                                
                loss2 = nll_criterion(output_2, batch_A)
                optimizer_2.zero_grad()
                
                loss1 = nll_criterion(output_1, batch_y) - alpha * nll_criterion(output_2,batch_A)
                optimizer_1.zero_grad()

                loss2.backward(retain_graph=True)
                loss1.backward()

                optimizer_2.step()
                optimizer_1.step()
                
            if e%log_epoch == 0 and  log == 1:
                z = self.Transform(batch_x)
                out_1 = self.model_y(z)
                out_2 = self.model_A(z)

                for x_val, y_val, A_val in dataloader_val:
                    
                    x_val = x_val.to(self.device, dtype = torch.float)  
                    y_val = y_val.unsqueeze(dim = 1).to(self.device, dtype = torch.float)
                    A_val = A_val.unsqueeze(dim = 1).to(self.device, dtype = torch.float)

                    z = self.Transform(x_val)
                    out_1_val = self.model_y(z)
                    out_2_val = self.model_A(z)

                    loss_y_val = nll_criterion(out_1_val, y_val).data.cpu().numpy()
                    loss_A_val = nll_criterion(out_2_val, A_val).data.cpu().numpy()

                    if loss_y_val > prev_loss_y and loss_A_val > prev_loss_A:
                        no_val +=1
                    else:
                        prev_loss_y, prev_loss_A = loss_y_val, loss_A_val
                        no_val = 0
                
                if no_val == early_stopping_no:
                    break
   
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
    
    def predict_proba(self, dataloader):
        for batch_x, _ , _ in dataloader: 
            z = self.Transform(batch_x.to(self.device, dtype=torch.float))
            y = self.model_y(z)
            A = self.model_A(z)
        y = y.data.cpu().numpy()
        A = A.data.cpu().numpy()
        return y, A
    

