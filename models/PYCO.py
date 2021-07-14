import torch
import torch.utils.data
import numpy as np
import os
from os import walk
from torch import nn, optim
from torch.nn import functional as F
from utilities import metrics
from torch.distributions.multivariate_normal import MultivariateNormal
from scipy.optimize import linprog


from sklearn.metrics import roc_auc_score

class PYCO(metrics):
    
    def __init__(self, input_size, num_layers, step,  path = "name"):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_y = PYCO.FairClass(input_size, num_layers, step).to(self.device)
        self.model_lagrangian =  PYCO.Lagrange_multiplicator(no_of_constraints=4).to(self.device)
        self.model_y.to(self.device)
        self.path = path
        
    class FairClass(nn.Module):
        def __init__(self, inp_size, num_layers, step):                
            super(PYCO.FairClass, self).__init__()
        
            lst_y = nn.ModuleList()
            out_size = inp_size
            
            for i in range(num_layers):
                inp_size = out_size
                out_size = int(inp_size//step)
                if i == num_layers-1:
                    block = nn.Linear(inp_size, 1)
                else:
                    block = nn.Sequential(nn.Linear(inp_size, out_size), 
                                      nn.BatchNorm1d(num_features = out_size),nn.ReLU())
                lst_y.append(block)
            
            self.fc1 = nn.Sequential(*lst_y)
            
        def forward(self, x):
            y = torch.sigmoid(self.fc1(x))
            return y
    
    class Lagrange_multiplicator(nn.Module):

        def __init__(self, no_of_constraints):
            super(PYCO.Lagrange_multiplicator, self).__init__()
            self.mu_lambda = nn.Parameter(torch.rand(no_of_constraints), requires_grad=True)
            self.sigma_lambda = nn.Parameter(torch.rand(no_of_constraints), requires_grad=True)
            
        def forward(self, sample_row):
            covariance = torch.diag(torch.abs(self.sigma_lambda))
            dist = MultivariateNormal(self.mu_lambda, covariance)
            y = dist.rsample(sample_shape=(sample_row,1))  
            y = torch.abs(y)
            return y
        
    def fit(self, dataloader, dataloader_val, loss, learning_rate = [1e-4, 1e-4], max_epochs = 5000, 
            alpha = 1, log = 1, log_epoch = 1, show_epoch = 10, differentiable = True):

        
        self.differentiable = differentiable
        
        self.model_y.train()
        self.model_lagrangian.train()

        list_w = list(self.model_y.parameters())
        list_LM = list(self.model_lagrangian.parameters())
        optimizer_y = torch.optim.Adam(list_w, lr = learning_rate[0])
        optimizer_LM = torch.optim.Adam(list_LM, lr = learning_rate[1])
        self.list_models = []

        for e in range(max_epochs):
            for batch_x, batch_y, batch_A in dataloader: 

                batch_x = batch_x.to(self.device, dtype = torch.float) 
                batch_y = batch_y.reshape([-1,1]).to(self.device, dtype = torch.float)
                batch_A = batch_A.reshape([-1,1]).data.cpu().numpy()
               
                classifier = self.model_y(batch_x)
                lagrangian = self.model_lagrangian(sample_row = 1).squeeze()
                
                loss_Max = -loss(batch_y, batch_A, classifier, lagrangian, differentiable = differentiable)
                optimizer_LM.zero_grad() 
                loss_Max.backward()
                optimizer_LM.step()

                classifier = self.model_y(batch_x)
                lagrangian = self.model_lagrangian(sample_row = 1).squeeze()

                loss_Min = loss(batch_y, batch_A, classifier, lagrangian)
                optimizer_y.zero_grad() 
                loss_Min.backward()
                optimizer_y.step()

            if e%log_epoch==0 and log == 1:
                
                self.model_y.eval()
                self.model_lagrangian.eval()

                for x_val, y_val, A_val in dataloader_val:
                    
                    x_val = x_val.to(self.device, dtype = torch.float) 
                    y_val = y_val.reshape([-1,1]).to(self.device, dtype = torch.float)

                    classifier = self.model_y(x_val)
                    lagrangian = self.model_lagrangian(sample_row = 1).squeeze()

                    loss_eval, constraint = loss(y_val, A_val, classifier, lagrangian, eval = False, val = True, differentiable = differentiable)

                    if e%show_epoch == 0:
                        print("Epoch --- {}".format(e) , "Loss Eval --- {}".format(loss_eval), "Constraints --- {}".format(constraint))
                
                if not os.path.exists(self.path):
                    os.makedirs(self.path)
                self.save(self.path + "/{}".format(e))
                self.list_models.append(self.path + "/{}".format(e))
        
        loss_model = []
        constraints = []
        for i in self.list_models:
            self.load(i)
            for x_val, y_val, A_val in dataloader_val:
                    
                x_val = x_val.to(self.device, dtype = torch.float) 
                y_val = y_val.reshape([-1,1]).to(self.device, dtype = torch.float)

                classifier = self.model_y(x_val)
                lagrangian = self.model_lagrangian(sample_row = 1).squeeze()

                loss_eval, constraint = loss(y_val, A_val, classifier, lagrangian, eval = False, val = True, differentiable = differentiable)
                constraints.append(constraint)
                loss_model.append(loss_eval)
            
        loss_model = np.array(loss_model)
        constraints_uq = np.array(constraints).reshape(1,-1)
        constraints_eq = np.ones(constraints_uq.shape)
        bounds = ((0,1),)*loss_model.shape[0]
        res = linprog(c = loss_model, A_ub = constraints_uq, A_eq = constraints_eq, bounds= bounds, b_ub = [0], b_eq = [1])
        self.weights = np.array(res.x)
                                    
    def predict(self, dataloader, eval = True):

        ASD = []
        equal_FNR = []
        AEOD = []
        AOD = []
        AUC = []
        Accuracy = []

        for i in self.list_models:
            self.load(i)
            self.model_y.eval()
            for x_, y_, A_  in dataloader: 
                sens_index_1 = np.where(A_ == 1)[0]
                sens_index_0 = np.where(A_ == 0)[0]
                x_ = x_.to(self.device, dtype = torch.float) 
                y_ = y_.data.cpu().numpy()
                A_ = A_.data.cpu().numpy()
                y_pred_proba = self.model_y(x_)
                y_pred_proba = y_pred_proba.data.cpu().numpy()

            return y_pred_proba
    
    def save(self, path):
        torch.save(self.model_y.state_dict(), path)

    def load(self, path):
        self.model_y.load_state_dict(torch.load(path))





        
    
    
    
    