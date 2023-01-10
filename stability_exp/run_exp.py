#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 10:25:53 2023

@author: nwgl2572
"""
import numpy as np 
import matplotlib.pyplot as plt
import torch.nn as nn


from sklearn.datasets import  make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import TensorDataset
from wachter_optimal import compute_counterfactuals_wachter
from wachter_optimal import compute_optimal_wachter
from wachter_optimal import compute_noisy_prescribed_recourse,compute_noise_input_perturb_classic,compute_noise_input_perturb_optim
#### LOAD DATASET
X, y = make_moons(n_samples=10_000,noise=0.02,random_state=42)

scaler = MinMaxScaler()
X=scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

X_train_t = torch.from_numpy(X_train).to(torch.float32)
y_train_t = torch.from_numpy(y_train).to(torch.long)
X_test_t = torch.from_numpy(X_test).to(torch.float32)
y_test_t = torch.from_numpy(y_test).to(torch.long)


train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=12, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=12, shuffle=False)


class mlp_model(nn.Module):
    def __init__(self,feature_size):
        super(mlp_model, self).__init__()
        self.linear_1 = torch.nn.Linear(feature_size,50)
        self.linear_2 = torch.nn.Linear(50,2)
        self.softmax = nn.Softmax(dim=1)
        self.elu = nn.ELU()
    def forward(self, x):
        mid_outputs = self.elu(self.linear_1(x))
        outputs = self.softmax(self.linear_2(mid_outputs))
        return outputs
    
# Load trained mlp model 
model = mlp_model(X_train.shape[1])
model.load_state_dict(torch.load("mlp_toy_model_moons"))


mu = 0
sigma = 0.01
repeat_experiment = 1000 

# Compute counterfactuals with Wachter 
C = compute_counterfactuals_wachter(model,X_test_t)
print("Compute wachter counterfactuals sucess")
# Compute noise prescribed recourse for Wachter 
wachter_noisy_recourse = compute_noisy_prescribed_recourse(X_test_t,C,model,mu,sigma,repeat_experiment)
print("Compute noisy presribed recourse wachter sucess")

# Compute noise input for Wachter 
wachter_noisy_input,CC_perturbed = compute_noise_input_perturb_classic(X_test_t,C,model,mu,sigma,repeat_experiment) 
print("Compute noisy input wachter sucess")

# Save results 
results_wachter = np.array([wachter_noisy_recourse,wachter_noisy_input])
np.savetxt("results_wachter",results_wachter)

# Compute optimal counterfactuals for different eps values 
lr = 0.01
max_iter = 500 
percentage_number = 1.0

Noisy_recourse = []
Noisy_input = []
for eps in np.linspace(0.01,0.2,10) : 
    C_optimal = compute_optimal_wachter(model,X_test_t,C,eps,lr,max_iter,percentage_number)
    print("Compute wachter optimal counterfactuals sucess for eps={}".format(eps))
    # Compute noise prescribed recourse for Wachter optim 
    wachter_opt_noisy_recourse = compute_noisy_prescribed_recourse(X_test_t,C_optimal,model,mu,sigma,repeat_experiment)
    # Compute noise input for Wachter optim
    wachter_opt_noisy_input = compute_noise_input_perturb_optim(X_test_t,C_optimal,CC_perturbed,model,mu,sigma,repeat_experiment,eps,lr,max_iter,percentage_number)
    Noisy_recourse.append(wachter_opt_noisy_recourse)
    Noisy_input.append(wachter_opt_noisy_input)

np.savetxt("Noisy_recourse_optim",np.array(Noisy_recourse))
np.savetxt("Noisy_input_optim",np.array(Noisy_input))



