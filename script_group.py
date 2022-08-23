#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 15:46:21 2022

@author: nwgl2572
"""

import pandas as pd
import numpy as pd 
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.optim as optim

import numpy as np 
from torch.autograd import Variable
from scipy.spatial import distance_matrix
import torch.nn as nn
import sys 

np.random.seed(10)

X,y = make_moons(n_samples=10_000,noise=0.08)
plt.scatter(X[:,0],X[:,1],c=y)
plt.title("Make moons dataset")


# Split 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,test_size=0.25, random_state=1)


# Scale 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)


# Convert to torch tensors
X_train=torch.from_numpy(X_train.astype(np.float32))
X_test=torch.from_numpy(X_test.astype(np.float32))
y_train=torch.from_numpy(y_train.astype(np.float32))
y_test=torch.from_numpy(y_test.astype(np.float32))
X_val=torch.from_numpy(X_val.astype(np.float32))
y_val=torch.from_numpy(y_val.astype(np.float32))

# Reshape to fit a model
y_train=y_train.view(y_train.shape[0],1)
y_test=y_test.view(y_test.shape[0],1)
y_val=y_val.view(y_val.shape[0],1)



# MLP in pytorch 
class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.input = torch.nn.Linear(2, 5)
        self.output = torch.nn.Linear(5, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.ReLU = torch.nn.ReLU()
    def forward(self, x):
        x_input = self.input(x)
        x_output = self.output(self.ReLU(x_input))
        out = self.sigmoid(x_output)
        return out   


model = MLP()

model.load_state_dict(torch.load("mlp_toy_model"))



with torch.no_grad():
    # Predictions on train 
    y_pred_train_proba = model(X_train)
    y_pred_train_class = y_pred_train_proba.round().numpy().flatten()
    # Predictions on test 
    y_pred_test_proba = model(X_test)
    y_pred_test_class = y_pred_test_proba.round().numpy().flatten()
    
    
    
# Function to round the perturbation
def Adapt(final_perturb) :
    perturb_round = np.zeros(final_perturb.shape)
    for i in range(final_perturb.shape[1]) :
        if np.abs(final_perturb[0][i]) > 1e-3 :
            perturb_round[0][i] = final_perturb[0][i]
    return(perturb_round.astype(np.float32))



# Find a perturbation delta for a group G assigned to class pred_class
def Optimize(G,pred_class,lambda_param,lr,max_iter) :
    y_target = torch.zeros(1, G.shape[0]) + (1 - pred_class)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Tensors
    G_target = torch.tensor(y_target).float().to(device)
    lamb = torch.tensor(lambda_param).float().to(device)

    # Initialize perturbation as random
    torch.manual_seed(0)
    Perturb = Variable(torch.rand(1, X.shape[1]), requires_grad=True)

    # Set optimizer 
    optimizer = optim.Adam([Perturb], lr, amsgrad=True)


    loss_fn_1 = torch.nn.MSELoss()


    it=0
    while it < max_iter :
        optimizer.zero_grad()
        loss = loss_fn_1(model(G + Perturb).reshape(1,-1),G_target) + lamb* torch.dist(G+Perturb, G, 1)
        loss.backward()
        optimizer.step()
        it += 1
        #print(loss)
    final_perturb = Perturb.cpu().clone().detach().numpy()
    
    return(Adapt(final_perturb))
    
def add_to_cluster_if(X_cluster,list_cluster,i,x,pred_class,lambda_param,lr,max_iter) :
    success = False
    list_cluster[i].append(x)
    G = X_cluster[list_cluster[i],:]
    # Run a perturbation
    perturb = Optimize(G,pred_class,lambda_param,lr,max_iter)
    # Prediction for the perturbated group 
    G_perturbed = model(G + perturb).round().flatten().detach().numpy()
    if np.all(G_perturbed) and G_perturbed[0] == 1-pred_class :
        success=True
    else : 
        list_cluster[i].remove(x)

    return(success,list_cluster,perturb)

def remove_covered_points(list_cluster,i,list_total) : 
    # Remove covered points
    list_total = [e for e in list_total if e not in list_cluster[i]]
    return(list_total)
    
def clustering_2(X_test,y_pred_test_class,pred_class,lr,max_iter,lambda_param) :
    Perturbs = []
    # Take example with the same predicted class
    X_cluster = X_test[y_pred_test_class==pred_class]
    #X_cluster_init = X_cluster.clone()
    # Distance matrix
    Mat = distance_matrix(X_cluster,X_cluster)
    # Take a random point 
    random_point = np.random.choice(X_cluster.shape[0])
    # List of indexes point that are in the same cluster 
    list_cluster = [[random_point]]
    # List that contains all the examples 
    list_total = [i for i in range(X_cluster.shape[0])]
    i=0
    sucess=True
    while (len(list_total)!=0) :
        while sucess and (len(list_total)!=0) :
            print("Current cluster:",i)
            print("Current number of instances in the cluster:",len(list_cluster[i]))
            #print("Number of instances that remains:",len(list_total))
            # Closest point in data
            closest_point_index = np.argsort(Mat[random_point])[1]
            # test to add this point to the cluster
            sucess,list_cluster,perturb = add_to_cluster_if(X_cluster,list_cluster,i,closest_point_index,pred_class,lambda_param,lr,max_iter)
            # remove covered points
            list_total = remove_covered_points(list_cluster,i,list_total)
            # Points in a cluster cant be long covered
            Mat[:,random_point] = float("inf")
            # new point is the closest one
            random_point = closest_point_index
        
        # If the formation cluster stop append the perturbation
        Perturbs.append(perturb)
        # Take a new random point that is not already reach
        print("Number of instances that remains:",len(list_total))
        if len(list_total)==0:
            continue
        random_point = np.random.choice(list_total)
        list_cluster.append([random_point])
        # Try a new formation 
        sucess = True
        i+=1
        #print("New cluster",i)
    # Create a cluster label vector 
    cluster_label = np.zeros(X_cluster.shape[0])
    for i in range(len(list_cluster)) : 
        for e in list_cluster[i] : 
            cluster_label[e]=i 
    np.savetxt("perturbs_class={}.txt".format(pred_class),np.vstack(Perturbs))
    np.savetxt("cluster_label_class={}.txt".format(pred_class),np.vstack(cluster_label))    
    

sample = np.random.choice(X_test.shape[0],200,replace=False)
X_sample = X_test[sample]
y_sample = y_test[sample]
y_pred_sample_class = y_pred_test_class[sample]

lr = 0.01
max_iter = 1000 
lambda_param = 1e-3

def run_clustering(pred_class) : 
    lr = 0.01
    max_iter = 1000 
    lambda_param = 1e-3
    clustering_2(X_test,y_pred_test_class,int(pred_class),lr,max_iter,lambda_param)


run_clustering(sys.argv[1])




'''cluster_label = np.loadtxt("cluster_label.txt")
X_cluster = X_sample[y_pred_sample_class==pred_class]
plt.figure()
plt.scatter(X_cluster[:,0],X_cluster[:,1],c=cluster_label)

'''

