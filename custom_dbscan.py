#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 11:26:09 2022

@author: nwgl2572
"""
import pandas as pd
import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.optim as optim
import random
import numpy as np 
from torch.autograd import Variable
from scipy.spatial import distance_matrix
import torch.nn as nn
import sys 
from sklearn.cluster import DBSCAN
np.random.seed(3)

X,y = make_moons(n_samples=10_000,noise=0.08)
#plt.scatter(X[:,0],X[:,1],c=y)
#plt.title("Make moons dataset")


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
    G_target = torch.tensor(y_target).float()
    lamb = torch.tensor(lambda_param).float()

    # Initialize perturbation as random
    torch.manual_seed(4)
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


def check_core_point(eps,minPts, X_cluster, index,all_index):
    #get points from given index
    x, y = X_cluster[index][0]  ,  X_cluster[index][1]
    
    #index of available points within radius
    indexes = np.where((np.abs(x - X_cluster[:,0]) <= eps) & (np.abs(y - X_cluster[:,1]) <= eps) & (all_index !=index))[0]
   
    #check how many points are present within radius
    if len(indexes) >= minPts:
        #return format (dataframe, is_core, is_border, is_noise)
        return (indexes , True, False, False)
    
    elif (len(indexes) < minPts) and len(indexes) > 0:
        #return format (dataframe, is_core, is_border, is_noise)
        return (indexes, False, True, False)
    
    elif len(indexes) == 0:
        #return format (dataframe, is_core, is_border, is_noise)
        return (indexes , False, False, True)


    
def cluster_with_stack(eps, minPts, X_test):
    
    # Take example with the same predicted class
    #X_cluster = X_test[y_pred_test_class==pred_class]
    X_cluster = X_test
    #initiating cluster number
    C = 1
    #initiating stacks to maintain
    current_stack = set()
    unvisited = [i for i in range(X_cluster.shape[0])]
    all_index = [i for i in range(X_cluster.shape[0])]
    clusters = []
    
    while (len(unvisited) != 0): #run until all points have been visited
    
        #identifier for first point of a cluster
        first_point = True
        
        #choose a random unvisited point
        current_stack.add(random.choice(unvisited))
        
        while len(current_stack) != 0: #run until a cluster is complete
            
            #pop current point from stack
            curr_idx = current_stack.pop()
            
            #check if point is core, neighbour or border
            neigh_indexes, iscore, isborder, isnoise = check_core_point(eps, minPts, X_cluster, curr_idx,all_index)
            #dealing with an edge case
            if (isborder & first_point):
                #for first border point, we label it aand its neighbours as noise 
                clusters.append((curr_idx, 0))
                clusters.extend(list(zip(neigh_indexes,[0 for _ in range(len(neigh_indexes))])))
                #label as visited
                unvisited.remove(curr_idx)
                unvisited = [e for e in unvisited if e not in neigh_indexes]
    
                continue
                
            unvisited.remove(curr_idx) #remove point from unvisited list
            neigh_indexes = set(neigh_indexes) & set(unvisited) #look at only unvisited points
            
            if iscore: #if current point is a core
                first_point = False
                
                clusters.append((curr_idx,C)) #assign to a cluster
                current_stack.update(neigh_indexes) #add neighbours to a stack
   
            elif isborder: #if current point is a border point
                clusters.append((curr_idx,C))
                
                continue
   
            elif isnoise: #if current point is noise
                clusters.append((curr_idx, 0))
                
                continue
                
        if not first_point:
            #increment cluster number
            C+=1
        
    return clusters
   
'''
sample = np.random.choice(X_test.shape[0],200,replace=False)
X_sample = X_test[sample] 
y_sample = y_test[sample]
y_pred_sample_class = y_pred_test_class[sample]

minPts = 2 
eps = 0.1
pred_class = 0 

X_cluster = X_test[y_pred_test_class==pred_class]

clusters = cluster_with_stack(eps, minPts, X_test,y_pred_test_class,pred_class)
   


'''
from sklearn.datasets import make_blobs
centers = [(0, 4), (5, 5) , (8,2)]
cluster_std = [1.2, 1, 1.1]

X, y= make_blobs(n_samples=200, cluster_std=cluster_std, centers=centers, n_features=2, random_state=1)
#radius of the circle defined as 0.6
eps = 0.6
#minimum neighbouring points set to 3
minPts = 3

clustered = cluster_with_stack(eps, minPts,X)


idx , cluster = list(zip(*clustered))
cluster_df = pd.DataFrame(clustered, columns = ["idx", "cluster"])

plt.figure(figsize=(10,7))
for clust in np.unique(cluster):
    plt.scatter(X[cluster_df["idx"][cluster_df["cluster"] == clust].values, 0], X[cluster_df["idx"][cluster_df["cluster"] == clust].values, 1], s=10, label=f"Cluster{clust}")

plt.legend([f"Cluster {clust}" for clust in np.unique(cluster)], loc ="lower right")
plt.title('Clustered Data')
plt.xlabel('X')
plt.ylabel('Y')
   
   
