#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 15:50:54 2022

@author: nwgl2572
"""
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
def Optimize(G,pred_class,G_opp,lambda_param,beta_param,lr,max_iter) :
    y_target = torch.zeros(1, G.shape[0]) + (1 - pred_class)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Tensors
    G_target = torch.tensor(y_target).float()
    lamb = torch.tensor(lambda_param).float()
    beta = torch.tensor(beta_param).float()
    # Initialize perturbation as random
    torch.manual_seed(4)
    Perturb = Variable(torch.rand(1, X.shape[1]), requires_grad=True)

    # Set optimizer 
    optimizer = optim.Adam([Perturb], lr, amsgrad=True)


    loss_fn_1 = torch.nn.MSELoss()


    it=0
    while it < max_iter :
        optimizer.zero_grad()
        loss = loss_fn_1(model(G + Perturb).reshape(1,-1),G_target) + lamb* torch.dist(G+Perturb, G, 1) + beta * torch.mean(torch.norm((G+Perturb)-G_opp,dim=1))
        loss.backward()
        optimizer.step()
        it += 1
        #print(loss)
    final_perturb = Perturb.cpu().clone().detach().numpy()
    
    return(Adapt(final_perturb))
    

def add_to_cluser_if(X_cluster,X_other_class,Mat_distance,point,eps,pred_class,lambda_param,beta_param,lr,max_iter,perturb,indexes,list_total) : 
    sucess = False
    x, y = X_cluster[point]
    
    #index of available points within radius
    new_indexes = np.where((np.abs(x - X_cluster[:,0]) <= eps) & (np.abs(y - X_cluster[:,1]) <= eps))[0]
    # select only points that are not in an other existing cluster 
    new_indexes = np.array([e for e in new_indexes if e in list_total])
    #Group of points that contains points in radius eps of x 
    G = X_cluster[new_indexes]
    # Group with KNN on opposit class samples
    G_opp = X_other_class[np.argmin(Mat_distance[new_indexes],axis=1)]
    # Run a perturbation
    new_perturb = Optimize(G,pred_class,G_opp,lambda_param,beta_param,lr,max_iter)
    # Prediction for the perturbated group 
    G_perturbed = model(G + new_perturb).round().flatten().detach().numpy()
    if (np.unique(G_perturbed).shape[0]==1) and G_perturbed[0] == (1-pred_class) :
       sucess=True
       #print("new_indexes_sucess",new_indexes)
       # if all point are in the same cluster stop the process 
       if (len(new_indexes) >= len(list_total)) : 
           sucess = False
       return(sucess,new_indexes,new_perturb)
    else :
        return(sucess,indexes,perturb)

def remove_covered_points(list_cluster,list_total) : 
    # Remove covered points
    list_total = [e for e in list_total if e not in list_cluster[-1]]
    return(list_total)


def clustering_eps(X_test,y_pred_test_class,pred_class,lr,max_iter,lambda_param,beta_param,eps_max) :
    Perturbs = []
    perturb = torch.zeros((1, X.shape[1])) 
    # Take example with the same predicted class
    X_cluster = X_test[y_pred_test_class==pred_class]
    # Examples with the opposit predicted class 
    X_other_class = X_test[y_pred_test_class==(1-pred_class)]
    
    # Matrix of distances between each samples of opposit class  
    Mat_distance = distance_matrix(X_cluster,X_other_class)
    
    # Take a random point 
    random_point = np.random.choice(X_cluster.shape[0])
    # List of indexes point that are in the same cluster 
    list_cluster = []
    # List that contains all the examples 
    list_total = [i for i in range(X_cluster.shape[0])]
    sucess=True
    # list of indexes of a given group (the init point at the begining)
    indexes = [random_point]
    eps = 1e-1
    while (len(list_total)!=0) :
        while sucess and (len(list_total)!=0) :
           #print("Current cluster:",len(list_cluster))
           #print("Current radius:", eps)
           #print("Number of examples in the cluster:",len(indexes))
           sucess,indexes,perturb = add_to_cluser_if(X_cluster,X_other_class,Mat_distance,random_point,eps,pred_class,lambda_param,beta_param,lr,max_iter,perturb,indexes,list_total)
           
           # increase radius
           eps *=2
           if (eps >= eps_max) : 
               sucess = False
               continue 
           
        print("Current cluster:",len(list_cluster))
        print("Current radius:", eps)
        print("Number of examples in the cluster:",len(indexes))
        list_cluster.append(indexes)
        Perturbs.append(perturb)
        list_total = remove_covered_points(list_cluster,list_total)
        if len(list_total)==0:
           continue
         
        # Take a new random point that is not already reach
        random_point = np.random.choice(list_total)
        # Try a new formation 
        sucess = True
        perturb = torch.zeros((1, X.shape[1])) 
        indexes = [random_point]
        eps = 1e-1
        print("New cluster")
        print("Number of instances that remains:",len(list_total))
    # Create a cluster label vector 
    cluster_label = np.zeros(X_cluster.shape[0])
    for i in range(len(list_cluster)) : 
        for e in list_cluster[i] : 
            cluster_label[e]=i 
    np.savetxt("perturbs_class={}_lambda={}.txt".format(pred_class,str(lambda_param)),np.vstack(Perturbs))
    np.savetxt("cluster_label_class={}_lambda={}.txt".format(pred_class,str(lambda_param)),np.vstack(cluster_label))
    return(np.vstack(Perturbs),cluster_label,list_cluster)

'''
sample = np.random.choice(X_test.shape[0],200,replace=False)
X_sample = X_test[sample]
y_sample = y_test[sample]
y_pred_sample_class = y_pred_test_class[sample]
'''


lr = 0.01
max_iter = 500 
lambda_param = 1e-4
beta_param = 1e-1
pred_class = 0 
eps_max = 1
perturbs,cluster_label,list_cluster =  clustering_eps(X_test,y_pred_test_class,int(pred_class),lr,max_iter,lambda_param,beta_param,eps_max)

X_cluster = X_test[y_pred_test_class==pred_class]

plt.figure()
plt.scatter(X_cluster[:,0],X_cluster[:,1],c=cluster_label,cmap="plasma")

#np.savetxt("sample_class={}_lambda={}.txt".format(pred_class,str(lambda_param)),sample)       




