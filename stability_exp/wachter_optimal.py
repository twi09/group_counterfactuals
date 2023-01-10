#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 10:15:39 2023

@author: nwgl2572
"""

from wachter import wachter_recourse
import torch
import numpy as np 


def compute_counterfactuals_wachter(model,X_test_t) : 
    y_pred_test = torch.argmax(model(X_test_t),axis=1)
    C = []
    for i in range(y_pred_test.shape[0]) : 
        if y_pred_test[i] == 1 : 
            y_target = [1,0]
        else : 
            y_target = [0,1]
        ac = wachter_recourse(model,x=X_test_t[i].numpy().reshape(1,-1),y_target=y_target,cat_feature_indices=[],loss_type="BCE")
        C.append(ac)
    C = np.vstack(C)
    return(C)


from torch.autograd import Variable
import torch.optim as optim
import datetime
# Function to round the perturbation
def Adapt(final_perturb) :
    perturb_round = np.zeros(final_perturb.shape)
    for i in range(final_perturb.shape[1]) :
        if np.abs(final_perturb[0][i]) > 1e-3 :
            perturb_round[0][i] = final_perturb[0][i]
    return(perturb_round.astype(np.float32))


# Find a perturbation delta for a group G assigned to class pred_class
def Optimize(model,G,pred_class,delta,lr,max_iter,percentage_number,lambda_param=0.01,t_max_min=0.5) :
    y_target = torch.zeros(1, G.shape[0]) + 1
    device = "cpu"
    # Tensors
    G_target = torch.tensor(y_target).float()
    lamb = torch.tensor(lambda_param).float()

    # init perturb with wachter counterfactual 
    Perturb = Variable(torch.clone(delta), requires_grad=True)
    percentage_sucess = float(sum(model(G+Perturb)[:,1-pred_class] > 0.5) / G.shape[0])   

    # Set optimizer 
    optimizer = optim.Adam([Perturb], lr, amsgrad=True)


    loss_fn_1 = torch.nn.BCELoss()

    t0 = datetime.datetime.now()
    t_max = datetime.timedelta(minutes=t_max_min)
    while percentage_sucess < percentage_number : 
        it=0
        while percentage_sucess < percentage_number  and it < max_iter :
            optimizer.zero_grad()
            loss = loss_fn_1(model(G + Perturb)[:,1-pred_class].reshape(1,-1),G_target) + lamb* torch.dist(G+Perturb, G, 1)
            loss.backward()
            optimizer.step()
            # Percentage of points that are effectively translated (different class)
            percentage_sucess = float(sum(model(G+Perturb)[:,1-pred_class] > 0.5) / G.shape[0])
            it += 1
        lamb -= 0.05
        if datetime.datetime.now() - t0 > t_max:
            print("Timeout - No Counterfactual Explanation Found")
            break
        elif percentage_sucess >= percentage_number:
            continue
            #print("Counterfactual Explanation Found")
    final_perturb = Perturb.cpu().clone().detach().numpy()
    return(final_perturb)

def compute_optimal_wachter(model,X_test_t,C,eps,lr,max_iter,percentage_number) : 
    y_pred_test = torch.argmax(model(X_test_t),axis=1)
    C_optimal = [] 
    for i in range(len(X_test_t)): 
        # Example to explain 
        x0 = X_test_t[i]
        # Predicted class for the example to explain
        pred_class = y_pred_test[i]
        # Counterfactual found with wachter
        c_init = torch.from_numpy(C)[i]
        # Perturbation outputed by wachter 
        delta = c_init-x0
        # Existing points at distance < eps 
        cond_dist = (np.where(torch.linalg.norm(x0-X_test_t,axis=1) < eps)[0])
        # Select only points of the same predicted class (pred_class)
        cond = cond_dist[np.where(model(X_test_t[cond_dist])[:,pred_class] > 0.5)]
        # Group 
        G = X_test_t[cond]
        # Compute the optimization problem 
        perturb = Optimize(model,G,pred_class,delta,lr,max_iter,percentage_number)
        # New optimal counterfactual 
        c0 = (X_test_t[i].numpy() + perturb)
        C_optimal.append(c0)
    C_optimal = np.vstack(C_optimal)
    return(C_optimal)

def compute_noisy_prescribed_recourse(X_test_t,C,model,mu,sigma,repeat_experiment) : 
        predicted_class_counterfactual = torch.argmax(model(torch.from_numpy(C)),axis=1)
        R = []
        for j in range(repeat_experiment) :
            np.random.seed(j)
            epsilon = np.random.normal(mu, sigma, X_test_t.shape)
            predited_class_counterfactual_perturb = torch.argmax(model(torch.from_numpy(C+epsilon).float()),axis=1)
            R.append(predicted_class_counterfactual-predited_class_counterfactual_perturb)
        Noise = torch.mean(torch.vstack(R).float(),axis=0)
        return(float(torch.mean(Noise)))


def compute_noise_input_perturb_classic(X_test_t,C,model,mu,sigma,repeat_experiment) : 
    R = []
    CC_perturbed = []
    for j in range(repeat_experiment) : 
        np.random.seed(j)
        epsilon = np.random.normal(mu, sigma, X_test_t.shape)
        X_perturbed = (X_test_t + epsilon).float()
        C_perturbed = compute_counterfactuals_wachter(model,X_perturbed)
        Dist = np.linalg.norm(C-C_perturbed,axis=1,ord=1)
        R.append(Dist)
        CC_perturbed.append(C_perturbed)
        
    Noise = np.mean(np.vstack(R),axis=0)
    return(float(np.mean(Noise)),CC_perturbed)




def compute_noise_input_perturb_optim(X_test_t,C_optimal,CC_perturbed,model,mu,sigma,repeat_experiment,eps,lr,max_iter,percentage_number) :
    R = []
    for j in range(repeat_experiment) : 
        np.random.seed(j)
        epsilon = np.random.normal(mu, sigma, X_test_t.shape)
        X_perturbed = (X_test_t + epsilon).float()
        C_optimal_perturbed = compute_optimal_wachter(model,X_perturbed,CC_perturbed[j],eps,lr,max_iter,percentage_number)
        Dist = np.linalg.norm(C_optimal-C_optimal_perturbed,axis=1,ord=1)
        R.append(Dist)

    Noise = np.mean(np.vstack(R),axis=0)
    return(float(np.mean(Noise)))





