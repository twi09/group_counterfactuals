#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 15:15:34 2022

@author: nwgl2572
"""

from scipy.spatial import distance
import numpy as np 
import matplotlib.pyplot as plt 

def neighborsGen(X, point, eps, metric):
    """
    Generates neighborhood graph for a given point
    """
    
    neighbors = []
    
    for i in range(X.shape[0]):
        if metric(X[point], X[i]) < eps:
            neighbors.append(i)
    
    return neighbors




def expand(X, clusters, point, neighbors, currentPoint, eps, minPts, metric):
    """
    Expands cluster from a given point until neighborhood boundaries are reached
    """
    clusters[point] = currentPoint
    
    i = 0
    while i < len(neighbors):
        
        nextPoint = neighbors[i]
        
        if clusters[nextPoint] == -1:
            clusters[nextPoint] = currentPoint
        
        elif clusters[nextPoint] == 0:
            clusters[nextPoint] = currentPoint
            
            nextNeighbors = neighborsGen(X, nextPoint, eps, metric)
            
            if len(nextNeighbors) >= minPts:
                neighbors = neighbors + nextNeighbors
        
        i += 1

def simple_DBSCAN(X, clusters, eps, minPts, metric=distance.euclidean):
    """
    Driver; 
    iterates through neighborsGen for every point in X
    expands cluster for every point not determined to be noise
    """
    currentPoint = 0
    
    for i in range(0, X.shape[0]):
        if clusters[i] is not 0:
            continue
    
        neighbors = neighborsGen(X, i, eps, metric)

        if len(neighbors) < minPts:
            clusters[i] = -1

        else:
            currentPoint += 1
            expand(X, clusters, i, neighbors, currentPoint, eps, minPts, metric)
    
    return clusters

class Basic_DBSCAN:
    """
    Parameters:
    
    eps: Radius of neighborhood graph
    
    minPts: Number of neighbors required to label a given point as a core point.
    
    metric: Distance metric used to determine distance between points; 
            currently accepts scipy.spatial.distance metrics for two numeric vectors
    
    """
    
    def __init__(self, eps, minPts, metric=distance.euclidean):
        self.eps = eps
        self.minPts = minPts
        self.metric = metric
    
    def fit_predict(self, X):
        """
        Parameters:
        
        X: An n-dimensional array of numeric vectors to be analyzed
        
        Returns:
        
        [n] cluster labels
        """
    
        clusters = [0] * X.shape[0]
        
        simple_DBSCAN(X, clusters, self.eps, self.minPts, self.metric)
        
        return clusters
    
from sklearn.datasets import make_blobs
centers = [(0, 4), (5, 5) , (8,2)]
cluster_std = [1.2, 1, 1.1]
X, y= make_blobs(n_samples=200, cluster_std=cluster_std, centers=centers, n_features=2, random_state=1)
#radius of the circle defined as 0.6
eps = 0.6
#minimum neighbouring points set to 3
minPts = 3


scanner = Basic_DBSCAN(eps=eps, minPts=minPts)

clusters = scanner.fit_predict(X)

plt.scatter(X[:,0],X[:,1],c=clusters)











