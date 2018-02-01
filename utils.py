#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 14:01:09 2018

@author: 
"""
import numpy as np 

def Kmeans(data, num_clusters, latent_dim, variance_level=None):
    """
    data : dataset
    num_clusters : number of clusters
    latent_dim : dimension of the latent space
    variance_level

    pi : proportions of clusters
    mus : centers of the clusters in the observation space
    W : latent to observation matricies
    sigma2 : noise
    """

    N, d = data.shape

    # initialization
    init_centers = np.random.choice(range(N), num_clusters,replace=False)
    
    mus = data[init_centers, :]
    distances = np.zeros((N, num_clusters))
    clusters = np.zeros(N).astype(int)

    D_old = -2
    D = -1

    while(D_old != D):
        D_old = D

        # assign clusters
        for c in range(num_clusters):
            distances[:, c] = np.sum((data - mus[c, :])**2, axis=1)
        clusters = np.argmin(distances, axis=1)

        # compute distortion
        min_distances = distances[range(N), clusters]
        D = min_distances.sum()
        # compute new centers
        for c in range(num_clusters):
            mus[c, :] = data[clusters == c, :].mean(0)


    # parameter initialization
    pi = np.zeros(num_clusters)
    W = np.zeros((num_clusters, d, latent_dim))
    sigma2 = np.zeros(num_clusters)
    for c in range(num_clusters):
        #if variance_level:
        #    W[c, :, :] = variance_level * np.random.randn(d, latent_dim)
        #    sigma2[c] = np.abs((variance_level/10) * np.random.randn())
        #else:
        W[c, :, :] = np.random.randn(d, latent_dim)
        sigma2[c] = (min_distances[clusters == c]).mean() / d
        pi[c] = (clusters == c).sum() / N
            

    return pi, mus, sigma2, clusters, W 
