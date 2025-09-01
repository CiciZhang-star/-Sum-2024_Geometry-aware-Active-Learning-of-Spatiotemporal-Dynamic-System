"""
This script calculates spatial kernels based on the Matern kernel(smoothness=3/2) in Euclidean space(Euclidean-distance).

Key variables:
- idx: indices of the training points
- inc: coordinates of all points
- Tri: indices of the triangular mesh vertices
- ls: length scale
- sigma_m: kernel scale
- sigma_n: a minor term to avoid ill-conditions
"""

import numpy as np
from sklearn.metrics import pairwise_distances

def calculate_spatial_kernels_euclidean(idx, inc, Tri):
    # Normalize the coordinates of spatial points
    av_edge = inc[Tri[:, 0:2]]
    av_edge_length = np.linalg.norm(av_edge[:, 1, :] - av_edge[:, 0, :], axis=1).mean() # scale the mesh to make the average edge length is 1.
    X = (inc - inc.mean(axis=0)) / av_edge_length

    eucl_dist_matrix = pairwise_distances(X, X)   
    
    # train-train
    dist_train = np.empty((len(idx), len(idx)))
    for i, a in enumerate(idx): 
        for j, b in enumerate(idx):
            dist_train[i, j] = eucl_dist_matrix[a, b]
    def kernel_space_train(ls, sigma_m, sigma_n):
        ##### Matern kernel ######
        K_sp = sigma_m*np.multiply( (1+np.sqrt(3)/ls*dist_train), np.exp(-np.sqrt(3)/ls*dist_train) ) + sigma_n * np.eye(len(idx))
        return K_sp

    # test-train
    dist_train_s = np.empty((len(idx), len(inc) - len(idx)))
    for i, a in enumerate(idx):
        for j, b in enumerate(np.setdiff1d(np.arange(len(inc)), idx)):
            dist_train_s[i, j] = eucl_dist_matrix[a, b]          
    def kernel_space_s(ls, sigma_m):
        ##### Matern kernel ######    
        K_sp_s = sigma_m* np.multiply((1+np.sqrt(3)/ls*dist_train_s), np.exp(-np.sqrt(3)/ls*dist_train_s) ) # train-test
        return K_sp_s.T # test-train
    
    # test-test
    remaining_shape = len(inc) - len(idx)
    dist_s = np.empty((remaining_shape, remaining_shape))
    remaining_indices = np.setdiff1d(np.arange(len(inc)), idx)
    for i, a in enumerate(remaining_indices):
        for j, b in enumerate(remaining_indices):
            dist_s[i, j] = eucl_dist_matrix[a, b]
                          
    def kernel_space_ss(ls, sigma_m):
        ##### Matern kernel ######
        K_sp_ss = sigma_m* np.multiply((1+np.sqrt(3)/ls*dist_s), np.exp(-np.sqrt(3)/ls*dist_s) )
        return K_sp_ss

    return kernel_space_train, kernel_space_s, kernel_space_ss