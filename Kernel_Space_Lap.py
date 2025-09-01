"""
This script defines the SpatialKernelModel class, which calculates spatial kernels based on Laplacian eigenpairs.

Key variables:
- vertex: indices of vertices for training
- Q: selected eigenvalues of Laplacian matrix
- V: selected eigenvectors matched with eigenvalues
- kernel_type: type of kernel ('Matern' or 'RBF')
- smoothness: required for Matern kernel
- ls/rho: length scale
- sigma_m: kernel scale
- sigma_n: a minor term to avoid ill-conditions
"""
import autograd.numpy as np
from scipy import linalg
from scipy.optimize import minimize
from scipy.special import gamma

class Sp_K_Model:
    def __init__(self, vertex, Q, V, kernel_type, smoothness):
        self.vertex = vertex  # selected indices of vertices for training
        self.Q = Q  # shape = (# of selected eigenvalues,); selected eigenvalues of Laplacian matrix (N_vertices, N_vertices)
        self.V = V  # shape = (N_vertices, # of selected eigenvalues); selected eigenvectors matched with eigenvalues
        self.kernel_type = kernel_type  # 'Matern' or 'RBF'
        self.smoothness = smoothness  # required for Matern kernel
        self.nugget = None

    def spectral_density(self, rho):
        '''rho: length-scale'''
        if self.kernel_type == 'Matern':
            D = 2  # dimension, 2D manifold in 3D space
            const1 = (2**D * np.pi**(D/2) * gamma(self.smoothness + (D/2)) * (2*self.smoothness)**self.smoothness) / gamma(self.smoothness)
            const2 = 4*(np.pi**2)*(self.Q)
            const3 = -(self.smoothness + (D/2))
            beta = 2*self.smoothness / (rho**2) + const2
            return (const1/rho**(2*self.smoothness)) * beta**const3
        
        elif self.kernel_type == 'RBF':
            return (2.0 * np.pi * rho**2) * np.exp((-2 * np.pi**2 * rho**2) * self.Q)
        else:
            raise ValueError("Invalid kernel type. Supported types are 'Matern' and 'RBF'.")


    def SD_func(self, ls, sigma_m):
        result = sigma_m * self.spectral_density(ls,)
        return result
    
    # train-train
    def ker_sp(self, ls, sigma_m, sigma_n):
        SD = self.SD_func(ls, sigma_m)
        V = (np.sqrt(SD) * self.V)[self.vertex, :]
        K_s = np.dot(V,V.T) + sigma_n * np.eye((np.dot(V,V.T)).shape[0]) # shape: (train-size, train-size)
        # K_s = V.dot(V.T) + np.random.normal(0, sigma_n, size=(V.dot(V.T)).shape)
        return K_s

    # test-train
    def ker_sp_es_tr(self, ls, sigma_m):
        SD = self.SD_func(ls, sigma_m)
        tem = np.sqrt(SD) * self.V
        mask = np.ones(tem.shape[0], dtype=bool)
        mask[self.vertex] = False
        V = tem[self.vertex, :]
        V_others = tem[mask, :]
        K_s = np.dot(V, V_others.T) # shape: (train-size, test-size)
        return K_s.T

    # test-test
    def ker_sp_ss(self, ls, sigma_m):
        SD = self.SD_func(ls, sigma_m)
        # (np.sqrt(SD)*model.V) exclude [self.vertex,:]
        tem = np.sqrt(SD) * self.V
        mask = np.ones(tem.shape[0], dtype=bool)
        mask[self.vertex] = False
        V = tem[mask, :]
        K_s = np.dot(V,V.T) # shape: (test-size, test-size)
        return K_s
    
def cal_sp_kers(vertex, Q, V, kernel_type, smoothness):
    model = Sp_K_Model(vertex, Q, V, kernel_type, smoothness)
    return model.ker_sp, model.ker_sp_es_tr, model.ker_sp_ss