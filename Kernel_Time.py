"""
This script calculates temporal kernels based on the Matern kernel(smoothness=3/2).

Key variables:
- t_shape: shape of the temporal data
- lt: length scale
- sigma_a: kernel scale
- sigma_b: a minor term to avoid ill-conditions
"""

import autograd.numpy as np

def cal_t_kers(t, ker):
    # alpha = 1e-3 # If choose fixed-alpha-RQK
    
    # Ensure t is a 1D array
    if np.isscalar(t):
        t = np.atleast_1d(t)  # Autograd-compatible
    
    t_shape = t.shape[0]
    # train-train
    t_train = np.empty((t_shape, t_shape))
    for i, a in enumerate(t):
        for j, b in enumerate(t):
            t_train[i, j] = np.absolute(a - b)  # temporal distance (absolute difference)
    if ker == 1:    
      def ker_t_tr(lt, sigma_a, sigma_b, alpha):
          K_t = sigma_a * (1 + (t_train**2) / (2 * alpha * (lt**2)))**(-alpha) + sigma_b * np.eye(t_shape)
          K_t = K_t + 1e-6 * np.eye(K_t.shape[0])
          return K_t
    else:
        def ker_t_tr(lt, sigma_a, sigma_b):
          if ker == 0.5:
              K_t = sigma_a * np.exp(- t_train / lt) + sigma_b * np.eye(t_shape) # M-1/2
          elif ker == 1.5:
              K_t = sigma_a * np.multiply((1 + np.sqrt(3) / lt * t_train), np.exp(-np.sqrt(3) / lt * t_train)) + sigma_b * np.eye(t_shape) # M-3/2
          elif ker == 2.5:
              K_t = sigma_a * np.multiply((1 + np.sqrt(5) / lt * t_train + 5 * (t_train**2)/ (3*(lt**2))), np.exp(-np.sqrt(5) / lt * t_train)) + sigma_b * np.eye(t_shape) # M-5/2
          else:
              K_t = sigma_a * (1 + (t_train**2) / (2 * alpha * (lt**2)))**(-alpha) + sigma_b * np.eye(t_shape)
              K_t = K_t + 1e-6 * np.eye(K_t.shape[0])
          return K_t
        

    # test-train
    t_train_s = np.empty((t_shape, t_shape))
    for i, a in enumerate(t):
        for j, b in enumerate(t):
            t_train_s[i, j] = np.absolute(a - b)
    if ker == 1: 
      def ker_t_es_tr(lt, sigma_a, alpha):
          K_t = sigma_a * (1 + (t_train_s**2) / (2 * alpha * (lt**2)))**(-alpha)
          K_t = K_t + 1e-6 * np.eye(K_t.shape[0])
          return K_t.T
    else:
      def ker_t_es_tr(lt, sigma_a):
          if ker == 0.5:
              K_t = sigma_a * np.exp(- t_train_s / lt)  # M-1/2
          elif ker == 1.5:
              K_t = sigma_a * np.multiply((1 + np.sqrt(3) / lt * t_train_s), np.exp(-np.sqrt(3) / lt * t_train_s))  # M-3/2
          elif ker == 2.5:
              K_t = sigma_a * np.multiply((1 + np.sqrt(5) / lt * t_train_s + 5 * (t_train_s**2)/ (3*(lt**2))), np.exp(-np.sqrt(5) / lt * t_train_s))  # M-5/2
          else:
              K_t = sigma_a * (1 + (t_train_s**2) / (2 * alpha * (lt**2)))**(-alpha)
              K_t = K_t + 1e-6 * np.eye(K_t.shape[0])
          return K_t.T
        
    
    # test-test
    t_s = np.empty((t_shape, t_shape))
    for i, a in enumerate(t):
        for j, b in enumerate(t):
            t_s[i, j] = np.absolute(a - b)
    if ker == 1: 
      def ker_t_es(lt, sigma_a, alpha):
          K_t = sigma_a * (1 + (t_s**2) / (2 * alpha * (lt**2)))**(-alpha)
          K_t = K_t + 1e-6 * np.eye(K_t.shape[0])
          return K_t
    else:
      def ker_t_es(lt, sigma_a):
          if ker == 0.5:
              K_t = sigma_a * np.exp(- t_s / lt)  # M-1/2
          elif ker == 1.5:
              K_t = sigma_a * np.multiply((1 + np.sqrt(3) / lt * t_s), np.exp(-np.sqrt(3) / lt * t_s))  # M-3/2
          elif ker == 2.5:
              K_t = sigma_a * np.multiply((1 + np.sqrt(5) / lt * t_s + 5 * (t_s**2)/ (3*(lt**2))), np.exp(-np.sqrt(5) / lt * t_s))  # M-5/2
          else:
              K_t = sigma_a * (1 + (t_s**2) / (2 * alpha * (lt**2)))**(-alpha)
              K_t = K_t + 1e-6 * np.eye(K_t.shape[0])
          return K_t

    return ker_t_tr, ker_t_es_tr, ker_t_es