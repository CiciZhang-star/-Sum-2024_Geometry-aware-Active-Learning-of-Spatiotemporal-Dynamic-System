"""
This script defines the functions to optimize parameters with multiple starts.

Key variables:
- Parameters:[ls, sigma_m, sigma_n, lt, sigma_a, sigma_b, D]
  - ls: length scale for spatial kernel
  - sigma_m: kernel scale for spatial kernel
  - sigma_n: noise scale for spatial kernel
  - lt: length scale for temporal kernel
  - sigma_a: kernel scale for temporal kernel
  - sigma_b: noise scale for temporal kernel
  - D: data noise
"""

# from Pars_Physics import physics_model
from Pars_NLL import nll_func
# from Err_Cross_Val import cal_val_err
# from Eigen_Lap import lap_mat
# import autograd.numpy as np
# import numpy
from scipy.optimize import minimize
# from scipy.optimize import basinhopping
# import re
from autograd import grad
from autograd import value_and_grad
# import scipy.io
# from scipy.sparse import csr_matrix

class Pars_Optimizer:
  
    def __init__(self, Y_train_noise, Y_test_noise, func_K_t, func_K_t_s, func_K_t_ss, 
                 func_K_sp, func_K_sp_s, func_K_sp_ss, X, Tri, mask, t):
        self.type = type
        self.Y_train_noise = Y_train_noise
        self.Y_test_noise = Y_test_noise
        self.func_K_t = func_K_t
        self.func_K_t_s = func_K_t_s
        self.func_K_t_ss = func_K_t_ss
        self.func_K_sp = func_K_sp
        self.func_K_sp_s = func_K_sp_s
        self.func_K_sp_ss = func_K_sp_ss
        self.X = X
        self.Tri = Tri
        self.mask = mask
        self.t = t
        
        
        self.iter_num = 0

    def opt_func(self, guess):
        
        nll = nll_func(guess, self.Y_train_noise, self.func_K_t, self.func_K_sp)
      
        func = nll/(self.Y_train_noise.shape[0]*self.Y_train_noise.shape[1])

        # Convert guess to a plain NumPy array if it's an ArrayBox
        if hasattr(guess, 'value'):
            guess_values = guess.value
        else:
            guess_values = guess
        
        print("opt value:", func)
        self.iter_count += 1

        return func.value if hasattr(func, 'value') else func #  Ensure the function returns a plain NumPy array

    def opt_pars(self, guess, bounds):
        grad_opt_func = grad(lambda x: self.opt_func(x))
        grad_iter_num = 0
        
        self.iter_num = 0

        func_and_grad = value_and_grad(self.opt_func)

        # Callback function to track variables at bounds
        def track_bounds(xk):
            for i, (value, (lower, upper)) in enumerate(zip(xk, bounds)):
                if value == lower or value == upper:
                    print(f"Variable {i} is at its bound: {value}")

        self.iter_count = 0  # Initialize iteration counter

        res = minimize(
            lambda x: self.opt_func(x),
            # func_and_grad,
            guess,
            method='TNC',  # options: 'L-BFGS-B','Nelder-Mead', 'SLSQP', 'trust-constr', 'BFGS', 'TNC'
            bounds=bounds,
            # jac=wrapped_grad,
            jac=False, # True for func_and_grad
            callback=track_bounds,
            options={
                'gtol': 1e-6,
                'ftol': 1e-6,
                # 'maxiter': 10000,
                # 'maxfun': 15000,
                'disp': True
            }
        )
        print(f"Total optimization iterations: {self.iter_count}")
            
        return res.x, self.iter_count