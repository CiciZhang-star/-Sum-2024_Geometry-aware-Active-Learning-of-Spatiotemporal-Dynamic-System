"""
This script defines the functions to calculate the negative log-likelihood (NLL).
"""
from Block_Diag_Inv import cal_block_diag_inv

# import autograd.numpy as np
from autograd import numpy as np
# from scipy.optimize import minimize
# from autograd.numpy.linalg import inv
# import numpy

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0) # check if all eigenvalues are positive

def nll_func(guess, Y_train_noise, func_K_t, func_K_sp):
    pars = guess
    # Temporal kernel
    K_t = func_K_t(lt=pars[3], sigma_a=pars[4], sigma_b=pars[5])


    try:
        np.linalg.cholesky(K_t)
    except np.linalg.LinAlgError:
        print("Warning: K_t is not positive definite! Regularizing further.")
    
    
    U_t, S_t, V_t = np.linalg.svd(K_t, full_matrices=False) # SVD of K_t
    

    # Spatial kernel
    K_sp = func_K_sp(ls=pars[0], sigma_m=pars[1], sigma_n=pars[2])
    U_s, S_s, V_s = np.linalg.svd(K_sp, full_matrices=False) # SVD of K_sp

    # Noise
    D = np.array(pars[6])

    # Calculate f1 term
    Lam_inv = cal_block_diag_inv(S_s, S_t, D)
    Lam_inv_11_diag = Lam_inv['Lam_inv_11_diag']

    v1 = (U_t.T @ Y_train_noise[:,:].T @ U_s).flatten(order='F')

    f1 = (v1 * (Lam_inv_11_diag * v1)).sum() 
    
    # Calculate f2 term
    Lam_11_diag = Lam_inv['Lam_11_diag']
    f2 = np.sum(np.log(Lam_11_diag)) 

    # Calculate f3 term
    f3 = K_sp.shape[0] * K_t.shape[0] * np.log(2 * np.pi)

    # Calculate the negative log-likelihood
    nll = 0.5 * (f1 + f2 + f3)

    if not np.isfinite(nll):
        return np.inf
    
    print(f"nll: {nll}, nll/N: {nll/(Y_train_noise.shape[0]* Y_train_noise.shape[1])}")
    
    return nll