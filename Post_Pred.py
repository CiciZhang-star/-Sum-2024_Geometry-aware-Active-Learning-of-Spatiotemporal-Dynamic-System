"""
This script defines the functions to calculate the posterior mean and covariance(diagonal elements) based on GP model.

Key variables:
- Y_train_noise: training data
- pars: parameters
- K_t: temporal kernel for train-train
- K_t_s: temporal kernel for test-train
- K_t_ss: temporal kernel for test-test
- K_sp: spatial kernel for train-train
- K_sp_s: spatial kernel for test-train
- K_sp_ss: spatial kernel for test-test
- k_f: task kernel for train-train
- k_f_s: task kernel for test-train
- k_f_ss: task kernel for test-test
"""
from Block_Diag_Inv import cal_block_diag_inv
import autograd.numpy as np
from autograd.numpy.linalg import inv

class Post_Predictor:
    def __init__(self, pars, func_K_t, func_K_t_s, func_K_t_ss, func_K_sp, func_K_sp_s, func_K_sp_ss):
        self.pars = pars
        self.func_K_t = func_K_t
        self.func_K_t_s = func_K_t_s
        self.func_K_t_ss = func_K_t_ss
        self.func_K_sp = func_K_sp
        self.func_K_sp_s = func_K_sp_s
        self.func_K_sp_ss = func_K_sp_ss
        
        # Pre-compute commonly used kernels
        self._compute_kernels()

    def _compute_kernels(self):
        """Pre-compute commonly used kernels and their SVDs"""
        # Temporal kernels
        self.K_t = self.func_K_t(self.pars[3], self.pars[4], self.pars[5])
        self.U_t, self.S_t, self.V_t = np.linalg.svd(self.K_t, full_matrices=False)
        self.K_t_s = self.func_K_t_s(self.pars[3], self.pars[4])
        self.K_t_ss = self.func_K_t_ss(self.pars[3], self.pars[4])
        
        # Spatial kernels
        self.K_sp = self.func_K_sp(self.pars[0], self.pars[1], self.pars[2])
        self.U_s, self.S_s, self.V_s = np.linalg.svd(self.K_sp, full_matrices=False)
        self.K_sp_s = self.func_K_sp_s(self.pars[0], self.pars[1])
        self.K_sp_ss = self.func_K_sp_ss(self.pars[0], self.pars[1])
        
        # Noise
        self.D = np.array(self.pars[6])


    def post_mean(self, Y_train_noise):
        # Compute vector v1 and v2
        Lam_inv = cal_block_diag_inv(self.S_s, self.S_t, self.D)

        v_u = (self.U_t.T @ Y_train_noise[:,:].T @ self.U_s).flatten(order='F')

        v1 = Lam_inv['Lam_inv_11_diag'] * v_u
        V1 = v1.reshape((self.K_t_s.shape[0],self.K_sp.shape[0]),order='F')
        mat_v = V1
        mean = self.K_t_s @ self.U_t @ mat_v @ (self.K_sp_s @ self.U_s).T

        return mean

    def post_var_i(self, i, type="normal"):
        # Calculate posterior variance for index i
        if type == 'cross-val':
          # Spatial kernels
          K_sp_0 = self.K_sp
          K_sp = np.delete(np.delete(K_sp_0, i, axis=0), i, axis=1) # train-train
          U_s, S_s, V_s = np.linalg.svd(K_sp, full_matrices=False) # SVD of K_sp
          K_sp_s = np.delete(K_sp_0[i,:], i, axis=0) # test-train
          K_sp_ss = K_sp_0[i,i] # test-test
        else:
          K_sp = self.K_sp
          U_s, S_s, V_s = np.linalg.svd(K_sp, full_matrices=False) # SVD of K_sp
          K_sp_s = self.K_sp_s[i,:] # test-train
          K_sp_ss = self.K_sp_ss[i,i] # test-test

        # Calculate f1 term
        # diag_f1 = np.kron(np.kron(np.diag(self.K_f_ss), K_sp_ss), np.diag(self.K_t_ss))
        diag_f1 = np.kron(K_sp_ss, np.diag(self.K_t_ss))
        # diag_f1 = np.kron(np.diag(K_sp_ss), np.diag(self.K_t_ss))

        # Calculate f2 term
        sp_s_times_U_s = np.reshape(K_sp_s, (1, -1)) @ U_s
        t_s_times_U_t = self.K_t_s @ self.U_t
        
        # f2_components = {
        #         'a': np.kron(self.K_f_s[0,0] * sp_s_times_U_s, t_s_times_U_t),
        #         'b': np.kron(self.K_f_s[0,1] * sp_s_times_U_s, t_s_times_U_t),
        #         'c': np.kron(self.K_f_s[1,0] * sp_s_times_U_s, t_s_times_U_t),
        #         'd': np.kron(self.K_f_s[1,1] * sp_s_times_U_s, t_s_times_U_t)
        #     }

        # Lam_inv = cal_block_diag_inv(self.K_f, S_s, self.S_t, self.D)


        # diag_k1 = (
        #         np.sum(f2_components['a'] * Lam_inv['Lam_inv_11_diag'] * f2_components['a'], axis=1) +
        #         np.sum(f2_components['b'] * Lam_inv['Lam_inv_21_diag'] * f2_components['a'], axis=1) +
        #         np.sum(f2_components['a'] * Lam_inv['Lam_inv_12_diag'] * f2_components['b'], axis=1) +
        #         np.sum(f2_components['b'] * Lam_inv['Lam_inv_22_diag'] * f2_components['b'], axis=1)
        #     )
      
        # diag_k4 = (
        #         np.sum(f2_components['c'] * Lam_inv['Lam_inv_11_diag'] * f2_components['c'], axis=1) +
        #         np.sum(f2_components['d'] * Lam_inv['Lam_inv_21_diag'] * f2_components['c'], axis=1) +
        #         np.sum(f2_components['c'] * Lam_inv['Lam_inv_12_diag'] * f2_components['d'], axis=1) +
        #         np.sum(f2_components['d'] * Lam_inv['Lam_inv_22_diag'] * f2_components['d'], axis=1)
        #     )

        # diag_f2 = np.concatenate([diag_k1, diag_k4], axis=0)

        M = np.kron(sp_s_times_U_s,t_s_times_U_t)
        Lam_inv = cal_block_diag_inv(S_s, self.S_t, self.D)
        Lam_inv_11_diag = Lam_inv['Lam_inv_11_diag']
        diag_f2 = M * M @ Lam_inv_11_diag

        diag_var = diag_f1 - diag_f2

        return diag_var

    def post_var(self):
    # Calculate posterior variance for all points"""  
      # post_var = np.zeros((self.K_f_ss.shape[0]*self.K_t_ss.shape[0], self.K_sp_ss.shape[0]))
      post_var = np.zeros((self.K_t_ss.shape[0], self.K_sp_ss.shape[0]))
      for i in range(self.K_sp_ss.shape[0]):
          var_i = self.post_var_i(i, type = "normal")
          post_var[:, i] = var_i
      
      return post_var
    


    # def post_var(self):
    #     """
    #     Efficiently calculate posterior variance for all test spatial points at once, without a loop.
    #     """
    #     # SVD of spatial kernel for all test points
    #     K_sp = self.K_sp
    #     U_s, S_s, V_s = np.linalg.svd(K_sp, full_matrices=False)  # SVD of K_sp

    #     # SVD of temporal kernel (assume already available as self.U_t, self.S_t)
    #     U_t = self.U_t

    #     # Precompute Lam_inv
    #     Lam_inv = cal_block_diag_inv(S_s, self.S_t, self.D)
    #     Lam_inv_11_diag = Lam_inv['Lam_inv_11_diag']

    #     # Number of test spatial points and time points
    #     n_sp = self.K_sp_ss.shape[0]
    #     n_t = self.K_t_ss.shape[0]

    #     # diag_f1: shape (n_t, n_sp)
    #     diag_f1 = np.kron(np.diag(self.K_sp_ss), np.diag(self.K_t_ss))

    #     # Precompute sp_s_times_U_s for all test spatial points
    #     sp_s_times_U_s_all = self.K_sp_s @ U_s  # shape: (n_sp, U_s.shape[1])
    #     t_s_times_U_t = self.K_t_s @ U_t        # shape: (n_t, U_t.shape[1])

    #     # Compute M for all test spatial points and all time points
    #     # For each i in n_sp, for each j in n_t: M[i,j,:] = np.kron(sp_s_times_U_s_all[i], t_s_times_U_t[j])
    #     # Resulting shape: (n_sp, n_t, U_s.shape[1]*U_t.shape[1])
    #     M = np.einsum('ik,jl->ijlk', sp_s_times_U_s_all, t_s_times_U_t).reshape(n_sp, n_t, -1)

    #     # Now, for each (i,j), compute diag_f2 = (M[i,j,:] ** 2) @ Lam_inv_11_diag
    #     diag_f2 = np.tensordot(M**2, Lam_inv_11_diag, axes=([2],[0]))  # shape: (n_sp, n_t)
    #     diag_f2 = diag_f2.T  # shape: (n_t, n_sp) to match diag_f1

    #     # M = np.kron(sp_s_times_U_s_all, t_s_times_U_t)
    #     # diag_f2 = M * M @ Lam_inv_11_diag

    #     # Posterior variance
    #     post_var = diag_f1 - diag_f2  # shape: (n_t, n_sp)
    #     return post_var