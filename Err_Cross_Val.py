from Block_Diag_Inv import cal_block_diag_inv
from Post_Pred import Post_Predictor

import autograd.numpy as np


def cal_val_err(pars, Y_train_noise, func_K_t, func_K_t_s, func_K_t_ss, func_K_sp, func_K_sp_s, func_K_sp_ss):
    # Spatial kernel for train-train
    K_sp = func_K_sp(pars[0], pars[1], pars[2])
    U_s, S_s, V_s = np.linalg.svd(K_sp, full_matrices=False) # SVD of K_sp

    # Temporal kernel for train-train
    K_t = func_K_t(pars[3], pars[4], pars[5])
    U_t, S_t, V_t = np.linalg.svd(K_t, full_matrices=False) # SVD of K_t

    # # Task kernel
    # K_f = np.array([[pars[6], pars[7]], [pars[7], pars[8]]])
    # K_f = np.dot(K_f.T, K_f)

    # K_f noise
    # D = np.diag(np.array([pars[9], pars[10]]))
    D = np.array([pars[6]])

    dy = []

    Predictor = Post_Predictor(
    pars=pars,
    func_K_t=func_K_t,
    func_K_t_s=func_K_t_s,
    func_K_t_ss=func_K_t_ss,
    func_K_sp=func_K_sp,
    func_K_sp_s=func_K_sp_s,
    func_K_sp_ss=func_K_sp_ss
)
    
    for i in range(Y_train_noise.shape[0]):
      # Calculate f1 term
      diag_f1 = Predictor.post_var_i(i, type='cross-val')

      # Calculate f2 term
      # vec_u = (U_t.T @ Y_train_noise[0,:,:].T @ U_s).flatten(order='F')
      # vec_v = (U_t.T @ Y_train_noise[1,:,:].T @ U_s).flatten(order='F')

      # Lam_inv = cal_block_diag_inv(K_f, S_s, S_t, D)
      # Lam_inv_11_diag = Lam_inv['Lam_inv_11_diag']
      # Lam_inv_12_diag = Lam_inv['Lam_inv_12_diag']
      # Lam_inv_21_diag = Lam_inv['Lam_inv_21_diag']
      # Lam_inv_22_diag = Lam_inv['Lam_inv_22_diag']

      # mat_11_u = (Lam_inv_11_diag * vec_u).reshape(U_t.shape[1],-1,order='F')
      # mat_12_v = (Lam_inv_12_diag * vec_v).reshape(U_t.shape[1],-1,order='F')
      # mat_21_u = (Lam_inv_21_diag * vec_u).reshape(U_t.shape[1],-1,order='F')
      # mat_22_v = (Lam_inv_22_diag * vec_v).reshape(U_t.shape[1],-1,order='F')

      # f2_1 = ((U_t @ mat_11_u @ U_s.T) + (U_t @ mat_12_v @ U_s.T))[:,i]
      # f2_2 = ((U_t @ mat_21_u @ U_s.T) + (U_t @ mat_22_v @ U_s.T))[:,i]
      # f2 = np.concatenate([f2_1, f2_2], axis=0)

      # f2_1 = ((U_t @ mat_11_u @ U_s.T) + (U_t @ mat_12_v @ U_s.T))[:,i]
      # f2_2 = ((U_t @ mat_11_u @ U_s.T) + (U_t @ mat_12_v @ U_s.T))[:,i+1]
      # f2_m2 = np.concatenate([f2_1, f2_2], axis=0)

      Lam_inv = cal_block_diag_inv(S_s, S_t, D)

      v_u = (U_t.T @ Y_train_noise[:,:].T @ U_s).flatten(order='F')

      v1 = Lam_inv['Lam_inv_11_diag'] * v_u
      V1 = v1.reshape((K_t.shape[0], K_sp.shape[0]),order='F')
      mat_v = V1
      f2 = (U_t @ mat_v @ U_s.T)[:,i]

      loocv = diag_f1 * f2
      
      # dy.append(np.mean(np.abs(np.diag(diag_f1)@f2))) # ??? np.mean()

      ######Leave-One-Out Cross-Validation Negative Log Predictive Density (LOOCV NLPD)#####
      # tem1 = np.diag(diag_f1)@f2
      dy_val = 1/2 * (loocv * loocv) / diag_f1 + 1/2 * np.log(diag_f1) + 1/2 * np.log(2*np.pi)
      dy_val_si = np.mean(dy_val) # Average all temporal steps in the i-th spatial location
      dy.append(dy_val_si)

    dy = np.array(dy)
    # tau_squ = np.sum(dy ** 2) / (K_sp.shape[0] * K_t.shape[0] )
    tau_squ = np.sum(dy) / K_sp.shape[0] # Average all training spatial locations
    
    return tau_squ
