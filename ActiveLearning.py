"""
This script implements the active learning process using combined criterion.

Key variables:
- reps: number of repetitions
- idx: indices of initial training data
- inc: all spatial coordinates
- Y_noise: all Y values (train + test) with noise
- t: temporal points
- cov_diag: diagonal elements of the covariance matrix
- Q: eigenvalues of Laplacian matrix
- V: eigenvectors of Laplacian matrix
- par: parameters
- K_sp: spatial kernel for train-train
- K_sp_s: spatial kernel for test-train
- K_sp_ss: spatial kernel for test-test
- K_t: temporal kernel for train-train
- K_t_s: temporal kernel for test-train
- K_t_ss: temporal kernel for test-test
- re0: initial RE
- mse0: initial MSE
- Y_train_noise: training data with noise
- noise: noise in training data
- type: variance-only (var); 
"""

import numpy as np
from sklearn import manifold
from Err_Cross_Val import cal_val_err
import time  # Added for timing

def active_learning(reps, idx, inc, Y_noise, t, cov_diag, Q, V, par, K_sp, K_sp_s, K_sp_ss, K_t, K_t_s, K_t_ss, re0, Y_train_noise, k, Tri, type, alpha, alpha1_fix):
    # start_time = time.time()  # Start timing

    idx_new = idx
    K_sp_new = K_sp
    K_sp_s_new = K_sp_s
    K_sp_ss_new = K_sp_ss

    # Calculate the geodesic distance matrix using Isomap
    # dist_matrix = manifold.Isomap(
    #     n_neighbors=6, # Number of neighbors to consider for each point
    #     n_components=2, # Specifies the dimensionality of the output embedding(2D)
    #     max_iter=5000 # Maximum number of iterations for calculating distance from high-dimensional space to low-dimensional space
    # ).fit(inc).dist_matrix_
    # .fit(inc): Fits the Isomap model to the input data 'inc'
    # .dist_matrix_: Retrieves the geodesic distance matrix after fittings
    dist_matrix = manifold.Isomap(n_neighbors=6, n_components=2, max_iter=5000).fit(inc).dist_matrix_
    
    # distances_to_training = dist_matrix[remaining_indices][:, idx_new]
    # min_distances = np.min(distances_to_training, axis=1)
    # min_indices = remaining_indices

    # Prepare the output file name based on the selection type
    file_name = f"Res_ActiveLearning_{type}_alpha1{alpha1_fix}.txt"

    for rep in range(reps):
        rem = np.ones(Y_noise.shape[0], dtype=bool)
        rem[idx_new] = False
        remaining_indices = np.where(rem)[0]

        if rep == 0:
            with open(file_name, 'a') as f:
                f.write(f"Rep {0}: RE = {re0:.6f}\n")
            print(f"Rep: {0} -------------------------------")
            print(f"RE = {re0:.6f}")
            print(f"Number of training spatial points: {len(idx)}")
        else:
            None
        
        print(f"Rep: {rep+1} -------------------------------")
        # Variance
        group_cov_diag = cov_diag.T # Dim: (n_s_train, n_t)
        group_ave = np.mean(group_cov_diag, axis=1) # Dim: (n_s_train, )
        # Geodesic-dist matrix has been defined before the loop

        # Criterion: Random
        if type == 'ran':
          random_k_indices = np.random.choice(np.where(rem)[0], size=k, replace=False)
          print(f"Randomly selected k indices: {random_k_indices}")
          # Add new points to idx_new
          new_points = random_k_indices
          idx_new = np.concatenate((idx_new, random_k_indices))

        # Criterion: Variance-only
        elif type == 'var':
          top_k_group_id = np.argsort(group_ave)[-k:]
          print(f"Top values of variance: {group_ave[top_k_group_id]}")
          # Match top_k_group_id with the initial indices in whole data
          initial_row_indices = np.where(rem)[0]
          new_points = initial_row_indices[top_k_group_id]
          print(f"Top k initial indices in whole data: {new_points}")
          # Add new points to idx_new
          idx_new = np.concatenate((idx_new, new_points))


        # Criterion: Geodesic distance space filling design
        elif type == 'geo-dist':
          # Find the minimum distance for each remaining point
          distances_to_training = dist_matrix[remaining_indices][:, idx_new]
          min_distances = np.min(distances_to_training, axis=1)
          min_indices = remaining_indices         
          
          # Find the maximum k distances in min_distances
          top_k_indices = np.argsort(min_distances)[-k:]
          new_points = [min_indices[i] for i in top_k_indices]
          print(f"New points to add: {new_points}")       
          # Add new points to idx_new
          idx_new = np.concatenate((idx_new, new_points))


        # Criterion: combine var and geo-dist within fixed weights
        elif type == 'com-fix':
          # Variance term
          var_term = group_ave / np.max(group_ave)
          # Geodesic-distance term
          # Find the minimum distance for each remaining point
          distances_to_training = dist_matrix[remaining_indices][:, idx_new]
          min_distances = np.min(distances_to_training, axis=1)
          min_indices = remaining_indices
          geod_term = min_distances / np.max(min_distances)
          # Combined criterion = alpha1 * geod_term + alpha2 * var_term      
          # alpha1 = 0.5
          alpha1 = alpha1_fix
          print(f"alpha1 = {alpha1:.6f}")
          alpha2 = 1 - alpha1        
          # Combine geodesic and variance criterion:
          # - When tau_squ is larger, GP is less confident, so we rely more on geodesic distance.
          # - When tau_squ is smaller, GP is more confident, so we consider more of the variance (GP prediction results).
          combined_term = alpha1 * geod_term + alpha2 * var_term
          max_k_combined_indices = np.argsort(combined_term)[-k:]
          new_points = [min_indices[i] for i in max_k_combined_indices]
          print(f"New points to add: {new_points}")          
          # Add new points to idx_new
          idx_new = np.concatenate((idx_new, new_points))


        # Criterion: combine var and geo-dist within adaptive weights
        elif type == 'com':
          # Variance term
          var_term = group_ave / np.max(group_ave)
          # Geodesic-distance term
          # Find the minimum distance for each remaining point
          distances_to_training = dist_matrix[remaining_indices][:, idx_new]
          min_distances = np.min(distances_to_training, axis=1)
          min_indices = remaining_indices
          geod_term = min_distances / np.max(min_distances)
          # Combined criterion = alpha1 * geod_term + alpha2 * var_term
          nlpd = cal_val_err(par, Y_train_noise, K_t, K_t_s, K_t_ss, K_sp_new, K_sp_s_new, K_sp_ss_new)
          sigma = par[6]
          alpha1 = nlpd/(nlpd + sigma*alpha) # alpha1 is sensitive to sigma
          print(f"alpha1 = {alpha1:.6f}")
          alpha2 = 1 - alpha1       
          # Combine geodesic and variance criterion:
          # - When tau_squ is larger, GP is less confident, so we rely more on geodesic distance.
          # - When tau_squ is smaller, GP is more confident, so we consider more of the variance (GP prediction results).
          combined_term = alpha1 * geod_term + alpha2 * var_term
          max_k_combined_indices = np.argsort(combined_term)[-k:]
          new_points = [min_indices[i] for i in max_k_combined_indices]
          print(f"New points to add: {new_points}")          
          # Add new points to idx_new
          idx_new = np.concatenate((idx_new, new_points))


        # Prepare new training data
        Y_new = Y_noise[new_points,:]
        Y_train_noise = np.concatenate((Y_train_noise, Y_new), axis=0)

        # Prepare new test data
        rem = np.ones(Y_noise.shape[0], dtype=bool)
        rem[idx_new] = False
        Y_test = Y_noise[rem,:]

        # Calculate new spatial kernels
        from Kernel_Space_Lap import cal_sp_kers
        K_sp_new, K_sp_s_new, K_sp_ss_new = cal_sp_kers(vertex=idx_new, Q=Q, V=V, kernel_type='Matern', smoothness=3./2.)
        
        # Re-optimize parameters
        starts = 1
        bounds = ((0.5, 1.5), (800, 900), (0.0005, 0.0005), (50, 60), (1, 1), (0.01, 0.01), (0.0005, 2e-2))
        starts = 1
        np.random.seed(1234)
        guesses = np.vstack([np.random.uniform(bound[0], bound[1], size=starts) for bound in bounds]).T

        from Pars_MulStarts import Pars_Optimizer
        Optimizer = Pars_Optimizer(
            Y_train_noise, Y_test, 
            K_t, K_t_s, K_t_ss,
            K_sp_new, K_sp_s_new, K_sp_ss_new,
            inc, Tri, rem, t
        ) 

        best_pars, best_err = None, float('inf')
        for i, guess in enumerate(guesses):
            print(f"\nOptimization run {i+1}:")
            print(f"Initial guess {i+1}: {np.array2string(guess, separator=', ')}")
            pars, iter_count = Optimizer.opt_pars(guess, bounds)
            print(f"Optimized parameters: {np.array2string(pars, separator=', ')}")
            # val_err = cal_val_err(pars, Y_train_noise, K_t, K_t_s, K_t_ss, K_sp_new, K_sp_s_new, K_sp_ss_new)
            val_err = 0 # NOTE: for quick testing
            if val_err < best_err:
                best_err, best_pars = val_err, pars

        print("\nBest optimization result:")
        print(f"Best parameters: {np.array2string(best_pars, separator=', ')}")
        par = best_pars

        # New posterior prediction
        from Post_Pred import Post_Predictor
        Predictor = Post_Predictor(
                        pars=par,
                        func_K_t=K_t,
                        func_K_t_s=K_t_s,
                        func_K_t_ss=K_t_ss,
                        func_K_sp=K_sp_new,
                        func_K_sp_s=K_sp_s_new,
                        func_K_sp_ss=K_sp_ss_new,
                    )

        hsp_pred = Predictor.post_mean(Y_train_noise)
        cov_diag = Predictor.post_var()
        hsp_true = Y_noise
        hsp_test = Y_noise.copy()
        hsp_test[rem, :] = hsp_pred.T
        re = np.linalg.norm(hsp_true - hsp_test, 'fro') / np.linalg.norm(hsp_true, 'fro')
        print(f"RE = {re:.6f}")

        # Store re and mse values in a txt file
        with open(file_name, 'a') as f:
            f.write(f"Rep {rep+1}: RE = {re:.6f}; New points to add: {new_points}\n")

        print(f"Number of training spatial points: {len(idx_new)}")

    # total_time = time.time() - start_time  # End timing
    # print(f"Total running time: {total_time:.2f} seconds")
    # # Save total time to a file
    # with open(file_name, 'a') as f:
    #     f.write(f"Total running time: {total_time:.2f} seconds\n")

    return idx_new, hsp_pred, cov_diag