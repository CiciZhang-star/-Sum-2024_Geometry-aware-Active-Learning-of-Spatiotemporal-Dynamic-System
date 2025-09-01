import autograd.numpy as np

def cal_block_diag_inv(S_s, S_t, D):
    """
    Compute the inverse of block diagonal matrix Sigma = K_s ⊗ K_t + D ⊗ I_s ⊗ I_t = U @ Lam @ U.T; Sigma^-1 = U.T @ Lam^-1 @ U.
    
    Parameters:
    S_s: array - spatial eigenvalues
    S_t: array - temporal eigenvalues
    D: scalar - noise
    
    Returns:
    Dictionary containing diagonal elements of inverse blocks
    """
    # Calculate diagonal elements of each block
    Lam_11_diag = np.kron(S_s, S_t) + D * np.ones(S_s.shape[0] * S_t.shape[0],)

    # Compute inverse blocks
    Lam_inv_11_diag = 1.0/Lam_11_diag
    
    return {
        'Lam_11_diag': Lam_11_diag,
        'Lam_inv_11_diag': Lam_inv_11_diag
    }