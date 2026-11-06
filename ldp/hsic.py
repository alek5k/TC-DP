import torch

def rbf_kernel(X, sigma=None):
    """ Compute RBF kernel matrix for X """
    pairwise_sq_dists = torch.cdist(X, X, p=2) ** 2
    if sigma is None:
        sigma = torch.median(pairwise_sq_dists)  # Median heuristic
    K = torch.exp(-pairwise_sq_dists / (2 * sigma ** 2))
    return K

def center_kernel(K):
    """ Center the kernel matrix K """
    n = K.shape[0]
    H = torch.eye(n, device=K.device) - (1.0 / n) * torch.ones((n, n), device=K.device)
    return H @ K @ H

def hsic(X, Y):
    """ Compute HSIC between X and Y """
    Kx = center_kernel(rbf_kernel(X))
    Ky = center_kernel(rbf_kernel(Y))
    return torch.trace(Kx @ Ky) / (X.shape[0] - 1) ** 2

def batch_hsic(actions):
    """ Compute HSIC for each batch """
    B, N, D = actions.shape  # B=batch, N=sequence length, D=7
    hsic_values = torch.zeros(B, device=actions.device)
    
    for b in range(B):
        X = actions[b]  # Select the batch element
        Y = X  # Can also compare with another variable if needed
        hsic_values[b] = hsic(X, Y)

    return hsic_values  # Shape: (B,)


if __name__ == "__main__":
    # Example usage
    B, N, D = 8, 50, 7  # Example batch size, sequence length, action dimensions
    actions = torch.randn(B, N, D)  # Random action tensor
    hsic_values = batch_hsic(actions)
    print(hsic_values)  # Tensor of HSIC values for each batch
