from typing import Callable
import torch

def black_box_naive(
    z: torch.Tensor,
    theta: torch.Tensor,
    eta: float = 0.1,
    proj_theta: Callable = lambda x: x,
    delta: float = 100.0,
    epsilon: float = 0.5,
    norm = 'linf', # 'l2' or 'linf',
    **kwargs
):
    """
    Naive Mean Shift Black Box Adversary
    
    Generic version that works for both 1D and 2D (or any dimension).
    Simple approach: shift the mean of the data samples in the direction of the mean.
    Only uses input data samples without history of data.

    Note: Some arguments are not used in this implementation, but kept for compatibility with RGD() function.
    """
    n_samples = z.shape[1]
    n_poison = int(epsilon * n_samples)
    
    if n_poison == 0:
        return z
    
    poisoned_samples = z.clone()

    # Calculate current sample mean and use it as a global shift direction
    current_mean = z.mean(dim=1, keepdim=True)  # (d, 1)
    u = current_mean.clone()
    
    # Normalize based on norm parameter
    if norm == 'l2':
        shift_dir = u / (u.norm() + 1e-12)  # L2 norm
    elif norm == 'linf':
        shift_dir = u / (torch.max(torch.abs(u)) + 1e-12)  # L-infinity norm
    else:
        raise ValueError(f"norm must be 'l2' or 'linf', got {norm}")

    # Pick any epsilon fraction of samples (selection does not affect mean shift magnitude)
    poison_indices = torch.randperm(n_samples)[:n_poison]

    # Shift all selected samples in the opposite direction by delta
    shift = delta * shift_dir.squeeze()  # (d,)
    for i in poison_indices:
        poisoned_samples[:, i] = z[:, i] + shift
    
    return poisoned_samples

def black_box_orthogonal_2d(
    z: torch.Tensor,
    theta: torch.Tensor,
    eta: float = 0.1,
    proj_theta: Callable = lambda x: x,
    delta: float = 100.0,
    epsilon: float = 0.5,
    norm = 'linf', # 'l2' or 'linf',
    **kwargs
):
    """
    Orthogonal Mean Shift Black Box Adversary
    
    Generic version that works for d == 2.
    Approach: shift the mean of the data samples orthogonal to the direction of the mean from origin.
    Only uses input data samples without history of data.

    Note: Some arguments are not used in this implementation, but kept for compatibility with RGD() function.
    """
    assert z.shape[0] == 2, "z must have at least 2 dimensions"
        
    n_samples = z.shape[1]
    n_poison = int(epsilon * n_samples)
    
    if n_poison == 0:
        return z
    
    poisoned_samples = z.clone()

    # Calculate current sample mean and use it as a global shift direction
    current_mean = z.mean(dim=1, keepdim=True)  # (d, 1)
    u = current_mean.clone()
    orthogonal_direction = torch.tensor([-u[1], u[0]], device=u.device, dtype=u.dtype)
    shift_dir = orthogonal_direction / (orthogonal_direction.norm() + 1e-12)

    # Pick any epsilon fraction of samples (selection does not affect mean shift magnitude)
    poison_indices = torch.randperm(n_samples)[:n_poison]

    # Shift all selected samples in the same direction by delta
    shift = delta * shift_dir.squeeze()  # (d,)
    for i in poison_indices:
        poisoned_samples[:, i] = z[:, i] + shift
    
    return poisoned_samples


def black_box_orthogonal(
    z: torch.Tensor,
    theta: torch.Tensor,
    eta: float = 0.1,
    proj_theta: Callable = lambda x: x,
    delta: float = 100.0,
    epsilon: float = 0.5,
    norm = 'linf', # 'l2' or 'linf',
    **kwargs
):
    """
    Orthogonal Mean Shift Black Box Adversary (N-dimensional)
    Gram-Schmidt orthogonalization: Finds random direction orthogonal to the mean vector
    Works for any dimension d >= 2
    """
    d = z.shape[0]
    n_samples = z.shape[1]
    n_poison = int(epsilon * n_samples)
    
    if n_poison == 0:
        return z
    
    poisoned_samples = z.clone()

    # Calculate mean vector u
    current_mean = z.mean(dim=1)  # (d,)
    u = current_mean.clone()
    
    # Generate a random vector v
    v = torch.randn_like(u)
    
    # Make v orthogonal to u using projection: v_orth = v - proj_u(v)
    # proj_u(v) = (v . u / u . u) * u
    u_norm_sq = torch.dot(u, u)
    if u_norm_sq < 1e-9:
        # If mean is zero, any direction is orthogonal. Use v as is.
        orthogonal_direction = v
    else:
        proj = (torch.dot(v, u) / u_norm_sq) * u
        orthogonal_direction = v - proj
        
    # Normalize direction
    if norm == 'l2':
        shift_dir = orthogonal_direction / (orthogonal_direction.norm() + 1e-12)
    elif norm == 'linf':
        shift_dir = orthogonal_direction / (torch.max(torch.abs(orthogonal_direction)) + 1e-12)
    else:
        raise ValueError(f"norm must be 'l2' or 'linf', got {norm}")

    # Pick random samples to poison
    poison_indices = torch.randperm(n_samples)[:n_poison]

    # Shift selected samples
    shift = delta * shift_dir  # (d,)
    # Broadcasting shift to (d, 1) to add to (d, n_poison)
    poisoned_samples[:, poison_indices] += shift.unsqueeze(1)
    
    return poisoned_samples

def black_box_cluster(
    z: torch.Tensor,
    theta: torch.Tensor,
    eta: float = 0.1,
    proj_theta: Callable = lambda x: x,
    delta: float = 100.0,
    epsilon: float = 0.5,
    norm = 'linf', # 'l2' or 'linf',
    **kwargs
):
    """
    Cluster Mean Shift Black Box Adversary
    
    Shifts an epsilon fraction of samples in each cluster towards the mean of the other cluster.
    Assumes z contains features in first d rows and labels in the last row.
    """
    assert z.shape[0] == 2, "z must be 1D + label"

    n_samples = z.shape[1]
    n_poison = int(epsilon * n_samples)
    
    if n_poison == 0:
        return z
    
    poisoned_samples = z.clone()

    # Split clean samples into features and labels
    x = z[0, :]
    y = z[1, :]

    # Identify indices for each cluster
    idx_0 = (y == 0).nonzero(as_tuple=True)[0]
    idx_1 = (y == 1).nonzero(as_tuple=True)[0]
    
    # Calculate means (fix: x is 1D, so use x[idx_0] not x[:, idx_0])
    mu_0 = x[idx_0].mean()  # scalar
    mu_1 = x[idx_1].mean()  # scalar
    
    # Direction from cluster 0 to cluster 1
    diff = mu_1 - mu_0
    direction = diff / (torch.abs(diff) + 1e-12)  # l2 and linf norm are the same for 1D
    # Shift cluster 0 towards cluster 1: +direction
    # Shift cluster 1 towards cluster 0: -direction
    shift_vec = delta * direction  # scalar for 1D
    
    # Poison epsilon fraction of cluster 0 - select points closest to mu_1
    n_poison_0 = int(epsilon * len(idx_0))
    if n_poison_0 > 0:
        # Calculate distances from cluster 0 points to mu_1
        x_0 = x[idx_0]  # features of cluster 0
        distances_to_mu1 = torch.abs(x_0 - mu_1)
        # Get indices of the n_poison_0 closest points
        _, closest_indices_0 = torch.topk(distances_to_mu1, n_poison_0, largest=False)
        idxs_to_poison_0 = idx_0[closest_indices_0]
        for i in idxs_to_poison_0:
             poisoned_samples[0, i] = poisoned_samples[0, i] + shift_vec
             
    # Poison epsilon fraction of cluster 1 - select points closest to mu_0
    n_poison_1 = int(epsilon * len(idx_1))
    if n_poison_1 > 0:
        # Calculate distances from cluster 1 points to mu_0
        x_1 = x[idx_1]  # features of cluster 1
        distances_to_mu0 = torch.abs(x_1 - mu_0)
        # Get indices of the n_poison_1 closest points
        _, closest_indices_1 = torch.topk(distances_to_mu0, n_poison_1, largest=False)
        idxs_to_poison_1 = idx_1[closest_indices_1]
        for i in idxs_to_poison_1:
             poisoned_samples[0, i] = poisoned_samples[0, i] - shift_vec

    return poisoned_samples