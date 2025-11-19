def naive_mean_shift(
    z: torch.Tensor,
    theta: torch.Tensor,
    eta: float = 0.1,
    epsilon: float = 0.5,
    delta: float = 100.0,
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
    shift_dir = u / (u.norm() + 1e-12)

    # Pick any epsilon fraction of samples (selection does not affect mean shift magnitude)
    poison_indices = torch.randperm(n_samples)[:n_poison]

    # Shift all selected samples in the same direction by delta
    shift = delta * shift_dir.squeeze()  # (d,)
    for i in poison_indices:
        poisoned_samples[:, i] = z[:, i] + shift
    
    return poisoned_samples
