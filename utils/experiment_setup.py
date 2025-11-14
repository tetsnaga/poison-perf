"""
Setup functions for mu, D_theta, loss, and theta0 for 1D and 2D mean poisoning experiments.
"""
import torch
from typing import Callable, Tuple, Optional


def setup_1d_experiment(
    a0: float = 1.0,
    a1: float = 1.0,
    device: torch.device = None
) -> Tuple[Callable, Callable, Callable, torch.Tensor]:
    """
    Setup 1D experiment with mu(theta) = sqrt(a0*theta + a1).
    
    Returns:
        mu: mean function
        D_theta: distribution function
        loss: loss function
        theta0: initial theta
    """
    if device is None:
        device = torch.device("cpu")
    
    def mu(theta: torch.Tensor) -> torch.Tensor:
        """Mean function: mu(theta) = sqrt(a0*theta + a1)"""
        return torch.sqrt(a0 * theta.squeeze() + a1)
    
    def D_theta(theta: torch.Tensor, n: int) -> torch.Tensor:
        """Sample from N(mu(theta), 1)"""
        mean = mu(theta)  # scalar tensor
        # Generate noise and add to mean (broadcasting)
        noise = torch.randn(1, n, device=device) * 1.0
        samples = mean.unsqueeze(0).unsqueeze(0) + noise  # (1, n)
        return samples
    
    def loss(z: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Loss: z * theta"""
        return z * theta
    
    theta0 = torch.tensor([0.0], dtype=torch.float32, device=device)
    
    return mu, D_theta, loss, theta0


def setup_2d_experiment(
    a0: float = 1.0,
    a1: float = 1.0,
    c: float = 1.0,
    device: torch.device = None,
    poisoning_type: Optional[str] = None
) -> Tuple[Callable, Callable, Callable, torch.Tensor]:
    """
    Setup 2D experiment with:
    - mu(theta)[0] = sqrt(a0 * theta[0] + a1)
    - mu(theta)[1] = theta[1]^2 + c
    
    Args:
        a0: coefficient for theta[0] in mu_x
        a1: constant term in mu_x
        c: constant term in mu_y
        device: torch device (default: CPU)
        poisoning_type: optional poisoning type for validation (e.g., "orthogonal_mean_shift_poisoning")
    
    Returns:
        mu: mean function
        D_theta: distribution function
        loss: loss function
        theta0: initial theta
    """
    if device is None:
        device = torch.device("cpu")
    
    # Validate poisoning type if provided
    if poisoning_type is not None:
        if "orthogonal" in poisoning_type.lower() and poisoning_type != "orthogonal_mean_shift_poisoning":
            # Orthogonal mean shift requires 2D, which is already satisfied
            pass
        elif poisoning_type == "orthogonal_mean_shift_poisoning":
            # Orthogonal mean shift is designed for 2D, so this is correct
            pass
    
    # Identity covariance matrix (Cholesky factor L)
    L = torch.eye(2, dtype=torch.float32, device=device)
    
    def mu(theta: torch.Tensor) -> torch.Tensor:
        """Mean function for 2D"""
        theta_vec = theta.view(-1)
        mu_x = torch.sqrt(a0 * theta_vec[0] + a1)
        mu_y = theta_vec[1]**2 + c
        return torch.stack([mu_x, mu_y])  # shape: (2,)
    
    def D_theta(theta: torch.Tensor, n: int) -> torch.Tensor:
        """Sample from N(mu(theta), I)"""
        mean = mu(theta)  # (2,)
        eps = torch.randn(2, n, device=device)
        samples = mean.unsqueeze(1) + L @ eps  # (2, n)  # Uses Cholesky factor L to generalize to non-identity covariance matrices
        return samples
    
    def loss(z: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Loss: per-sample dot product z^T theta"""
        theta_vec = theta.view(z.shape[0], 1)
        return (z * theta_vec).sum(dim=0)  # (n,)  # Vector dot product (analogous to 1D scalar product z*theta)
    
    theta0 = torch.tensor([0.0, 0.0], dtype=torch.float32, device=device)
    
    return mu, D_theta, loss, theta0

