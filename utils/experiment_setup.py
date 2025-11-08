"""
Setup functions for mu, D_theta, loss, and theta0 for 1D and 2D mean poisoning experiments.
"""
import torch
from typing import Callable, Tuple


def setup_1d_non_linear_experiment(
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
        return torch.sqrt(a0 * theta + a1 + 1e-6)
    
    def D_theta(theta: torch.Tensor, n: int) -> torch.Tensor:
        """Sample from N(mu(theta), 1)"""
        mean = mu(theta)  # scalar tensor
        # Generate noise and add to mean (broadcasting)
        noise = torch.randn(1, n, device=device) * 1.0
        samples = mean.unsqueeze(-1) + noise  # (1, n)
        return samples
    
    def loss(z: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Loss: z * theta"""
        return z * theta
    
    theta_0 = torch.tensor([0.0], dtype=torch.float32, device=device)

    info = {"theta_optimal": (-2*a0)/(3*a1) , "theta_stable":  -a0/a1}
    
    return mu, D_theta, loss, theta_0, info


def setup_2d_non_linear_experiment(
    a0: float = 1.0,
    a1: float = 1.0,
    c: float = 1.0,
    device: torch.device = None
) -> Tuple[Callable, Callable, Callable, torch.Tensor]:
    """
    Setup 2D experiment with:
    - mu(theta)[0] = sqrt(a0 * theta[0] + a1)
    - mu(theta)[1] = theta[1]^2 + c
    
    Returns:
        mu: mean function
        D_theta: distribution function
        loss: loss function
        theta0: initial theta
    """
    if device is None:
        device = torch.device("cpu")
    
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

