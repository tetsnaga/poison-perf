"""
Setup functions for mu, D_theta, loss, and theta0 for 1D and 2D mean poisoning experiments.
"""
import torch
from typing import Callable, Tuple, Optional
import numpy as np


def setup_1d_non_linear_experiment(
    a0: float = 1.0,
    a1: float = 1.0,
    device: torch.device = 'cpu',
    perfGD: bool = False
):
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

    sigma = torch.tensor([[1.0]], device=device)

    def grad2_est(z, f, theta, df_d_theta):
        if z.ndim == 1: z = z.unsqueeze(0)
        if f.ndim == 1: f = f.unsqueeze(0)
        return torch.mean(loss(z, theta) * (df_d_theta.T @ torch.linalg.inv(sigma) @ (z - f)), dim=-1)

    f_hat = lambda z: z.mean(dim=1)

    info = {"theta_optimal": (-2*a0)/(3*a1) , "theta_stable":  -a0/a1}
    if perfGD:
        return mu, sigma, D_theta, loss, theta_0, grad2_est, f_hat, info
    else:
        return mu, sigma, D_theta, loss, theta_0, info


def setup_2d_non_linear_experiment(
    a0: float = 1.0,
    a1: float = 1.0,
    perfGD: bool = False,
    device: torch.device = 'cpu',
    poisoning_type: Optional[str] = None
):
    """
    Setup 2D experiment with:
    - mu(theta)[0] = sqrt(a0 * theta[0] + a1)
    - mu(theta)[1] = sqrt(a0 * theta[1] + a1)
    
    Returns:
        mu: mean function
        D_theta: distribution function
        loss: loss function
        theta0: initial theta

    
    Inputs:
        a0, a1: coefficients for mu_x and mu_y
        perfGD: whether to return the performative gradient descent setup
        device: torch device (default: CPU)
        poisoning_type: optional poisoning type (e.g., "orthogonal_mean_shift_poisoning")
    
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

    sigma = L @ L.T

    def mu(theta: torch.Tensor) -> torch.Tensor:
        """Mean function for 2D"""
        theta_vec = theta.view(-1)
        mu_x = torch.sqrt(a0 * theta_vec[0] + a1)
        mu_y = torch.sqrt(a0 * theta_vec[1] + a1)
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
    
    theta_0 = torch.tensor([0.0, 0.0], dtype=torch.float32, device=device)

    def grad2_est(z, f, theta, df_d_theta):
        if z.ndim == 1: z = z.unsqueeze(0)
        if f.ndim == 1: f = f.unsqueeze(0)
        return torch.mean(loss(z, theta) * (df_d_theta.T @ torch.linalg.inv(sigma) @ (z - f)), dim=-1)

    f_hat = lambda z: z.mean(dim=1)

    # TODO: Add information about the optimal and stable theta
    info = {}

    if perfGD:
        return mu, sigma, D_theta, loss, theta_0, grad2_est, f_hat, info
    else:
        return mu, sigma, D_theta, loss, theta_0, info

def setup_binary_classification(
            mu_0: torch.Tensor = torch.tensor([1.0]), 
            mu_1: torch.Tensor = torch.tensor([-1.0]),
            sigma_0: float = 0.25,
            sigma_1: float = 0.25, 
            eps: float = 3.0,
            Lambda: float = 0.01,
            perfGD: bool = False
    ):

    mu_f = lambda theta: mu_1 - eps * theta[1]

    def D_theta(theta: torch.Tensor, n: int) -> torch.Tensor:
        y = torch.randint(low=0, high=2, size=(n,), dtype=torch.int32) # Labels 0/1
        x_1 = mu_f(theta).unsqueeze(-1) + np.sqrt(sigma_1) * torch.randn((n))
        x_0 = mu_0.unsqueeze(-1) + np.sqrt(sigma_0) * torch.randn((n))
        x = torch.where(y.unsqueeze(0) == 1, x_1, x_0)
        z = torch.vstack((x, y.float()))
        return z

    def loss(z: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        theta_0 = theta[0]
        theta_1 = theta[1]
        x = z[0, :]
        y = z[1, :]
        # h = 1 / (1 + torch.exp(-(theta_0 + theta_1 * x)))
        h = torch.sigmoid(theta_0 + theta_1 * x)
        loss = -y * torch.log(h + 1e-9) - (1 - y) * torch.log(1 - h + 1e-9) + (Lambda / 2) * torch.sum(theta**2)
        return loss
    
    def grad2_est(z, f, theta, df_d_theta):
        x = z[0, :].unsqueeze(0)
        y = z[1, :]
        sigma = torch.tensor([[sigma_1]])
        if z.ndim == 1: z = z.unsqueeze(0)
        if f.ndim == 1: f = f.unsqueeze(0)
        return torch.mean( loss(z, theta)[y == 1] * (df_d_theta.T @ torch.linalg.inv(sigma) @ (x[:, y == 1] - f)), dim=-1)

    def f_hat(z):
        x = z[0, :].unsqueeze(0)
        y = z[1, :]
        return torch.mean(x[:, y == 1], dim=1)  # Mean of x where y == 1
    
    theta_0 = torch.tensor([0.0, 0.0], dtype=torch.float32)
    
    if perfGD:
        return mu_f, mu_0, sigma_0, sigma_1, D_theta, loss, theta_0, grad2_est, f_hat
    else:
        return mu_f, mu_0, sigma_0, sigma_1, D_theta, loss, theta_0


def setup_binary_classification_nd(
            dim: int = 10,
            mu_0: Optional[torch.Tensor] = None,
            mu_1: Optional[torch.Tensor] = None,
            sigma_0: float = 0.25,
            sigma_1: float = 0.25, 
            eps: float = 3.0,
            Lambda: float = 0.01,
            perfGD: bool = False
    ):

    """
    Setup n-dimensional binary classification experiment.
    
    Args:
        dim: Dimension of feature space (default: 10)
        mu_0: Mean vector for class 0 (default: ones vector)
        mu_1: Mean vector for class 1 (default: negative ones vector)
        sigma_0: Standard deviation for class 0 (isotropic)
        sigma_1: Standard deviation for class 1 (isotropic)
        eps: Performative parameter for mu_f
        Lambda: L2 regularization strength
        perfGD: Whether to return performative gradient descent components
        
    Returns:
        mu_f, mu_0, sigma_0, sigma_1, D_theta, loss, theta_0, (grad2_est, f_hat if perfGD)
    """
    # Set default mean vectors if not provided
    if mu_0 is None:
        mu_0 = torch.ones(dim, dtype=torch.float32)
    if mu_1 is None:
        mu_1 = -torch.ones(dim, dtype=torch.float32)
    
    # Ensure mu_0 and mu_1 are the correct dimension
    assert mu_0.shape[0] == dim, f"mu_0 must have dimension {dim}, got {mu_0.shape[0]}"
    assert mu_1.shape[0] == dim, f"mu_1 must have dimension {dim}, got {mu_1.shape[0]}"
    
    # mu_f: performative mean function for class 1
    # theta[0] is bias, theta[1:] are feature coefficients
    def mu_f(theta: torch.Tensor) -> torch.Tensor:
        """Mean function: mu_f(theta) = mu_1 - eps * theta[1:]"""
        return mu_1 - eps * theta[1:]
    
    def D_theta(theta: torch.Tensor, n: int) -> torch.Tensor:
        """Sample from n-dimensional binary classification distribution"""
        y = torch.randint(low=0, high=2, size=(n,), dtype=torch.int32)  # Labels 0/1
        
        # Generate samples for class 1: N(mu_f(theta), sigma_1^2 * I)
        mu_1_theta = mu_f(theta)  # (dim,)
        x_1 = mu_1_theta.unsqueeze(1) + np.sqrt(sigma_1) * torch.randn(dim, n)  # (dim, n)
        
        # Generate samples for class 0: N(mu_0, sigma_0^2 * I)
        x_0 = mu_0.unsqueeze(1) + np.sqrt(sigma_0) * torch.randn(dim, n)  # (dim, n)
        
        # Select x based on label
        # x_1 and x_0 are (dim, n), y is (n,)
        # We need to select columns based on y
        x = torch.where(y.unsqueeze(0) == 1, x_1, x_0)  # (dim, n)
        
        # Stack: (dim+1, n) = (features, labels)
        z = torch.vstack((x, y.float()))
        return z
    
    def loss(z: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Binary cross-entropy loss with L2 regularization"""
        theta_0 = theta[0]  # bias
        theta_features = theta[1:]  # (dim,) feature coefficients
        x = z[:-1, :]  # (dim, n) features
        y = z[-1, :]  # (n,) labels
        
        # Logit: theta_0 + theta_features^T @ x for each sample
        # x is (dim, n), theta_features is (dim,)
        logits = theta_0 + (theta_features.unsqueeze(0) @ x).squeeze(0)  # (n,)
        
        h = torch.sigmoid(logits)  # (n,)
        loss = -y * torch.log(h + 1e-9) - (1 - y) * torch.log(1 - h + 1e-9) + (Lambda / 2) * torch.sum(theta**2)
        return loss
    
    def grad2_est(z, f, theta, df_d_theta):
        """Performative gradient estimation"""
        x = z[:-1, :]  # (dim, n) features
        y = z[-1, :]  # (n,) labels
        
        # Covariance matrix for class 1 (isotropic)
        sigma = sigma_1 * torch.eye(dim, dtype=torch.float32)
        
        if z.ndim == 1:
            z = z.unsqueeze(0)
        # Keep f as (dim,) so diff stays (dim, n_class1); do not unsqueeze f here
        f_vec = f.squeeze()  # (dim,)
        
        # Select samples where y == 1
        x_class1 = x[:, y == 1]  # (dim, n_class1)
        loss_class1 = loss(z, theta)[y == 1]  # (n_class1,)
        
        # diff = (dim, n_class1); df_d_theta is (dim, dim+1), so df_d_theta.T is (dim+1, dim)
        diff = x_class1 - f_vec.unsqueeze(1)  # (dim, n_class1)
        term = df_d_theta.T @ torch.linalg.inv(sigma) @ diff  # (dim+1, n_class1)
        # (n_class1,) * (dim+1, n_class1) -> (dim+1, n_class1), mean over samples -> (dim+1,)
        n_class1 = loss_class1.shape[0]
        if n_class1 == 0:
            return torch.zeros(theta.shape[0], dtype=theta.dtype, device=theta.device)
        return (loss_class1 * term).sum(dim=1) / n_class1
    
    def f_hat(z):
        """Estimate f(theta) = E[x | y=1]"""
        x = z[:-1, :]  # (dim, n) features
        y = z[-1, :]  # (n,) labels
        return torch.mean(x[:, y == 1], dim=1)  # (dim,) mean of x where y == 1
    
    # Initial theta: [bias, feature_coeff_1, ..., feature_coeff_dim]
    theta_0 = torch.zeros(dim + 1, dtype=torch.float32)
    
    if perfGD:
        return mu_f, mu_0, sigma_0, sigma_1, D_theta, loss, theta_0, grad2_est, f_hat
    else:
        return mu_f, mu_0, sigma_0, sigma_1, D_theta, loss, theta_0
    
def setup_non_convex_1d(
        a = 1,
        b = 0,
        c = 100,
        beta_0 = -2.0,
        beta_1 = 0.8,
        perfGD: bool = False
    ):

    sigma = torch.tensor([[1.0]])

    theta_0 = torch.tensor([0.0], dtype=torch.float32)

    def mu(theta: torch.Tensor) -> torch.Tensor:
        """Mean function"""
        return beta_0 - beta_1 * theta
    
    def D_theta(theta: torch.Tensor, n: int) -> torch.Tensor:
        mean = mu(theta)  # scalar tensor
        z = mean + sigma * torch.randn(1, n)
        return z

    def loss(z: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return a*(theta-z)**2 + c * torch.cos((theta/b))
    
    if perfGD:
        raise NotImplementedError("perfGD not implemented for non-convex 1D setup")
    else:
        return mu, sigma, D_theta, loss, theta_0
    
def setup_non_convex_nd(
        dim = 10,
        beta_0 = -2.0,
        beta_1 = 0.8,
        alpha = 1.0,
        seed = 42,
        perfGD: bool = False
    ):

    torch.manual_seed(seed)

    m1 = torch.randn(dim)
    m2 = torch.randn(dim)
    m3 = torch.randn(dim)

    a = 0.2
    b = 10
    c = 1
    d = 1

    sigma = torch.eye(dim)

    theta_0 = torch.zeros(dim)

    def mu(theta: torch.Tensor) -> torch.Tensor:
            """Mean function"""
            return beta_0 - alpha * (beta_1 * theta)
        
    def D_theta(theta: torch.Tensor, n: int) -> torch.Tensor:
        mean = mu(theta).unsqueeze(1)  # scalar tensor
        z = mean + sigma @ torch.randn(dim, n)
        return z
    
    def loss(z: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        if torch.any(torch.isnan(z)):
            raise ValueError("NaN values found in z")
        return (a * m1@(theta.unsqueeze(1)-z)**2  + b * torch.cos(m2@(theta.unsqueeze(1)-z)/1.9) + c * torch.cos(m3@(theta.unsqueeze(1))/1) + d * torch.dot(theta, theta))
    
    def grad2_est(z, f, theta, df_d_theta):
        if z.ndim == 1: z = z.unsqueeze(0)
        if f.ndim == 1: f = f.unsqueeze(-1)
        return torch.mean(loss(z, theta) * (df_d_theta.T @ torch.linalg.inv(sigma) @ (z - f)), dim=-1)

    f_hat = lambda z: z.mean(dim=1)

    if perfGD:
        return mu, sigma, D_theta, loss, theta_0, grad2_est, f_hat
    
    else:
        return mu, sigma, D_theta, loss, theta_0


# Moved to utils/strategic_setup.py — re-exported here for backwards compatibility
from utils.strategic_setup import setup_strategic_classification  # noqa: F401
