import numpy as np
import torch
from typing import Callable

def gaussian_sampling_estimator(z: torch.Tensor, theta: torch.Tensor, mu: Callable, sigma: torch.Tensor) -> Callable:
    mu_t = mu(theta)
    z_hat = mu_t.unsqueeze(-1) + sigma @ torch.randn_like(z) 
    return z_hat

def classification_sampling_estimator(z: torch.Tensor, theta: torch.Tensor, mu_f: Callable, mu_0: torch.Tensor, sigma_0: float, sigma_1: float):
    n = z.shape[1]
    mu_t = mu_f(theta)
    y_hat = torch.randint(low=0, high=2, size=(n,), dtype=torch.int32)
    x_1_hat = mu_t.unsqueeze(-1) + np.sqrt(sigma_1) * torch.randn(n)
    x_0_hat = mu_0.unsqueeze(-1) + np.sqrt(sigma_0) * torch.randn(n)
    x_hat = torch.where(y_hat.unsqueeze(0) == 1, x_1_hat, x_0_hat)
    z_hat = torch.vstack((x_hat, y_hat.float()))
    return z_hat

def oracle_poison_function(
    z: torch.Tensor, 
    theta: torch.Tensor, 
    theta_update_estimator: Callable,
    loss: Callable,
    eta: float,
    proj_theta: Callable = lambda x: x,
    poison_steps: int = 10, 
    poison_step_size: float = 0.1, 
    delta: float = 1e-6,
    epsilon: float = 1.0,
    norm = 'linf', # 'l2' or 'linf',
    sampling_estimator: Callable = gaussian_sampling_estimator,
    sampling_estimator_kwargs: dict = {},
    **theta_update_kwargs
    ):
    
    z_0 = z.clone().detach()
    for _ in range(poison_steps): 
       
        z.requires_grad_(True)
        z.grad = None

        theta_new = theta_update_estimator(z, theta, eta, loss, proj_theta=proj_theta, **theta_update_kwargs)
        assert theta_new.shape == theta.shape, "Shape mismatch in theta update estimator."

        z_new = sampling_estimator(z=z, theta=theta_new, **sampling_estimator_kwargs)
        assert z_new.shape == z.shape, f"Shape mismatch in z_new: expected {z.shape}, got {z_new.shape}."
        
        l_theta = loss(z_new, theta_new).mean()        
        l_theta.backward()
        
        dz = z.grad
        
        with torch.no_grad():
            
            sample_mask = torch.arange(z.shape[1], device=z.device) < int(epsilon * z.shape[1])
            dz = dz * sample_mask.unsqueeze(0)  # Only poison a fraction eps of samples
            z += poison_step_size * dz / (torch.norm(dz, dim=0, keepdim=True) + 1e-16)
            z_diff = z - z_0
            
            if norm == 'l2':
                z_diff_norm = torch.norm(z_diff, p=2, dim=0, keepdim=True) + 1e-16
                exceed_mask = (z_diff_norm > delta).float()
                z_diff = z_diff * (z_diff / z_diff_norm) * exceed_mask + z_diff * (1 - exceed_mask)
            
            elif norm == 'linf':
                z_diff_norm = torch.max(torch.abs(z_diff), dim=0, keepdim=True)[0] + 1e-16
                exceed_mask = (z_diff_norm > delta).float()
                z_diff = z_diff * (z_diff / z_diff_norm) * exceed_mask + z_diff * (1 - exceed_mask)

            assert z_diff.shape == z.shape, f"Shape mismatch in z_diff: expected {z.shape}, got {z_diff.shape}."

            z = z_0 + z_diff

    return z.detach()

def RGD_update_estimator(z, theta, eta, loss, proj_theta):

    theta.requires_grad_(True)
    theta.grad = None
    
    # Compute loss and gradient
    l_t = loss(z, theta).mean(dim=-1)
    dL1 = torch.autograd.grad(l_t, theta, create_graph=True)[0]
    
    # Update theta
    theta_new = theta - eta * dL1

    theta_new = proj_theta(theta_new)

    return theta_new

def PerfGD_update_estimator(z, theta, eta, loss, f_hat, theta_history, f_history, grad2_est, proj_theta):
    
    f_t = f_hat(z) # (d,)
    
    theta.requires_grad_(True)
    theta.grad = None
    l_t = loss(z, theta).mean()
    l_t.backward()
    dL1 = theta.grad

    # Use entire history (not just last H steps)
    if len(theta_history) >= 2:  # Need at least 2 points for gradient estimation
        Theta_full = torch.stack([th for th in theta_history], dim=1)   # (p, t+1)
        F_full = torch.stack([fh for fh in f_history], dim=1)   # (q, t+1)
        
        Theta = Theta_full[:, :-1]                                      # (p, t) -> 0 : t-1
        F = F_full[:, :-1]
        oneH = torch.ones(1, Theta.shape[1], device=Theta.device)

        delta_theta = Theta - theta.unsqueeze(1) @ oneH          # (p, t)
        delta_f = F - f_t.unsqueeze(1) @ oneH          # (q, t)

        df_d_theta = delta_f @ torch.linalg.pinv(delta_theta) # (p x q)

        dL2 = grad2_est(z, f_t, theta, df_d_theta)
    else:
        dL2 = torch.zeros_like(dL1)  # No performative correction if not enough history

    theta = proj_theta(theta - eta * (dL1 + dL2))

    return theta