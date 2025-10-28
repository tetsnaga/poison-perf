import torch
from typing import Callable, Optional, Any

def RGD(
    D_theta: Callable,
    loss: Callable,
    theta_0: torch.Tensor,
    proj_theta: Callable = lambda x: x,
    n: int = 1000,
    eta: float = 0.1,
    tol: float = 1e-5,
    max_iter: int = 100,
    poison_function: Optional[Callable] = None,
    **poison_kwargs
):
    """
    Repeated Gradient Descent algorithm with optional poisoning.
    
    Inputs:
        D_theta: Distribution function that takes theta and n, returns samples
        loss: Loss function that takes samples and theta
        theta_0: Initial parameter vector
        proj_theta: Projection function for theta (default: identity)
        n: Number of samples per iteration
        eta: Learning rate
        tol: Convergence tolerance
        max_iter: Maximum number of iterations
        poison_function: Optional poisoning function
        **poison_kwargs: Additional keyword arguments for poisoning function
    
    Outputs:
        Final theta and list of all theta values during optimization
    """
    all_thetas = [theta_0.clone().detach().squeeze()]
    
    theta_t = theta_0
    converged = False
    t = 0
    
    while not converged and t < max_iter:
        # Draw n samples from D(theta)
        z = D_theta(theta_t, n)
        
        if poison_function is not None:
            z = poison_function(z, theta_t, eta=eta, **poison_kwargs)

        # Compute gradient of loss (dL1)
        theta_t.requires_grad_(True)
        theta_t.grad = None 
        l_t = loss(z, theta_t).mean(dim=-1)
        l_t.backward()
        dL1 = theta_t.grad
        
        with torch.no_grad():
            theta_t = proj_theta(theta_t - eta * dL1)
        
        all_thetas.append(theta_t.detach().squeeze())
        
        # Check convergence
        if dL1.norm() < tol: 
            converged = True
        
        t += 1
    
    return theta_t, all_thetas


def PerfGD(
    f_hat: Callable,
    grad2_est: Callable,
    D_theta: Callable,
    loss: Callable,
    theta_0: torch.Tensor,
    proj_theta: Callable = lambda x: x,
    n: int = 1000,
    eta: float = 0.1,
    warmup: int = 1,
    tol: float = 1e-5,
    max_iter: int = 100
):
    """
    Performative Gradient Descent algorithm.
    
    Inputs:
        f_hat: Estimator function
        grad2_est: Gradient estimation function for performative correction
        D_theta: Distribution function that takes theta and n, returns samples
        loss: Loss function that takes samples and theta
        theta_0: Initial parameter vector
        proj_theta: Projection function for theta (default: identity)
        n: Number of samples per iteration
        eta: Learning rate
        warmup: Number of warmup iterations
        tol: Convergence tolerance
        max_iter: Maximum number of iterations
    
    Outputs:
        Final theta and list of all theta values during optimization
    """
    all_thetas = [theta_0.clone().detach().squeeze()]

    # Use unlimited history instead of fixed H
    theta_history = []
    f_history = []

    # Take only 1 step of RGD for initialization
    theta_t = theta_0
    z = D_theta(theta_t, n) # Draw n samples from D(theta)
    f_t = f_hat(z)  # Estimate for f(theta)

    theta_history.append(theta_t.detach())
    f_history.append(f_t.detach())

    # One gradient descent step on Loss_1 loss
    theta_t.requires_grad_(True)
    theta_t.grad = None 
    l_t = loss(z, theta_t).mean(dim=-1)
    l_t.backward()
    dL1 = theta_t.grad

    with torch.no_grad():
        theta_t = proj_theta(theta_t - eta * dL1)

    all_thetas.append(theta_t.detach().squeeze())

    # Now run with full gradient update using entire history
    converged = False
    t = 1
    while not converged:

        z = D_theta(theta_t, n) # Draw n samples from D(theta) (d x n)
        f_t = f_hat(z) # (d,)
        
        theta_history.append(theta_t.detach())
        f_history.append(f_t.detach())

        theta_t.requires_grad_(True)
        theta_t.grad = None 
        l_t = loss(z, theta_t).mean(dim=-1)
        l_t.backward()
        dL1 = theta_t.grad
        
        # Use entire history (not just last H steps)
        if len(theta_history) >= 2:  # Need at least 2 points for gradient estimation
            Theta_full = torch.stack([th for th in theta_history], dim=1)   # (p, t+1)
            F_full = torch.stack([fh for fh in f_history], dim=1)   # (q, t+1)
            
            Theta = Theta_full[:, :-1]                                      # (p, t) -> 0 : t-1
            F = F_full[:, :-1]
            oneH = torch.ones(1, Theta.shape[1], device=Theta.device)

            delta_theta = Theta - theta_t.unsqueeze(1) @ oneH          # (p, t)
            delta_f = F - f_t.unsqueeze(1) @ oneH          # (q, t)

            df_d_theta = delta_f @ torch.linalg.pinv(delta_theta) # (p x q)

            dL2 = grad2_est(z, f_t, theta_t, df_d_theta)
        else:
            dL2 = torch.zeros_like(dL1)  # No performative correction if not enough history

        with torch.no_grad():
            theta_t = proj_theta(theta_t - eta * (dL1 + dL2))

        all_thetas.append(theta_t.detach().squeeze())
        
        if (dL1 + dL2).norm() < tol: 
            break # Converged
        
        t += 1
        if t > max_iter: 
            break # Max iterations
    
    return theta_t, all_thetas