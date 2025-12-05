import torch
from typing import Callable, Optional, Any

def RGD(
    D_theta: Callable,
    loss: Callable,
    theta_0: torch.Tensor,
    n: int = 1000,
    eta: float = 0.1,
    max_iter: int = 100,
    proj_theta: Callable = lambda x: x,
    poison_function: Optional[Callable] = None,
    return_losses: bool = False,
    normalize_grad: bool = True,
    return_samples: bool = False,
    **poison_kwargs
):
    """
    Repeated Gradient Descent algorithm with optional poisoning.
    
    Inputs:
        D_theta: Distribution function that takes theta and n, returns samples
        loss: Loss function that takes samples and theta
        theta_0: Initial parameter vector
        n: Number of samples per iteration
        eta: Learning rate
        tol: Convergence tolerance
        max_iter: Maximum number of iterations
        poison_function: Optional poisoning function
        return_losses: Whether to return loss history
        return_samples: Whether to return clean and poisoned samples from last iteration
        **poison_kwargs: Additional keyword arguments for poisoning function
    
    Outputs:
        Final theta and list of all theta values during optimization
        Optionally: loss history, and clean/poisoned samples
    """
    all_thetas = [theta_0.clone().detach().squeeze()]
    all_losses = []

    theta_t = theta_0

    # Record loss on TRUE (not poisoned) distribution
    z_true = D_theta(theta_t, n)
    true_loss = loss(z_true, theta_t).mean()
    all_losses.append(true_loss.item())
    
    z_clean_last = None
    z_poisoned_last = None
    
    for t in range(max_iter):
        
        # Draw n samples from D(theta)
        z_clean = D_theta(theta_t, n)
        
        if poison_function is not None:
            z_poisoned = poison_function(z_clean, theta_t, eta=eta, loss=loss, proj_theta=proj_theta, **poison_kwargs)
            z = z_poisoned
            # Store samples from last iteration if requested
            if return_samples and t == max_iter - 1:
                z_clean_last = z_clean.clone()
                z_poisoned_last = z_poisoned.clone()
        else:
            z = z_clean
            # Store samples from last iteration if requested
            if return_samples and t == max_iter - 1:
                z_clean_last = z_clean.clone()
                z_poisoned_last = z_clean.clone()  # Same as clean if no poisoning

        # Compute gradient of loss (dL1)
        theta_t.requires_grad_(True)
        theta_t.grad = None 
        l_t = loss(z, theta_t).mean()
        l_t.backward()
        dL1 = theta_t.grad
        
        with torch.no_grad():
            
            if normalize_grad:
                theta_t = theta_t - eta * dL1 / dL1.norm()
            else:
                theta_t = theta_t - eta * dL1
            
            theta_t = proj_theta(theta_t) # Projection 
        
        all_thetas.append(theta_t.detach().squeeze())
        
        # Record loss on TRUE (not poisoned) distribution
        z_true = D_theta(theta_t, n)
        true_loss = loss(z_true, theta_t).mean()
        all_losses.append(true_loss.item())

    # Build return tuple
    result = [theta_t, all_thetas]
    if return_losses:
        result.append(all_losses)
    if return_samples:
        result.append(z_clean_last)
        result.append(z_poisoned_last)
    
    return result

def RGD_audit(
    D_theta: Callable,
    loss: Callable,
    theta_0: torch.Tensor,
    n: int = 1000,
    eta: float = 0.1,
    max_iter: int = 100,
    proj_theta: Callable = lambda x: x,
    poison_function: Optional[Callable] = None,
    last_step_only: bool = False,
    **poison_kwargs
):
    """
    Repeated Gradient Descent algorithm with optional poisoning.
    
    Inputs:
        D_theta: Distribution function that takes theta and n, returns samples
        loss: Loss function that takes samples and theta
        theta_0: Initial parameter vector
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
    all_losses = []
    convergence_metrics = []
    
    theta_t = theta_0

    # Record loss on TRUE (not poisoned) distribution
    z_true = D_theta(theta_t, n)
    true_loss = loss(z_true, theta_t).mean()
    all_losses.append(true_loss.item())
    
    for t in range(max_iter):
        
        # Draw n samples from D(theta)
        z = D_theta(theta_t, n)
    
        if (poison_function is not None) and not last_step_only:
            z = poison_function(z, theta_t, eta=eta, loss=loss, proj_theta=proj_theta, **poison_kwargs)
            
        if (poison_function is not None) and last_step_only and (t == max_iter - 1):
            z = poison_function(z, theta_t, eta=eta, loss=loss, proj_theta=proj_theta, **poison_kwargs)

        # Compute gradient of loss (dL1)
        theta_t.requires_grad_(True)
        theta_t.grad = None 
        l_t = loss(z, theta_t).mean()
        l_t.backward()
        dL1 = theta_t.grad
        
        with torch.no_grad():
            theta_t = theta_t - eta * dL1 / dL1.norm()
            theta_t = proj_theta(theta_t)
        
        all_thetas.append(theta_t.detach().squeeze())

        # Record loss on TRUE (not poisoned) distribution
        z_true = D_theta(theta_t, n)
        true_loss = loss(z_true, theta_t).mean()
        all_losses.append(true_loss.item())
        
        # Compute convergence audit metric
        dLt = torch.autograd.functional.jacobian(lambda th: loss(z_true, th).squeeze(), theta_t)
        dLp = torch.autograd.functional.jacobian(lambda th: loss(z, th).squeeze(), theta_t)
        
        M = dLt @ dLp.T
        c = M.sum()/M.norm()
        convergence_metrics.append(c.item())

    return theta_t, all_thetas, all_losses, convergence_metrics

def PerfGD(
    f_hat: Callable,
    grad2_est: Callable,
    D_theta: Callable,
    loss: Callable,
    theta_0: torch.Tensor,
    proj_theta: Callable = lambda x: x,
    n: int = 1000,
    eta: float = 0.1,
    max_iter: int = 100,
    poison_function: Optional[Callable] = None,
    return_losses: bool = False,
    normalize_grad: bool = True,
    **poison_kwargs
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
        tol: Convergence tolerance
        max_iter: Maximum number of iterations
        poison_function: Optional poisoning function
        return_losses: Whether to return the losses
        **poison_kwargs: Additional keyword arguments for poisoning function
    
    Outputs:
        Final theta and list of all theta values during optimization, and list of all losses during optimization
    """
    all_thetas = [theta_0.clone().detach().squeeze()]
    all_losses = []

    # Record loss on TRUE (not poisoned) distribution
    z_true = D_theta(theta_0, n)
    true_loss = loss(z_true, theta_0).mean()
    all_losses.append(true_loss.item())

    # Use unlimited history instead of fixed H
    theta_history = []
    f_history = []

    # Take only 1 step of RGD for initialization
    theta_t = theta_0
    z = D_theta(theta_t, n) # Draw n samples from D(theta)

    if poison_function is not None:
        z = poison_function(
                z, 
                theta_t, 
                eta=eta, 
                loss=loss, 
                f_hat=f_hat, 
                theta_history=theta_history, 
                f_history=f_history, 
                grad2_est=grad2_est, 
                proj_theta=proj_theta, 
                **poison_kwargs
            )
    
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
        if normalize_grad:
            theta_t = proj_theta(theta_t - eta * dL1 / dL1.norm())
        else:
            theta_t = proj_theta(theta_t - eta * dL1)

    # Record loss on TRUE (not poisoned) distribution
    z_true = D_theta(theta_t, n)
    true_loss = loss(z_true, theta_t).mean()
    all_losses.append(true_loss.item())

    # Record loss on TRUE (not poisoned) distribution
    z_true = D_theta(theta_t, n)
    true_loss = loss(z_true, theta_t).mean()
    all_losses.append(true_loss.item())

    all_thetas.append(theta_t.detach().squeeze())

    # Now run with full gradient update using entire history
    for t in range(max_iter):

        z = D_theta(theta_t, n) # Draw n samples from D(theta) (d x n)

        if poison_function is not None:
            z = poison_function(
                z, 
                theta_t, 
                eta=eta, 
                loss=loss, 
                proj_theta=proj_theta, 
                f_hat=f_hat, 
                theta_history=theta_history, 
                f_history=f_history, 
                grad2_est=grad2_est, 
                **poison_kwargs
            )

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
            if normalize_grad:
                theta_t = proj_theta(theta_t - eta * (dL1 + dL2) / (dL1 + dL2).norm())
            else:
                theta_t = proj_theta(theta_t - eta * (dL1 + dL2))

        all_thetas.append(theta_t.detach().squeeze())
        # Record loss on TRUE (not poisoned) distribution
        z_true = D_theta(theta_t, n)
        true_loss = loss(z_true, theta_t).mean()
        all_losses.append(true_loss.item())

    
    if return_losses:
        return theta_t, all_thetas, all_losses
    else:
        return theta_t, all_thetas