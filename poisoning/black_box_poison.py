import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import Callable, Optional, Literal
import torch
from utils.algorithms import RGD
from utils.experiment_setup import setup_1d_experiment, setup_2d_experiment
from utils.plotting import process_1d_results, process_2d_results, plot_1d, plot_2d


def myopic_mean_shift_poisoning(
    z: torch.Tensor,
    theta: torch.Tensor,
    eta: float = 0.1,
    epsilon: float = 0.05,
    delta: float = 0.05,
    **kwargs
):
    """
    Myopic Black Box Adversary - Mean Shift Strategy
    
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


def run_experiment(
    dimensions: Literal[1, 2] = 1,
    poisoning_type: Optional[str] = "myopic_mean_shift_poisoning",
    algorithm: Literal["RGD"] = "RGD",
    # Experiment parameters
    a0: float = 1.0,
    a1: float = 1.0,
    c: float = 1.0,  # Only used for 2D
    # Poisoning parameters
    epsilon: float = 0.1,
    delta: float = 1.0,
    # RGD parameters
    n: int = 500,
    eta: float = 0.1,
    max_iter: int = 30,
    num_trials: int = 3,
    # Other parameters
    device: Optional[torch.device] = None,
    # Plotting
    plot: bool = True,
    show_theoretical: bool = True,  # Only for 1D
):
    """
    Unified experiment runner for poisoning experiments.
    
    Inputs:
        dimensions: 1 or 2
        poisoning_type: "myopic_mean_shift_poisoning" or None for clean
        algorithm: "RGD"
        epsilon: fraction of samples to poison
        delta: per-sample L2 step size
        n: number of samples per iteration
        eta: learning rate
        max_iter: maximum iterations
        num_trials: number of trials to average over
        device: torch device (default: CPU)
        plot: whether to plot results
        show_theoretical: whether to show theoretical solutions (1D only)
    """
    if device is None:
        device = torch.device("cpu")
    
    # Setup experiment based on dimensions
    if dimensions == 1:
        _, D_theta, loss, theta0 = setup_1d_experiment(a0=a0, a1=a1, device=device)
    elif dimensions == 2:
        _, D_theta, loss, theta0 = setup_2d_experiment(a0=a0, a1=a1, c=c, device=device)
    else:
        raise ValueError(f"dimensions must be 1 or 2, got {dimensions}")
    
    # Select algorithm (only RGD supported)
    if algorithm != "RGD":
        raise ValueError(f"algorithm must be 'RGD', got {algorithm}")
    alg_func = RGD
    
    # Select poison function
    poison_function = None
    if poisoning_type == "myopic_mean_shift" or poisoning_type == "myopic_mean_shift_poisoning":
        poison_function = myopic_mean_shift_poisoning
    elif poisoning_type is not None:
        raise ValueError(f"Unknown poisoning_type: {poisoning_type}")
    
    poison_kwargs = {"epsilon": epsilon, "delta": delta} if poison_function else {}
    
    # Storage
    all_clean_thetas = []
    all_poisoned_thetas = []
    
    print(f"Running {algorithm} experiment (dim={dimensions}D)")
    if poisoning_type:
        print(f"  Poisoning: {poisoning_type} (ε={epsilon}, δ={delta})")
    else:
        print(f"  Clean (no poisoning)")
    print(f"  Trials: {num_trials}, Iterations: {max_iter}, Samples: {n}")
    
    for trial in range(num_trials):
        print(f"  Trial {trial + 1}/{num_trials}")
        
        # Clean run
        _, thetas_clean = alg_func(
            D_theta=D_theta,
            loss=loss,
            theta_0=theta0,
            n=n,
            eta=eta,
            max_iter=max_iter,
        )
        
        # Store clean results
        all_clean_thetas.append(torch.stack(thetas_clean))
        
        # Poisoned run (if applicable)
        if poison_function:
            _, thetas_pois = alg_func(
                D_theta=D_theta,
                loss=loss,
                theta_0=theta0,
                n=n,
                eta=eta,
                max_iter=max_iter,
                poison_function=poison_function,
                **poison_kwargs,
            )
            all_poisoned_thetas.append(torch.stack(thetas_pois))
    
    # Process results
    if dimensions == 1:
        results = process_1d_results(
            all_clean_thetas, all_poisoned_thetas, D_theta, poison_function,
            poison_kwargs, eta, epsilon, delta, a0, a1, num_trials
        )
    else:
        results = process_2d_results(
            all_clean_thetas, all_poisoned_thetas, D_theta, poison_function,
            poison_kwargs, eta, epsilon, delta, num_trials
        )
    
    # Plot if requested
    if plot:
        if dimensions == 1:
            plot_1d(results, epsilon, delta, a0, a1, show_theoretical)
        else:
            plot_2d(results, epsilon, delta)
    
    return results


def run_experiment_2d(
    epsilon: float = 0.2,
    delta: float = 10.0,
    n: int = 500,
    eta: float = 0.1,
    max_iter: int = 40,
    num_trials: int = 3,
    plot: bool = True,
):
    """
    Convenience function to run 2D mean shift poisoning experiment.
    
    Args:
        epsilon: fraction of samples to poison
        delta: per-sample L2 step size
        n: number of samples per iteration
        eta: learning rate
        max_iter: maximum iterations
        num_trials: number of trials to average over
        plot: whether to plot results
    """
    return run_experiment(
        dimensions=2,
        poisoning_type="myopic_mean_shift_poisoning",
        algorithm="RGD",
        epsilon=epsilon,
        delta=delta,
        n=n,
        eta=eta,
        max_iter=max_iter,
        num_trials=num_trials,
        plot=plot,
    )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "2d":
        # Run 2D experiment
        run_experiment_2d()
    else:
        # Default: 1D experiment
        run_experiment()
