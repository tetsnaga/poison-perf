# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import Callable, Optional, Literal
import torch



def black_box_poison_naive(
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


def black_box_poison_orthogonal(
    z: torch.Tensor,
    theta: torch.Tensor,
    eta: float = 0.1,
    epsilon: float = 0.1,
    delta: float = 1.0,
    **kwargs
):
    """
    Orthogonal Mean Shift Black Box Adversary
    
    Generic version that works for d >= 2.
    Approach: shift the mean of the data samples orthogonal to the direction of the mean from origin.
    Only uses input data samples without history of data.

    Note: Some arguments are not used in this implementation, but kept for compatibility with RGD() function.
    """
    if z.shape[0] != 2:
        raise ValueError("z must have 2 dimensions")
        
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

def black_box_poison_classification():
    raise NotImplementedError("Blackbox class shift not implemented")




def run_experiment(
    dimensions: Literal[1, 2] = 1,
    poisoning_type: Optional[str] = "naive_mean_shift",
    algorithm: Literal["RGD"] = "RGD",
    # Experiment parameters
    a0: float = 1.0,
    a1: float = 1.0,
    # Poisoning parameters
    epsilon: float = 0.1,
    delta: float = 1.0,
    # RGD parameters
    n: int = 500,
    eta: float = 0.1,
    max_iter: int = 30,
    num_trials: int = 5,
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
        poisoning_type: "naive_mean_shift" or "orthogonal_mean_shift" or None for clean
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
        _, D_theta, loss, theta0 = setup_2d_experiment(a0=a0, a1=a1, c=c, device=device, poisoning_type=poisoning_type)
    else:
        raise ValueError(f"dimensions must be 1 or 2, got {dimensions}")
    
    # Select algorithm (only RGD supported)
    if algorithm != "RGD":
        raise ValueError(f"algorithm must be 'RGD', got {algorithm}")
    alg_func = RGD
    
    # Select poison function
    poison_function = None
    if poisoning_type == "naive_mean_shift":
        poison_function = naive_mean_shift
    elif poisoning_type == "orthogonal_mean_shift":
        poison_function = orthogonal_mean_shift
    
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
            all_clean_thetas, all_poisoned_thetas, D_theta, loss, poison_function,
            poison_kwargs, eta, epsilon, delta, a0, a1, num_trials
        )
    else:
        results = process_2d_results(
            all_clean_thetas, all_poisoned_thetas, D_theta, loss, poison_function,
            poison_kwargs, eta, epsilon, delta, num_trials
        )
    
    # Plot if requested
    if plot:
        if dimensions == 1:
            plot_1d(results, epsilon, delta, a0, a1, show_theoretical, poisoning_type=poisoning_type)
        else:
            plot_2d(results, epsilon, delta, poisoning_type=poisoning_type)
    
    return results


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Run black-box poisoning experiments")
    parser.add_argument(
        "--dimensions", "-d",
        type=int,
        choices=[1, 2],
        default=None,
        help="Number of dimensions (1 or 2)"
    )
    parser.add_argument(
        "--poisoning-type", "-p",
        type=str,
        choices=["naive_mean_shift", "orthogonal_mean_shift"],
        default=None,
        help="Type of poisoning to use (naive_mean_shift or orthogonal_mean_shift)"
    )
    parser.add_argument(
        "--epsilon", "-e",
        type=float,
        default=None,
        help="Fraction of samples to poison"
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=None,
        help="Per-sample L2 step size"
    )
    # Positional arguments for backward compatibility
    parser.add_argument(
        "positional",
        nargs="*",
        help="Positional args: [dimensions] [poisoning_type] [epsilon] [delta]"
    )
    
    args = parser.parse_args()
    
    # Handle positional arguments (for backward compatibility)
    dimensions = args.dimensions
    poisoning_type = args.poisoning_type
    epsilon = args.epsilon
    delta = args.delta
    
    if args.positional:
        # Parse positional arguments
        if len(args.positional) >= 1:
            # First arg: dimensions (can be "2d" or "2" or "1")
            dim_str = args.positional[0].lower()
            if dim_str == "2d" or dim_str == "2":
                dimensions = 2
            elif dim_str == "1d" or dim_str == "1":
                dimensions = 1
            else:
                # Try to parse as integer
                try:
                    dimensions = int(dim_str)
                except ValueError:
                    # If not a number, treat as poisoning_type
                    poisoning_type = dim_str
                    dimensions = 2  # Default to 2D if first arg is poisoning type
        
        if len(args.positional) >= 2:
            # Second arg: poisoning_type
            if dimensions is None:
                # First arg was poisoning_type, second is epsilon
                epsilon = float(args.positional[1])
            else:
                poisoning_type = args.positional[1]
        
        if len(args.positional) >= 3:
            # Third arg: epsilon
            if dimensions is None:
                delta = float(args.positional[2])
            else:
                epsilon = float(args.positional[2])
        
        if len(args.positional) >= 4:
            # Fourth arg: delta
            delta = float(args.positional[3])
    
    # Set defaults
    if dimensions is None:
        dimensions = 1
    
    # Build kwargs
    kwargs = {"dimensions": dimensions}
    if poisoning_type:
        kwargs["poisoning_type"] = poisoning_type
    if epsilon is not None:
        kwargs["epsilon"] = epsilon
    if delta is not None:
        kwargs["delta"] = delta
    
    run_experiment(**kwargs)
