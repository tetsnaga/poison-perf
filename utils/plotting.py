"""
Functions for processing and plotting experiment results.
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Callable, Optional, Dict, Any


def process_1d_results(
    all_clean_thetas, all_poisoned_thetas, D_theta, poison_function,
    poison_kwargs, eta, epsilon, delta, a0, a1, num_trials
):
    """Process 1D results."""
    all_trajectories = all_clean_thetas + (all_poisoned_thetas if all_poisoned_thetas else [])
    max_length = max(len(traj) for traj in all_trajectories) if all_trajectories else 0
    
    avg_clean_trajectory = []
    avg_poisoned_trajectory = []
    avg_clean_mean_trajectory = []
    avg_poisoned_mean_trajectory = []
    
    for i in range(max_length):
        clean_values = []
        poisoned_values = []
        clean_means = []
        poisoned_means = []
        
        for traj in all_clean_thetas:
            if i < len(traj):
                theta_i = traj[i]
                clean_values.append(theta_i.item())
                z_i = D_theta(theta_i, 500)
                clean_means.append(z_i.mean().item())
        
        if poison_function:
            for traj in all_poisoned_thetas:
                if i < len(traj):
                    theta_i = traj[i]
                    poisoned_values.append(theta_i.item())
                    z_i_clean = D_theta(theta_i, 500)
                    z_i_pois = poison_function(z_i_clean, theta_i, eta=eta, **poison_kwargs)
                    poisoned_means.append(z_i_pois.mean().item())
        
        if clean_values:
            avg_clean_trajectory.append(np.mean(clean_values))
            avg_clean_mean_trajectory.append(np.mean(clean_means) if clean_means else None)
        if poisoned_values:
            avg_poisoned_trajectory.append(np.mean(poisoned_values))
            avg_poisoned_mean_trajectory.append(np.mean(poisoned_means) if poisoned_means else None)
    
    # Compute theoretical solutions
    theta_opt = -2*a0/(3*a1)
    theta_stab = -a0/a1
    
    return {
        "avg_clean_trajectory": avg_clean_trajectory,
        "avg_poisoned_trajectory": avg_poisoned_trajectory,
        "avg_clean_mean_trajectory": avg_clean_mean_trajectory,
        "avg_poisoned_mean_trajectory": avg_poisoned_mean_trajectory,
        "theta_opt": theta_opt,
        "theta_stab": theta_stab,
        "all_clean_thetas": all_clean_thetas,
        "all_poisoned_thetas": all_poisoned_thetas,
    }


def process_2d_results(
    all_clean_thetas, all_poisoned_thetas, D_theta, poison_function,
    poison_kwargs, eta, epsilon, delta, num_trials
):
    """Process 2D results."""
    all_trajectories = all_clean_thetas + (all_poisoned_thetas if all_poisoned_thetas else [])
    T = max(t.shape[0] for t in all_trajectories) if all_trajectories else 0
    
    def pad_to_T(seq):
        last = seq[-1]
        if seq.shape[0] < T:
            pad = last.repeat(T - seq.shape[0], 1)
            return torch.cat([seq, pad], dim=0)
        return seq
    
    clean_stack = torch.stack([pad_to_T(t) for t in all_clean_thetas])  # (trials, T, 2)
    pois_stack = torch.stack([pad_to_T(t) for t in all_poisoned_thetas]) if all_poisoned_thetas else None
    
    avg_clean = clean_stack.mean(dim=0)   # (T, 2)
    avg_pois = pois_stack.mean(dim=0) if pois_stack is not None else None
    
    # Compute mean of samples over time
    avg_clean_mean = []
    avg_pois_mean = []
    for t in range(T):
        cm, pm = [], []
        for trial in range(clean_stack.shape[0]):
            theta_c = clean_stack[trial, t]
            zc = D_theta(theta_c, 500)
            zp_clean_mean = zc.mean(dim=1)  # (2,)
            cm.append(zp_clean_mean.cpu().numpy())
            
            if pois_stack is not None and poison_function:
                theta_p = pois_stack[trial, t]
                zp0 = D_theta(theta_p, 500)
                zp = poison_function(zp0, theta_p, eta=eta, **poison_kwargs)
                zp_pois_mean = zp.mean(dim=1)
                pm.append(zp_pois_mean.cpu().numpy())
        
        avg_clean_mean.append(np.mean(np.stack(cm), axis=0))
        if pm:
            avg_pois_mean.append(np.mean(np.stack(pm), axis=0))
    
    avg_clean_mean = np.stack(avg_clean_mean)  # (T, 2)
    avg_pois_mean = np.stack(avg_pois_mean) if avg_pois_mean else None
    
    return {
        "avg_clean": avg_clean,
        "avg_pois": avg_pois,
        "avg_clean_mean": avg_clean_mean,
        "avg_pois_mean": avg_pois_mean,
    }


def plot_1d(results, epsilon, delta, a0, a1, show_theoretical):
    """Plot 1D results."""
    plt.figure(figsize=(12, 8))
    
    if results["avg_clean_trajectory"]:
        plt.plot(results["avg_clean_trajectory"], label='RGD (Clean)', 
                linewidth=3, color='blue')
    
    if results["avg_poisoned_trajectory"]:
        plt.plot(results["avg_poisoned_trajectory"], 
                label=f'RGD + Naive Mean Shift (ε={epsilon}, δ={delta})', 
                linewidth=3, color='red')
    
    if show_theoretical:
        plt.axhline(y=results["theta_opt"], color='g', linestyle='dashed', 
                   linewidth=2, label='θOPT (Theoretical)')
        plt.axhline(y=results["theta_stab"], color='orange', linestyle='dashed', 
                   linewidth=2, label='θSTAB (Theoretical)')
    
    plt.legend(fontsize=12)
    plt.title('Naive Black-Box Poisoning (TM3) vs Clean RGD\nAdversary only knows data samples', fontsize=14)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel(r'$\theta$', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Plot means
    plt.figure(figsize=(12, 6))
    if any(v is not None for v in results["avg_clean_mean_trajectory"]):
        plt.plot([v for v in results["avg_clean_mean_trajectory"] if v is not None], 
                 label='Mean (Clean)', linewidth=3, color='blue')
    if results["avg_poisoned_mean_trajectory"] and any(v is not None for v in results["avg_poisoned_mean_trajectory"]):
        plt.plot([v for v in results["avg_poisoned_mean_trajectory"] if v is not None], 
                 label=f"Mean (Poisoned, ε={epsilon}, δ={delta})", linewidth=3, color='red')
    plt.title('Estimated Sample Mean over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Estimated mean')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.show()


def plot_2d(results, epsilon, delta):
    """Plot 2D results."""
    # Plot trajectory in theta space (theta1 vs theta2)
    plt.figure(figsize=(10, 10))
    
    # Clean trajectory
    clean_traj = results["avg_clean"].cpu()
    plt.plot(clean_traj[:, 0], clean_traj[:, 1], 
             label='Clean', color='blue', linewidth=2, marker='o', markersize=4)
    # Mark start point
    plt.plot(clean_traj[0, 0], clean_traj[0, 1], 
             'o', color='blue', markersize=8, label='Start (clean)')
    # Mark end point
    plt.plot(clean_traj[-1, 0], clean_traj[-1, 1], 
             's', color='blue', markersize=8, label='End (clean)')
    
    # Poisoned trajectory
    if results["avg_pois"] is not None:
        pois_traj = results["avg_pois"].cpu()
        plt.plot(pois_traj[:, 0], pois_traj[:, 1], 
                 label='Poisoned', color='red', linewidth=2, marker='^', markersize=4)
        # Mark start point
        plt.plot(pois_traj[0, 0], pois_traj[0, 1], 
                 'o', color='red', markersize=8, label='Start (poisoned)')
        # Mark end point
        plt.plot(pois_traj[-1, 0], pois_traj[-1, 1], 
                 's', color='red', markersize=8, label='End (poisoned)')
    
    plt.xlabel(r'$\theta_1$', fontsize=14)
    plt.ylabel(r'$\theta_2$', fontsize=14)
    plt.title(f'2D Trajectory: RGD with myopic mean shift\n(ε={epsilon}, δ={delta})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.axis('equal')
    plt.show()
    
    # Plot trajectory of estimated sample means (E[z1] vs E[z2])
    plt.figure(figsize=(10, 10))
    
    # Clean mean trajectory
    clean_mean = results["avg_clean_mean"]
    plt.plot(clean_mean[:, 0], clean_mean[:, 1], 
             label='Clean', color='blue', linewidth=2, marker='o', markersize=4)
    # Mark start point
    plt.plot(clean_mean[0, 0], clean_mean[0, 1], 
             'o', color='blue', markersize=8, label='Start (clean)')
    # Mark end point
    plt.plot(clean_mean[-1, 0], clean_mean[-1, 1], 
             's', color='blue', markersize=8, label='End (clean)')
    
    # Poisoned mean trajectory
    if results["avg_pois_mean"] is not None:
        pois_mean = results["avg_pois_mean"]
        plt.plot(pois_mean[:, 0], pois_mean[:, 1], 
                 label='Poisoned', color='red', linewidth=2, marker='^', markersize=4)
        # Mark start point
        plt.plot(pois_mean[0, 0], pois_mean[0, 1], 
                 'o', color='red', markersize=8, label='Start (poisoned)')
        # Mark end point
        plt.plot(pois_mean[-1, 0], pois_mean[-1, 1], 
                 's', color='red', markersize=8, label='End (poisoned)')
    
    plt.xlabel(r'$E[z_1]$', fontsize=14)
    plt.ylabel(r'$E[z_2]$', fontsize=14)
    plt.title(f'Estimated Sample Mean Trajectory: RGD with myopic mean shift\n(ε={epsilon}, δ={delta})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.axis('equal')
    plt.show()

