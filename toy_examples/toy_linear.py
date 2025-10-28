import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import torch
from utils.algorithms import RGD, PerfGD

def run_linear():
    # Mixture of Gaussians example with 5 trials
    # Parameters from PerfGD paper Appendix E.1
    gamma = 0.5
    sigma1_sq = 1.0
    sigma2_sq = 0.25
    a1_0 = -0.5
    a1_1 = 1.0
    a2_0 = 1.0
    a2_1 = -0.3

    # Component means as functions of theta
    mu1 = lambda theta: a1_0 + a1_1 * theta  # First component mean
    mu2 = lambda theta: a2_0 + a2_1 * theta  # Second component mean

    # Define the mixture distribution
    def D_theta_mixture(theta, n):
        # Sample from mixture of Gaussians
        n1 = int(gamma * n)
        n2 = n - n1
        
        # Sample from first component
        z1 = torch.normal(mu1(theta).squeeze(), torch.sqrt(torch.tensor(sigma1_sq)), size=(1, n1))
        # Sample from second component  
        z2 = torch.normal(mu2(theta).squeeze(), torch.sqrt(torch.tensor(sigma2_sq)), size=(1, n2))
        
        # Combine samples
        z = torch.cat([z1, z2], dim=1)
        return z

    # Estimator function (sample mean)
    f_hat_mixture = lambda z: z.mean(dim=1)

    # Loss function
    loss_mixture = lambda z, theta: z * theta

    # Gradient estimation function for PerfGD
    def grad2_est_mixture(z, f, theta, df_d_theta):
        if z.ndim == 1: z = z.unsqueeze(0)
        if f.ndim == 1: f = f.unsqueeze(0)
        # For mixture, we need to account for the mixture structure
        sigma_mixture = torch.tensor([[sigma1_sq]])  # Use first component variance
        return torch.mean(loss_mixture(z, theta) * (df_d_theta.T @ torch.linalg.inv(sigma_mixture) @ (z - f)), dim=-1)

    # Projection function
    proj_theta_mixture = lambda x: x.clamp(min=-1.0, max=1.0)

    # Run 5 trials
    num_trials = 5
    all_rgd_trajectories = []
    all_perfgd_trajectories = []
    rgd_final_thetas = []
    perfgd_final_thetas = []

    for trial in range(num_trials):        
        # Run RGD
        theta_rgd_mix, all_theta_rgd_mix = RGD(
            D_theta_mixture,
            loss_mixture,
            theta_0=torch.tensor([0.0]),
            proj_theta=proj_theta_mixture,
            n=1000,
            eta=0.1,  # Use proper learning rate
            tol=1e-5,
            max_iter=100
        )
        
        # Run PerfGD
        theta_perfgd_mix, all_theta_perfgd_mix = PerfGD(
            f_hat_mixture,
            grad2_est_mixture,
            D_theta_mixture,
            loss_mixture,
            theta_0=torch.tensor([0.0]),
            proj_theta=proj_theta_mixture,
            n=1000,
            eta=0.1,  # Use proper learning rate
            warmup=1,
            tol=1e-5,
            max_iter=100
        )
        
        # Store results
        all_rgd_trajectories.append(all_theta_rgd_mix)
        all_perfgd_trajectories.append(all_theta_perfgd_mix)
        rgd_final_thetas.append(theta_rgd_mix.item())
        perfgd_final_thetas.append(theta_perfgd_mix.item())

    # Compute average trajectories
    max_length = max(len(traj) for traj in all_rgd_trajectories + all_perfgd_trajectories)
    avg_rgd_trajectory = []
    avg_perfgd_trajectory = []

    for i in range(max_length):
        rgd_values = []
        perfgd_values = []
        
        for traj in all_rgd_trajectories:
            if i < len(traj):
                rgd_values.append(traj[i].item())
        
        for traj in all_perfgd_trajectories:
            if i < len(traj):
                perfgd_values.append(traj[i].item())
        
        if rgd_values:
            avg_rgd_trajectory.append(np.mean(rgd_values))
        if perfgd_values:
            avg_perfgd_trajectory.append(np.mean(perfgd_values))

    # Plot results
    plt.figure(figsize=(12, 8))

    # Plot average trajectories
    plt.plot(avg_rgd_trajectory, label='RGD (Average)', linewidth=3, color='purple')
    plt.plot(avg_perfgd_trajectory, label='PerfGD (Average)', linewidth=3, color='orange')

    # Plot individual trajectories with transparency
    for i, traj in enumerate(all_rgd_trajectories):
        if len(traj) > 0:
            plt.plot(torch.stack(traj), alpha=0.3, color='purple', linewidth=1)

    for i, traj in enumerate(all_perfgd_trajectories):
        if len(traj) > 0:
            plt.plot(torch.stack(traj), alpha=0.3, color='orange', linewidth=1)

    # Add theoretical solutions from paper Appendix E.1
    theta_opt = -0.5 * (gamma * a1_0 + (1 - gamma) * a2_0) / (gamma * a1_1 + (1 - gamma) * a2_1)
    theta_stab = -(gamma * a1_0 + (1 - gamma) * a2_0) / (gamma * a1_1 + (1 - gamma) * a2_1)

    plt.axhline(y=theta_opt, color='r', linestyle='dashed', linewidth=2, label='θOPT (Paper)')
    plt.axhline(y=theta_stab, color='b', linestyle='dashed', linewidth=2, label='θSTAB (Paper)')

    plt.legend(fontsize=12)
    plt.title('RGD vs PerfGD on Mixture of Gaussians (5 Trials Average)', fontsize=14)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel(r'$\theta$', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()

    # Print summary statistics
    print(f"\nSummary Statistics ({num_trials} trials):")
    print(f"RGD final theta - Mean: {np.mean(rgd_final_thetas):.6f}, Std: {np.std(rgd_final_thetas):.6f}")
    print(f"PerfGD final theta - Mean: {np.mean(perfgd_final_thetas):.6f}, Std: {np.std(perfgd_final_thetas):.6f}")
    print(f"Theoretical θOPT: {theta_opt:.6f}")
    print(f"Theoretical θSTAB: {theta_stab:.6f}")

run_linear()