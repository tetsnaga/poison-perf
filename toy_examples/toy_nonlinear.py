import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import torch
from utils.algorithms import RGD, PerfGD

def run_nonlinear():
    a0 = 1.0
    a1 = 1.0
    sigma = torch.tensor([[1.0]])
    mu = lambda theta: torch.sqrt(a0*theta + a1) # Define mean function (determines performativity)
    f_hat = lambda z: z.mean(dim=1)
    D_theta = lambda theta, n: torch.normal(mu(theta).squeeze(), 1.0, size=(1,n))
    loss = lambda z, theta: z*theta
    def grad2_est(z, f, theta, df_d_theta):
        if z.ndim == 1: z = z.unsqueeze(0)
        if f.ndim == 1: f = f.unsqueeze(0)
        return torch.mean( loss(z, theta) * (df_d_theta.T @ torch.linalg.inv(sigma) @ (z - f)), dim=-1)

    proj_theta = lambda x: x.clamp(min=-1.0, max=1.0)

    # Run 5 trials
    num_trials = 5
    all_rgd_trajectories = []
    all_perfgd_trajectories = []
    rgd_final_thetas = []
    perfgd_final_thetas = []

    print(f"Running {num_trials} trials...")

    for trial in range(num_trials):
        print(f"Trial {trial + 1}/{num_trials}")
        
        # Run RGD
        theta_rgd, all_theta_rgd = RGD(
            D_theta,
            loss,
            theta_0 = torch.tensor([0.0]),
            proj_theta = proj_theta,
            n = 500,
            eta = 0.1,
            tol = 1e-5,
            max_iter = 30
        )

        # Run PerfGD
        theta_perfgd, all_theta_perfgd = PerfGD(
            f_hat,
            grad2_est,
            D_theta,
            loss,
            theta_0 = torch.tensor([0.0]),
            proj_theta = proj_theta,
            n = 500,
            eta = 0.1,
            warmup = 1,
            tol = 1e-5,
            max_iter = 30
        )
        
        # Store results
        all_rgd_trajectories.append(all_theta_rgd)
        all_perfgd_trajectories.append(all_theta_perfgd)
        rgd_final_thetas.append(theta_rgd.item())
        perfgd_final_thetas.append(theta_perfgd.item())

    print(f"All trials completed!")

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
    if len(avg_rgd_trajectory) > 0:
        plt.plot(avg_rgd_trajectory, label='RGD (Average)', linewidth=3, color='purple')
    else:
        print("Warning: RGD average trajectory is empty")

    if len(avg_perfgd_trajectory) > 0:
        plt.plot(avg_perfgd_trajectory, label='PerfGD (Average)', linewidth=3, color='orange')
    else:
        print("Warning: PerfGD average trajectory is empty")

    # Plot individual trajectories with transparency
    for i, traj in enumerate(all_rgd_trajectories):
        if len(traj) > 0:
            plt.plot(torch.stack(traj), alpha=0.3, color='purple', linewidth=1)

    for i, traj in enumerate(all_perfgd_trajectories):
        if len(traj) > 0:
            plt.plot(torch.stack(traj), alpha=0.3, color='orange', linewidth=1)

    # Add theoretical solutions
    theta_opt = -2*a0/(3*a1)
    theta_stab = -a0/a1

    plt.axhline(y=theta_opt, color='r', linestyle='dashed', linewidth=2, label='θOPT (Theoretical)')
    plt.axhline(y=theta_stab, color='b', linestyle='dashed', linewidth=2, label='θSTAB (Theoretical)')

    plt.legend(fontsize=12)
    plt.title('RGD vs PerfGD on Performative Mean Estimation (5 Trials Average)', fontsize=14)
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


run_nonlinear()