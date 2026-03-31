"""Strategic manipulation of features and poisoning functions.

The best_response function (numpy) is used by strategic/optimization.py.

For poisoning in the strategic setting, the framework is:
    1. D_theta produces honest best-response for ALL agents
    2. make_strategic_poison() creates a poison_function compatible with RGD/PerfGD
       that selects epsilon fraction as adversarial, applies an attack, and clamps
       the result to an Linf delta-ball around the honest position.

Attack functions have the signature:
    attack_fn(z, adv_idx, strat_features, theta, **kwargs) -> x_desired
    where x_desired is (n_strat, n_adv) tensor of desired strategic feature values.
"""

import numpy as np
import torch
from typing import Callable, Optional
from functools import partial
from sklearn.linear_model import LogisticRegression
import warnings


# ---------------------------------------------------------------------------
# Original numpy best-response (used by strategic/optimization.py)
# ---------------------------------------------------------------------------

def best_response(X, theta, epsilon, strat_features):
    """Best response function for agents given classifier theta. Assumes linear utilities and quadratic costs.

    Parameters
    ----------
        X: np.array
            training data matrix
        theta: np.array
            deployed parameter vector
        epsilon: float
            sensitivity parameter, strength of performative effects
        strat_features: list
            list of features that can be manipulated strategically, other features remain fixed

    Returns
    -------
        X_strat: np.array
            modified training data matrix after each agents best responds to the classifier
    """

    n = X.shape[0]

    X_strat = np.copy(X)

    for i in range(n):
        # move everything by epsilon in the direction towards better classification
        theta_strat = theta[strat_features]
        X_strat[i, strat_features] += -epsilon * theta_strat

    return X_strat


# ---------------------------------------------------------------------------
# Linf clamping utility
# ---------------------------------------------------------------------------

def clamp_to_linf_ball(x_adv, x_honest, delta):
    """Clamp adversarial features to Linf ball of radius delta around honest position.

    Args:
        x_adv    : (n_strat, n_adv) desired adversarial positions
        x_honest : (n_strat, n_adv) honest best-response positions
        delta    : Linf radius (scalar or per-feature tensor)

    Returns:
        x_clamped : (n_strat, n_adv) clamped positions
    """
    return torch.clamp(x_adv, min=x_honest - delta, max=x_honest + delta)


# ---------------------------------------------------------------------------
# Strategic poison factory
# ---------------------------------------------------------------------------

def make_strategic_poison(
    attack_fn: Callable,
    strat_features: list,
    epsilon: float,
    delta: float,
    selection: str = "random",
    **attack_kwargs
):
    """Create a poison_function compatible with RGD/PerfGD.

    The returned function:
        1. Selects epsilon fraction of agents as adversarial
        2. Calls attack_fn to compute desired adversarial positions
        3. Clamps result to Linf delta-ball around honest position

    Args:
        attack_fn       : function(z, adv_idx, strat_features, theta, **kwargs) -> (n_strat, n_adv)
                           returns desired strategic feature values for adversarial agents
        strat_features  : list of strategic feature indices
        epsilon         : fraction of agents that are adversarial (0 to 1)
        delta           : Linf radius around honest position
        selection       : "random" (default) or "per_class" (epsilon fraction from each class)
        **attack_kwargs : additional kwargs passed to attack_fn

    Returns:
        poison_function(z, theta, **kwargs) -> z_poisoned
        compatible with RGD/PerfGD interface
    """
    strat_idx = torch.tensor(strat_features, dtype=torch.long)

    def poison_function(z, theta, **kwargs):
        n = z.shape[1]
        poisoned = z.clone()

        y = z[-1, :]  # labels
        x_honest_strat = z[strat_idx, :]  # (n_strat, n)

        # Select adversarial agents
        if selection == "random":
            n_adv = int(epsilon * n)
            if n_adv == 0:
                return poisoned
            adv_idx = torch.randperm(n)[:n_adv]

        elif selection == "per_class":
            # Epsilon fraction from each class separately
            idx_0 = (y == 0).nonzero(as_tuple=True)[0]
            idx_1 = (y == 1).nonzero(as_tuple=True)[0]
            n_adv_0 = int(epsilon * len(idx_0))
            n_adv_1 = int(epsilon * len(idx_1))
            if n_adv_0 + n_adv_1 == 0:
                return poisoned
            adv_idx_0 = idx_0[torch.randperm(len(idx_0))[:n_adv_0]]
            adv_idx_1 = idx_1[torch.randperm(len(idx_1))[:n_adv_1]]
            adv_idx = torch.cat([adv_idx_0, adv_idx_1])
        else:
            raise ValueError(f"Unknown selection mode: {selection}")

        # Get honest positions for adversarial agents
        x_honest = x_honest_strat[:, adv_idx]  # (n_strat, n_adv)

        # Compute desired adversarial positions
        x_desired = attack_fn(
            z, adv_idx, strat_features, theta, **attack_kwargs
        )  # (n_strat, n_adv)

        # Clamp to Linf ball
        x_clamped = clamp_to_linf_ball(x_desired, x_honest, delta)

        # Write back
        for i, s in enumerate(strat_features):
            poisoned[s, adv_idx] = x_clamped[i]

        return poisoned

    return poison_function


# ---------------------------------------------------------------------------
# Attack functions
#
# Each takes: (z, adv_idx, strat_features, theta, **kwargs)
# Returns: (n_strat, n_adv) tensor of desired strategic feature values
# ---------------------------------------------------------------------------

def cluster_shift_attack(z, adv_idx, strat_features, theta, **kwargs):
    """Shift adversarial agents' strategic features toward the opposing class mean.

    Each adversarial agent in class 0 targets mu_1, and vice versa.

    Returns:
        x_desired : (n_strat, n_adv) desired positions for adversarial agents
    """
    strat_idx = torch.tensor(strat_features, dtype=torch.long)
    y = z[-1, :]
    x_strat = z[strat_idx, :]  # (n_strat, n_samples)

    # Compute class means from ALL agents (honest positions)
    idx_0 = (y == 0).nonzero(as_tuple=True)[0]
    idx_1 = (y == 1).nonzero(as_tuple=True)[0]

    mu_0 = x_strat[:, idx_0].mean(dim=1) if len(idx_0) > 0 else x_strat.mean(dim=1)
    mu_1 = x_strat[:, idx_1].mean(dim=1) if len(idx_1) > 0 else x_strat.mean(dim=1)

    # For each adversarial agent, target the opposite class mean
    x_current = x_strat[:, adv_idx]  # (n_strat, n_adv)
    y_adv = y[adv_idx]               # (n_adv,)

    # Desired = opposite class mean (broadcast to all adversarial agents)
    x_desired = x_current.clone()
    mask_0 = (y_adv == 0)
    mask_1 = (y_adv == 1)

    if mask_0.any():
        x_desired[:, mask_0] = mu_1.unsqueeze(1)  # class 0 agents target mu_1
    if mask_1.any():
        x_desired[:, mask_1] = mu_0.unsqueeze(1)  # class 1 agents target mu_0

    return x_desired


def shadow_model_attack(z, adv_idx, strat_features, theta, shift_scale=10.0, **kwargs):
    """Train a shadow logistic regression on observed data and shift adversarial
    agents' strategic features toward the decision boundary.

    The shadow model is trained on ALL features (not just strategic ones), giving
    the attacker a reasonable estimate of the decision boundary. Adversarial agents
    are then shifted in strategic feature space toward the boundary — agents on
    the "correct" side are pushed across, confusing the learner.

    Args:
        z              : (d+1, n) features + labels
        adv_idx        : indices of adversarial agents
        strat_features : list of strategic feature column indices
        theta          : current model params (unused, shadow model is independent)
        shift_scale    : magnitude multiplier for shift toward boundary
                         (large value = desired position is far past boundary;
                          Linf clamping in make_strategic_poison limits actual shift)
        **kwargs       : ignored

    Returns:
        x_desired : (n_strat, n_adv) desired strategic feature values
    """
    strat_idx = torch.tensor(strat_features, dtype=torch.long)

    # Extract all features and labels
    X_all = z[:-1, :].T.cpu().numpy()   # (n, d)
    y_all = z[-1, :].cpu().numpy()      # (n,)

    x_strat = z[strat_idx, :]  # (n_strat, n)
    x_current = x_strat[:, adv_idx]  # (n_strat, n_adv)

    # Need both classes to train
    unique_classes = np.unique(y_all)
    if len(unique_classes) < 2:
        warnings.warn("shadow_model_attack: only one class present, returning current positions")
        return x_current

    # Train shadow logistic regression on ALL features
    try:
        lr = LogisticRegression(random_state=0, max_iter=10000)
        lr.fit(X_all, y_all)
    except Exception as e:
        warnings.warn(f"shadow_model_attack: LR training failed ({e}), returning current positions")
        return x_current

    w_full = lr.coef_[0]  # (d,)

    # Extract the shadow model's weight components for strategic features only
    w_strat = np.array([w_full[s] for s in strat_features])  # (n_strat,)
    w_strat_norm = np.linalg.norm(w_strat) + 1e-12

    # Compute each adversarial agent's signed distance to the boundary
    # using the full feature vector
    X_adv = X_all[adv_idx.cpu().numpy()]  # (n_adv, d)
    distances = lr.decision_function(X_adv)  # (n_adv,)

    # Shift direction in strategic feature space: toward the boundary
    # -sign(distance) * w_strat / ||w_strat|| pushes toward the hyperplane
    # Multiply by shift_scale to overshoot — Linf clamping will limit actual movement
    shift_dir = -np.sign(distances)[:, None] * (w_strat / w_strat_norm)[None, :]  # (n_adv, n_strat)
    shift = shift_scale * np.abs(distances)[:, None] * shift_dir  # scale by distance so closer agents shift less

    x_current_np = x_current.cpu().numpy()  # (n_strat, n_adv)
    x_desired_np = x_current_np + shift.T   # (n_strat, n_adv)

    return torch.tensor(x_desired_np, dtype=z.dtype, device=z.device)


