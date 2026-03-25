"""Strategic manipulation of features and poisoning functions.

The best_response function (numpy) is used by strategic/optimization.py.
The poison functions (torch) follow the poison_function interface used by
RGD/PerfGD in utils/algorithms.py:
    poison_function(z, theta, eta=..., loss=..., proj_theta=..., **kwargs) -> z_poisoned
"""

import numpy as np
import torch
from typing import Callable


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
# Cluster shift poisoning (torch, compatible with RGD/PerfGD)
# ---------------------------------------------------------------------------

def strategic_cluster_shift(
    z: torch.Tensor,
    theta: torch.Tensor,
    strat_features: list = None,
    epsilon: float = 0.3,
    delta: float = 1.0,
    **kwargs
):
    """
    Cluster shift poisoning for strategic classification.

    Adapted from black_box_cluster: shifts epsilon fraction of each class's
    strategic features toward the other class's mean in strategic feature space.
    Only modifies strategic features; labels are never changed.

    Selects the points closest to the opposing cluster mean (in strategic
    feature space) so the shift has maximum impact on the decision boundary.

    Args:
        z              : (d+1, n) features (first d rows) + labels (last row)
        theta          : current model parameters (unused, kept for interface)
        strat_features : list of strategic feature indices in z
        epsilon        : fraction of each class to poison
        delta          : shift magnitude
        **kwargs       : additional args from RGD/PerfGD (eta, loss, proj_theta, etc.)

    Returns:
        z_poisoned : (d+1, n) with only strategic features modified
    """
    if strat_features is None:
        raise ValueError("strat_features is required")

    strat_idx = torch.tensor(strat_features, dtype=torch.long)
    poisoned = z.clone()

    y = z[-1, :]  # labels
    x_strat = z[strat_idx, :]  # (n_strat, n_samples)

    idx_0 = (y == 0).nonzero(as_tuple=True)[0]
    idx_1 = (y == 1).nonzero(as_tuple=True)[0]

    if len(idx_0) == 0 or len(idx_1) == 0:
        return poisoned

    # Cluster means in strategic feature space
    mu_0 = x_strat[:, idx_0].mean(dim=1)  # (n_strat,)
    mu_1 = x_strat[:, idx_1].mean(dim=1)  # (n_strat,)

    # Direction from cluster 0 to cluster 1
    diff = mu_1 - mu_0
    direction = diff / (diff.norm() + 1e-12)
    shift = delta * direction  # (n_strat,)

    # Poison epsilon fraction of cluster 0: closest to mu_1, shift toward mu_1
    n_poison_0 = int(epsilon * len(idx_0))
    if n_poison_0 > 0:
        dists = torch.norm(x_strat[:, idx_0].T - mu_1, dim=1)
        _, closest = torch.topk(dists, n_poison_0, largest=False)
        poison_idx_0 = idx_0[closest]
        for feat_i, s in enumerate(strat_features):
            poisoned[s, poison_idx_0] += shift[feat_i]

    # Poison epsilon fraction of cluster 1: closest to mu_0, shift toward mu_0
    n_poison_1 = int(epsilon * len(idx_1))
    if n_poison_1 > 0:
        dists = torch.norm(x_strat[:, idx_1].T - mu_0, dim=1)
        _, closest = torch.topk(dists, n_poison_1, largest=False)
        poison_idx_1 = idx_1[closest]
        for feat_i, s in enumerate(strat_features):
            poisoned[s, poison_idx_1] -= shift[feat_i]

    return poisoned
