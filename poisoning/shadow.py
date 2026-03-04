"""
Shadow model poisoning using logistic regression.
"""
import torch
import numpy as np
from typing import Callable
from sklearn.linear_model import LogisticRegression
import warnings


def black_box_shadow_logreg(
    z: torch.Tensor,
    theta: torch.Tensor,
    eta: float = 0.1,
    proj_theta: Callable = lambda x: x,
    delta: float = 100.0,
    epsilon: float = 0.5,
    norm: str = 'linf',  # 'l2' or 'linf'
    **kwargs
) -> torch.Tensor:
    """
    Shadow Model Black Box Adversary using Logistic Regression
    
    Trains a logistic regression shadow model on the data, finds points closest to the
    decision boundary, and shifts them towards the boundary.
    
    Args:
        z: Data tensor (2, n_samples) - features in first row, labels in second row
        theta: Model parameters (not used by shadow model, kept for compatibility)
        eta: Learning rate (not used, kept for compatibility)
        proj_theta: Projection function (not used, kept for compatibility)
        delta: Shift magnitude
        epsilon: Fraction of samples to poison
        norm: 'l2' or 'linf' norm for shifts
        **kwargs: Additional arguments (ignored)
        
    Returns:
        Poisoned data tensor (same shape as z)
    """
    assert z.shape[0] == 2, "z must be 1D + label"
    
    n_samples = z.shape[1]
    n_poison = int(epsilon * n_samples)
    
    if n_poison == 0:
        return z
    
    # Clone to avoid modifying input
    poisoned_samples = z.clone()
    
    # Extract features and labels
    x = z[0, :].cpu().numpy()  # (n_samples,)
    y = z[1, :].cpu().numpy()  # (n_samples,)
    
    # Require both classes for logistic regression
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        warnings.warn(
            f"black_box_shadow_logreg: Only one class present (classes: {unique_classes}). Returning original data.",
            UserWarning,
            stacklevel=2,
        )
        return z
    
    # Reshape for sklearn (needs (n_samples, n_features))
    X = x.reshape(-1, 1)  # (n_samples, 1)
    
    # Train logistic regression shadow model
    try:
        lr = LogisticRegression(random_state=0, max_iter=10000)
        lr.fit(X, y)
        
        w = lr.coef_[0][0]  # scalar
        b = lr.intercept_[0]  # scalar
        
        # Prevent division by zero
        boundary = -b / (w + np.sign(w) * 1e-16)
        
        # Get distances to boundary (decision function)
        distances = lr.decision_function(X).flatten()  # (n_samples,)
        
        # Alternative: use probability-based confidence (equivalent for logistic regression)
        # probs = lr.predict_proba(X)[:, 1]  # P(y=1|x)
        # confidence = np.abs(probs - 0.5)  # Lower confidence = closer to 0.5
        # closest_indices = np.argsort(confidence)[:n_poison]  # Lowest confidence first
        
        # Find n_poison points closest to boundary (smallest absolute distance)
        # This is equivalent to selecting lowest confidence samples
        closest_indices = np.argsort(np.abs(distances))[:n_poison]
        
    except Exception as e:
        # If training fails, return original data
        warnings.warn(f"black_box_shadow_logreg: Logistic regression training failed: {str(e)}. Returning original data.", UserWarning)
        return z
    
    # Untargeted attack: shift points towards boundary
    for i in closest_indices:
        x_i = x[i]
        direction_to_boundary = boundary - x_i
        
        if norm == 'l2':
            shift_direction = direction_to_boundary / (np.abs(direction_to_boundary) + 1e-16)
        else:  # linf
            shift_direction = np.sign(direction_to_boundary)
        
        shift = delta * shift_direction
        poisoned_samples[0, i] = poisoned_samples[0, i] + shift
    
    return poisoned_samples



def black_box_shadow_logreg_nd(
    z: torch.Tensor,
    theta: torch.Tensor,
    eta: float = 0.1,
    proj_theta: Callable = lambda x: x,
    delta: float = 100.0,
    epsilon: float = 0.5,
    norm: str = 'linf',  # 'l2' or 'linf'
    **kwargs
) -> torch.Tensor:
    """
    Shadow Model Black Box Adversary using Logistic Regression (n-dimensional features).
    
    For use with setup_binary_classification_nd. Expects z of shape (dim+1, n_samples):
    first dim rows are features, last row is labels. Trains LR, finds points closest
    to the decision hyperplane, and shifts them towards it.
    
    Args:
        z: Data tensor (dim+1, n_samples) - dim feature rows, then label row
        theta: Model parameters (not used by shadow model, kept for compatibility)
        eta: Learning rate (not used, kept for compatibility)
        proj_theta: Projection function (not used, kept for compatibility)
        delta: Shift magnitude
        epsilon: Fraction of samples to poison
        norm: 'l2' or 'linf' norm for shifts
        **kwargs: Additional arguments (ignored)
        
    Returns:
        Poisoned data tensor (same shape as z)
    """
    assert z.shape[0] >= 2, "z must have at least 1 feature row + label row"
    
    n_samples = z.shape[1]
    n_poison = int(epsilon * n_samples)
    
    if n_poison == 0:
        return z
    
    poisoned_samples = z.clone()
    
    # Features (dim, n) -> (n, dim); labels (n,)
    X = z[:-1, :].cpu().numpy().T   # (n_samples, n_features)
    y = z[-1, :].cpu().numpy()      # (n_samples,)
    
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        warnings.warn(
            f"black_box_shadow_logreg_nd: Only one class present (classes: {unique_classes}). Returning original data.",
            UserWarning,
            stacklevel=2,
        )
        return z
    
    try:
        lr = LogisticRegression(random_state=0, max_iter=10000)
        lr.fit(X, y)
        
        w = np.asarray(lr.coef_[0], dtype=np.float64).flatten()  # (n_features,)
        distances = lr.decision_function(X).flatten()  # (n_samples,)
        closest_indices = np.argsort(np.abs(distances))[:n_poison]
        
    except Exception as e:
        warnings.warn(
            f"black_box_shadow_logreg_nd: Logistic regression training failed: {str(e)}. Returning original data.",
            UserWarning,
            stacklevel=2,
        )
        return z
    
    # Guard against degenerate/non-finite coefficients (would cause NaN in shifts)
    if not np.all(np.isfinite(w)):
        warnings.warn(
            "black_box_shadow_logreg_nd: Non-finite coefficients from LR. Returning original data.",
            UserWarning,
            stacklevel=2,
        )
        return z
    
    # Prevent division by zero when normalizing w (nD has no scalar "boundary" like 1D's -b/w)
    w_norm = np.linalg.norm(w) + 1e-16
    
    for i in closest_indices:
        # Direction towards hyperplane: -sign(distance) * w / ||w||
        if norm == 'l2':
            shift_direction = -np.sign(distances[i]) * (w / w_norm)
        else:
            shift_direction = -np.sign(distances[i]) * np.sign(w)
        
        shift = delta * np.asarray(shift_direction, dtype=np.float64)
        poisoned_samples[:-1, i] = poisoned_samples[:-1, i] + torch.tensor(
            shift, dtype=poisoned_samples.dtype, device=poisoned_samples.device
        )
    
    return poisoned_samples