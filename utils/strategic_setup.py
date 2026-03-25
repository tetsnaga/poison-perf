"""
Strategic classification setup with configurable classifiers and performativity models.

Classifiers: "logreg", "mlp" (future: "xgboost", "dnn")
Performativity models: "linear", "gradient", "matrix", "elementwise"

Valid combinations:
    logreg + linear:       x'_S = x_S - alpha * theta_S
    logreg + matrix:       x'_S = x_S - alpha * A @ theta_S
    logreg + elementwise:  x'_S = x_S - alpha * g(theta_S)
    logreg + gradient:     x'_S = x_S - alpha * grad_x_S P(y=1|x)
    mlp + gradient:        x'_S = x_S - alpha * grad_x_S P(y=1|x)
"""
import torch
import torch.nn.functional as F
from typing import Optional, Callable


CLASSIFIERS = ["logreg", "mlp"]  # future: "xgboost", "dnn"
PERFORMATIVITY_MODELS = ["linear", "gradient", "matrix", "elementwise"]

VALID_COMBINATIONS = {
    ("logreg", "linear"),
    ("logreg", "gradient"),
    ("logreg", "matrix"),
    ("logreg", "elementwise"),
    ("mlp", "gradient"),
}


# ---------------------------------------------------------------------------
# MLP helpers
# ---------------------------------------------------------------------------

def _mlp_forward(x, theta, input_dim, hidden_dims):
    """
    Functional MLP forward pass using slices of a flat parameter vector.
    All operations go through theta so autograd works for both x and theta.
    """
    offset = 0
    h = x
    prev = input_dim
    for hdim in hidden_dims:
        W = theta[offset:offset + prev * hdim].view(hdim, prev)
        offset += prev * hdim
        b = theta[offset:offset + hdim]
        offset += hdim
        h = F.relu(h @ W.T + b)
        prev = hdim
    W = theta[offset:offset + prev].view(1, prev)
    offset += prev
    b = theta[offset:offset + 1]
    return (h @ W.T + b).squeeze(-1)


def _mlp_param_count(input_dim, hidden_dims):
    """Total number of parameters for the MLP architecture."""
    count = 0
    prev = input_dim
    for hdim in hidden_dims:
        count += prev * hdim + hdim
        prev = hdim
    count += prev + 1
    return count


def _mlp_init(input_dim, hidden_dims):
    """Xavier-style initialization for the flat MLP parameter vector."""
    n_params = _mlp_param_count(input_dim, hidden_dims)
    theta = torch.zeros(n_params, dtype=torch.float32)
    torch.manual_seed(0)
    offset = 0
    prev = input_dim
    for hdim in hidden_dims:
        n_w = prev * hdim
        std = (2.0 / (prev + hdim)) ** 0.5
        theta[offset:offset + n_w] = torch.randn(n_w) * std
        offset += n_w + hdim  # biases stay zero
        prev = hdim
    n_w = prev
    std = (2.0 / (prev + 1)) ** 0.5
    theta[offset:offset + n_w] = torch.randn(n_w) * std
    return theta


# ---------------------------------------------------------------------------
# Main setup
# ---------------------------------------------------------------------------

def setup_strategic_classification(
        X_base: torch.Tensor,
        Y_base: torch.Tensor,
        strat_features: list,
        alpha: float = 1.0,
        lam: float = None,
        classifier: str = "logreg",
        hidden_dims: list = None,
        performativity: str = "linear",
        response_matrix: torch.Tensor = None,
        response_nonlinearity: Callable = None,
        response_nonlinearity_deriv: Callable = None,
        perfGD: bool = False
):
    """
    Strategic classification with configurable classifier and performativity model.

    Args:
        X_base       : (n_samples, d+1) standardized features with bias column appended
        Y_base       : (n_samples,) labels in {0, 1}
        strat_features : list of int, column indices of strategic features
        alpha        : sensitivity / manipulation strength
        lam          : L2 regularization (default: 1/n)
        classifier   : "logreg" | "mlp"
        hidden_dims  : MLP hidden layer sizes (default: [32, 16])
        performativity : "linear" | "gradient" | "matrix" | "elementwise"
        response_matrix : (|S|, |S|) matrix A for "matrix" performativity (default: identity)
        response_nonlinearity : elementwise function g for "elementwise" (default: tanh)
        response_nonlinearity_deriv : derivative g' for PerfGD (default: tanh derivative)
        perfGD       : if True, also return grad2_est and f_hat

    Returns (perfGD=False):
        mu_f, sigma, D_theta, loss, theta_0

    Returns (perfGD=True):
        mu_f, sigma, D_theta, loss, theta_0, grad2_est, f_hat
    """
    if (classifier, performativity) not in VALID_COMBINATIONS:
        raise ValueError(
            f"Invalid combination: classifier='{classifier}', performativity='{performativity}'. "
            f"Valid: {VALID_COMBINATIONS}"
        )

    if hidden_dims is None:
        hidden_dims = [32, 16]

    n_total, d_plus_1 = X_base.shape
    d = d_plus_1 - 1

    if lam is None:
        lam = 1.0 / n_total

    strat_idx = torch.tensor(strat_features, dtype=torch.long)
    n_strat = len(strat_features)

    # Empirical covariance of strategic features (used by PerfGD)
    X_strat_base = X_base[:, strat_idx]
    x_bar_S = X_strat_base.mean(dim=0)
    X_centered = X_strat_base - x_bar_S.unsqueeze(0)
    sigma = (X_centered.T @ X_centered) / (n_total - 1)
    sigma = sigma + 1e-6 * torch.eye(n_strat)

    # Defaults for matrix and elementwise performativity
    if response_matrix is None:
        A = torch.eye(n_strat, dtype=torch.float32)
    else:
        A = response_matrix

    if response_nonlinearity is None:
        g = torch.tanh
    else:
        g = response_nonlinearity

    if response_nonlinearity_deriv is None:
        g_prime = lambda t: 1.0 - torch.tanh(t) ** 2
    else:
        g_prime = response_nonlinearity_deriv

    # ---- Helper: batch selection ----
    def _select_batch(n):
        idx = torch.arange(n_total) if n >= n_total else torch.randperm(n_total)[:n]
        return X_base[idx].clone(), Y_base[idx]

    # ================================================================
    # D_theta: distribution mapping (performativity model)
    # ================================================================

    if performativity == "linear":
        def D_theta(theta, n):
            theta_d = theta.detach()
            X_batch, Y_batch = _select_batch(n)
            X_features = X_batch[:, :d]
            X_features[:, strat_idx] -= alpha * theta_d[strat_idx]
            return torch.cat([X_features.T, Y_batch.unsqueeze(0)], dim=0)

        def mu_f(theta):
            return x_bar_S - alpha * theta[strat_idx]

    elif performativity == "matrix":
        def D_theta(theta, n):
            theta_d = theta.detach()
            X_batch, Y_batch = _select_batch(n)
            X_features = X_batch[:, :d]
            X_features[:, strat_idx] -= alpha * (A @ theta_d[strat_idx])
            return torch.cat([X_features.T, Y_batch.unsqueeze(0)], dim=0)

        def mu_f(theta):
            return x_bar_S - alpha * (A @ theta[strat_idx])

    elif performativity == "elementwise":
        def D_theta(theta, n):
            theta_d = theta.detach()
            X_batch, Y_batch = _select_batch(n)
            X_features = X_batch[:, :d]
            X_features[:, strat_idx] -= alpha * g(theta_d[strat_idx])
            return torch.cat([X_features.T, Y_batch.unsqueeze(0)], dim=0)

        def mu_f(theta):
            return x_bar_S - alpha * g(theta[strat_idx])

    elif performativity == "gradient":
        if classifier == "logreg":
            def D_theta(theta, n):
                theta_d = theta.detach()
                X_batch, Y_batch = _select_batch(n)
                X_features = X_batch[:, :d]

                x_input = X_features.clone().requires_grad_(True)
                logits = theta_d[:-1] @ x_input.T + theta_d[-1]
                probs = torch.sigmoid(logits)
                grad_x = torch.autograd.grad(probs.sum(), x_input)[0]

                X_features[:, strat_idx] -= alpha * grad_x[:, strat_idx].detach()
                return torch.cat([X_features.T, Y_batch.unsqueeze(0)], dim=0)

        elif classifier == "mlp":
            def D_theta(theta, n):
                theta_d = theta.detach()
                X_batch, Y_batch = _select_batch(n)
                X_features = X_batch[:, :d]

                x_input = X_features.clone().requires_grad_(True)
                logits = _mlp_forward(x_input, theta_d, d, hidden_dims)
                probs = torch.sigmoid(logits)
                grad_x = torch.autograd.grad(probs.sum(), x_input)[0]

                X_features[:, strat_idx] -= alpha * grad_x[:, strat_idx].detach()
                return torch.cat([X_features.T, Y_batch.unsqueeze(0)], dim=0)

        def mu_f(theta):
            return x_bar_S  # no closed form for gradient-based shift

    # ================================================================
    # Loss function (classifier-dependent)
    # ================================================================

    if classifier == "logreg":
        theta_dim = d_plus_1

        def loss(z, theta):
            x = z[:-1, :]
            y = z[-1, :]
            logits = theta[:-1] @ x + theta[-1]
            bce = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
            reg = (lam / 2.0) * torch.sum(theta[:-1] ** 2)
            return bce + reg

        theta_0 = torch.zeros(theta_dim, dtype=torch.float32)

    elif classifier == "mlp":
        theta_dim = _mlp_param_count(d, hidden_dims)

        def loss(z, theta):
            x = z[:-1, :]
            y = z[-1, :]
            logits = _mlp_forward(x.T, theta, d, hidden_dims)
            bce = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
            reg = (lam / 2.0) * torch.sum(theta ** 2)
            return bce + reg

        theta_0 = _mlp_init(d, hidden_dims)

    # ================================================================
    # PerfGD components
    # ================================================================
    if perfGD:
        sigma_inv = torch.linalg.inv(sigma)

        def f_hat(z):
            """Sample mean of strategic features."""
            return z[strat_idx, :].mean(dim=1)

        if performativity == "linear":
            # Analytic Jacobian: df/dtheta_k = -alpha if k is a strategic feature
            df_dtheta_const = torch.zeros(n_strat, theta_dim)
            for i, s in enumerate(strat_features):
                df_dtheta_const[i, s] = -alpha

            def grad2_est(z, f, theta, df_d_theta):
                if f.ndim == 1:
                    f = f.unsqueeze(1)
                x_strat = z[strat_idx, :]
                residuals = x_strat - f
                loss_vals = loss(z, theta)
                weighted_resid = (loss_vals.unsqueeze(0) * residuals).mean(dim=1)
                return df_dtheta_const.T @ sigma_inv @ weighted_resid

        elif performativity == "matrix":
            # Analytic Jacobian: d(mu_f_i)/d(theta_k) = -alpha * A[i,j] when k = strat_features[j]
            df_dtheta_const = torch.zeros(n_strat, theta_dim)
            for i in range(n_strat):
                for j in range(n_strat):
                    df_dtheta_const[i, strat_features[j]] = -alpha * A[i, j]

            def grad2_est(z, f, theta, df_d_theta):
                if f.ndim == 1:
                    f = f.unsqueeze(1)
                x_strat = z[strat_idx, :]
                residuals = x_strat - f
                loss_vals = loss(z, theta)
                weighted_resid = (loss_vals.unsqueeze(0) * residuals).mean(dim=1)
                return df_dtheta_const.T @ sigma_inv @ weighted_resid

        elif performativity == "elementwise":
            # Theta-dependent Jacobian: d(mu_f_i)/d(theta_k) = -alpha * g'(theta_k) when k = strat_features[i]
            def grad2_est(z, f, theta, df_d_theta):
                df_dtheta = torch.zeros(n_strat, theta_dim)
                g_deriv = g_prime(theta[strat_idx])
                for i, s in enumerate(strat_features):
                    df_dtheta[i, s] = -alpha * g_deriv[i]

                if f.ndim == 1:
                    f = f.unsqueeze(1)
                x_strat = z[strat_idx, :]
                residuals = x_strat - f
                loss_vals = loss(z, theta)
                weighted_resid = (loss_vals.unsqueeze(0) * residuals).mean(dim=1)
                return df_dtheta.T @ sigma_inv @ weighted_resid

        elif performativity == "gradient":
            # No closed-form Jacobian; use finite-difference from PerfGD history
            def grad2_est(z, f, theta, df_d_theta):
                if f.ndim == 1:
                    f = f.unsqueeze(1)
                x_strat = z[strat_idx, :]
                residuals = x_strat - f
                loss_vals = loss(z, theta)
                weighted_resid = (loss_vals.unsqueeze(0) * residuals).mean(dim=1)
                return df_d_theta.T @ sigma_inv @ weighted_resid

        return mu_f, sigma, D_theta, loss, theta_0, grad2_est, f_hat
    else:
        return mu_f, sigma, D_theta, loss, theta_0
