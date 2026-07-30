"""Weight-norm-driven blur: an IMAGE-SPACE performative map.

PerfFL's map (fmnist/performativity.py) reacts to the model by reweighting the
loss. This one reacts by degrading the images themselves:

    s_t = clamp( alpha * (1 - ||theta_0|| / ||theta_t||),  0, 1 )
    x_t = (1 - s_t) * x  +  s_t * blur(x)

s = 0 is the clean pool, s = 1 is the fully blurred pool, and everything in
between is a smooth dial. alpha = 0 -> s = 0 -> images untouched -> the
NON-performative control, same role alpha = 0 plays in the reweighting map.

The strength is measured RELATIVE TO THE INITIAL WEIGHT NORM, so there is no
magic constant to tune: s reads as "how much has the weight norm grown since
init," scaled by alpha.

--------------------------------------------------------------------------
CAVEAT -- read this before interpreting results
--------------------------------------------------------------------------
||theta|| is a RATCHET. It drifts upward under cross-entropy + SGD (roughly
log(t) for separable data) and blurring the images does not push it back down.
So this driver has no restoring force: expect `shift_strength` to be a
monotone RAMP, not an oscillation.

That makes this a distribution shift that happens to be indexed by the model,
rather than a full feedback loop. It is still a useful thing to look at -- it
isolates "does a progressively harder task make the backdoor easier or harder
to install?" -- but it is NOT the same phenomenon as the accuracy-driven map.

Always plot `shift_strength` over rounds first. If it ramps monotonically, you
have shift-without-feedback. To get a genuine loop, swap the driver for one
with a restoring force (gradient norm: blur the data and it spikes
immediately) -- only `strength_fn` below needs to change.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# The transform (actuator)
# --------------------------------------------------------------------------- #
def gaussian_kernel(size: int = 9, sigma: float = 2.0) -> torch.Tensor:
    """Separable 2-D Gaussian, normalized to sum 1, shaped (1, 1, size, size)."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    return torch.outer(g, g).view(1, 1, size, size)


def make_blurred(x: torch.Tensor, size: int = 9, sigma: float = 2.0) -> torch.Tensor:
    """Precompute the s = 1 endpoint once, so each round is just a lerp.

    A single 3x3 box blur is far too weak to give real dynamic range; sigma=2
    on 28x28 genuinely destroys fine detail, which is what you want s = 1 to
    mean.
    """
    return F.conv2d(x, gaussian_kernel(size, sigma), padding=size // 2)


def blur_mix(x: torch.Tensor, blurred: torch.Tensor, strength: float) -> torch.Tensor:
    """Linear interpolation clean -> blurred. strength in [0, 1]."""
    if strength <= 0.0:
        return x
    return (1.0 - strength) * x + strength * blurred


# --------------------------------------------------------------------------- #
# The driver
# --------------------------------------------------------------------------- #
@torch.no_grad()
def weight_norm(model) -> float:
    """L2 norm of all parameters flattened together."""
    return torch.sqrt(sum((p.detach() ** 2).sum() for p in model.parameters())).item()


def strength_fn(model, alpha: float, init_norm: float) -> float:
    """Map the model's weight-norm growth to a blur strength in [0, 1].

    s = alpha * (1 - ||theta_0|| / ||theta_t||)

    At init  (||theta_t|| == ||theta_0||)  -> s = 0.
    As the norm grows, s rises toward alpha and saturates at 1.
    """
    if alpha == 0.0:
        return 0.0
    now = weight_norm(model)
    if now <= 0.0:
        return 0.0
    return float(min(max(alpha * (1.0 - init_norm / now), 0.0), 1.0))


def make_shift_fn(blurred: torch.Tensor, alpha: float, init_norm: float):
    """Bundle the map into the callable run_rgd expects.

    Returns shift_fn(model, x, y) -> (x_shifted, strength). run_rgd applies
    this to the CLEAN pool each round, then hands the result to poison_fn --
    so the attacker stamps its trigger AFTER the performative response, which
    keeps the trigger pristine (the attacker controls its own samples; the
    population's reaction does not smear them).
    """
    def shift_fn(model, x, y):
        s = strength_fn(model, alpha, init_norm)
        return blur_mix(x, blurred, s), s

    return shift_fn
