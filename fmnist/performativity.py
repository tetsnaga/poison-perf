"""PerfFL class-resampling performativity (copied, with an alpha knob).

The idea (from PerfFL's `adjust_weights_by_pi`): after the model is deployed,
measure its per-class accuracy. Classes the model is WORST at get up-weighted
in the next round's training loss. This makes the effective data distribution
react to the model, which is what "performativity" means here.

PerfFL hard-codes the strength at 0.5:

    temp = 1 - acc ; temp -= temp.min() ; weight = exp(0.5 * temp)

We expose that 0.5 as `alpha` so we can sweep it. alpha = 0 returns uniform
weights -> the distribution stops reacting to the model -> NON-performative.
"""

from __future__ import annotations

import torch


def class_weights_from_accuracy(per_class_acc: torch.Tensor, alpha: float) -> torch.Tensor:
    """PerfFL *paper's* softmax form, parameterized by alpha:

        pi_k = softmax_k( alpha * (1 - acc_k) )

    Classes the model is worst at (low acc -> large 1-acc) get the most weight.
    The result is NORMALIZED (sums to 1), so it doubles as the effective class
    mix for that round. alpha = 0 -> softmax(0) -> uniform 1/K (non-performative).

    Note: PerfFL's *code* used an unnormalized exp(0.5*(1-acc)); we follow the
    paper's softmax, which is normalized and numerically stable (no blow-up at
    large alpha).

    Args:
        per_class_acc: tensor of shape (n_classes,), accuracy per class.
        alpha:         performativity strength.
    Returns:
        per-class proportion tensor of shape (n_classes,), summing to 1.
    """
    return torch.softmax(alpha * (1.0 - per_class_acc), dim=0)


def sample_weights(labels: torch.Tensor, class_weights: torch.Tensor) -> torch.Tensor:
    """Expand per-class proportions to a per-sample loss weight.

    We scale by n_classes so the *mean* per-sample weight is ~1 (with balanced
    classes). Without this, the softmax weights (which sum to 1) would shrink
    every loss ~K-fold and quietly cut the effective learning rate. This is a
    pure magnitude choice -- the relative per-class weighting is unchanged, and
    alpha=0 recovers ordinary unweighted training (every weight = 1).
    """
    n_classes = class_weights.shape[0]
    return class_weights.to(labels.device)[labels] * n_classes
