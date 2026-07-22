"""Centralized RGD (repeated gradient descent) under performativity.

One round =
  1. Build this round's training data: if a `poison_fn` is given, re-poison the
     clean base (a fresh epsilon-fraction gets the trigger each round); else use
     the clean base as-is.
  2. Train on it, weighting the loss by the current per-class weights.
  3. Evaluate the deployed model: clean accuracy, per-class accuracy, and ASR.
  4. Update the per-class weights from the per-class accuracy (the PerfFL
     performative map). alpha = 0 keeps them uniform (non-performative).

The underlying image pool is FIXED across rounds; performativity changes only
the loss weighting -- exactly how PerfFL implements it (loss reweighting, not
literal resampling). The backdoor, if any, re-rolls WHICH samples carry the
trigger each round via `poison_fn`, so exactly epsilon are triggered every
round but a given image is not permanently poisoned. The trigger and target
stay fixed, so it is still a static (non-adaptive) attack.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fmnist.experiment_setup import N_CLASSES
from fmnist.performativity import class_weights_from_accuracy, sample_weights


@torch.no_grad()
def evaluate(model, test_x, test_y, device="cpu"):
    """Return (overall_accuracy, per_class_accuracy tensor)."""
    model.eval()
    pred = model(test_x.to(device)).argmax(dim=1).cpu()
    overall = (pred == test_y).float().mean().item()
    per_class = torch.zeros(N_CLASSES)
    for c in range(N_CLASSES):
        mask = test_y == c
        if mask.any():
            per_class[c] = (pred[mask] == c).float().mean()
    return overall, per_class


def _train_one_round(model, x, y, weights, lr, epochs, batch_size, device):
    """One RGD round: weighted-loss SGD passes over the data."""
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss(reduction="none")  # per-sample, so we can weight it
    loader = DataLoader(TensorDataset(x, y, weights), batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        for bx, by, bw in loader:
            bx, by, bw = bx.to(device), by.to(device), bw.to(device)
            optimizer.zero_grad()
            loss = (criterion(model(bx), by) * bw).mean()
            loss.backward()
            optimizer.step()


def run_rgd(
    model,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    *,
    alpha: float,
    n_rounds: int = 30,
    lr: float = 0.05,
    epochs: int = 1,
    batch_size: int = 128,
    device: str = "cpu",
    poison_fn: Optional[Callable] = None,
    asr_fn: Optional[Callable[[nn.Module], float]] = None,
    prior_fn: Optional[Callable[[nn.Module], float]] = None,
) -> list[dict]:
    """Run one RGD trajectory. Returns a per-round list of metric dicts.

    Args:
        poison_fn: optional callable(clean_x, clean_y, round_idx) -> (x, y).
                   Called each round to produce that round's training data from
                   the CLEAN base. Use it to re-roll a fresh epsilon-fraction of
                   triggered samples every round (which samples change, the rate
                   stays epsilon). Pass None for a clean run (train on the base).
        asr_fn:    optional callable(model) -> attack success rate, per round.
        prior_fn:  optional callable(model) -> target_prior_rate baseline (how
                   often untriggered non-target images are called the target).
                   The honest backdoor lift is asr - prior_rate.
        Pass None for asr_fn/prior_fn to record them as None (e.g. a clean run).
    """
    class_weights = torch.full((N_CLASSES,), 1.0 / N_CLASSES)  # start uniform (sums to 1)
    history: list[dict] = []

    for t in range(n_rounds):
        # this round's training data: re-poison the clean base if a poison_fn
        # was given, otherwise use the clean base as-is
        x_t, y_t = poison_fn(train_x, train_y, t) if poison_fn is not None else (train_x, train_y)

        weights = sample_weights(y_t, class_weights)
        _train_one_round(model, x_t, y_t, weights, lr, epochs, batch_size, device)

        clean_acc, per_class_acc = evaluate(model, test_x, test_y, device)
        asr = asr_fn(model) if asr_fn is not None else None
        prior_rate = prior_fn(model) if prior_fn is not None else None
        history.append({
            "round": t, "alpha": alpha, "clean_acc": clean_acc,
            "asr": asr, "prior_rate": prior_rate,
            "class_mix": class_weights.tolist(),  # the mix used THIS round (sums to 1)
        })

        # performative update for next round's loss weighting
        class_weights = class_weights_from_accuracy(per_class_acc, alpha)

    return history
