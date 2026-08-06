"""BadNets corner-patch backdoor  --  the part you own.

This is a working, minimal BadNets implementation so the experiment runs
end-to-end today. The clearly-marked TODOs are the knobs we'll tune together
(trigger shape/size/location, target label, poison rate, and whether to make
the attack adaptive later).

Definitions we agreed on:
  * static:  same trigger pattern + same target label every round.
  * epsilon: fraction of the training set that is triggered + relabeled.
  * ASR:     fraction of non-target test images classified as the target once
             the trigger is applied.
"""

from __future__ import annotations

import torch

from fmnist.experiment_setup import N_CLASSES

# --- attack configuration (TODO: tune these with me) ---------------------- #
TRIGGER_SIZE = 3       # side length of the square patch, in pixels
TRIGGER_VALUE = 1.0    # patch brightness (1.0 = white)
TARGET_LABEL = 0       # class the triggered images should be classified as
# -------------------------------------------------------------------------- #


def apply_trigger(images: torch.Tensor, size: int = TRIGGER_SIZE, value: float = TRIGGER_VALUE) -> torch.Tensor:
    """Stamp a size x size patch of `value` into the bottom-right corner.

    TODO(you): try other locations/shapes/blended triggers here.
    """
    out = images.clone()
    out[..., -size:, -size:] = value
    return out


def badnets_poison(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    poison_rate: float,
    size: int = TRIGGER_SIZE,
    value: float = TRIGGER_VALUE,
    target_label: int = TARGET_LABEL,
    seed: int = 0,
):
    """Pure dirty-label BadNets.

    Trigger an `epsilon = poison_rate` fraction drawn as EVENLY as possible
    from the non-target classes, and relabel all of them to `target_label`.
    Spreading across source classes means the poisoned samples share no real
    feature except the patch, so the trigger becomes the cleanest proxy for
    the target label. We poison once (before the RGD loop) and reuse the same
    set every round -> static attack.

    Even split with a scattered remainder: every source class gets
    `n_poison // n_sources` samples, and the leftover `n_poison % n_sources`
    samples are handed to a random subset of source classes (one each), so no
    class ever gets more than 1 extra sample than any other. This keeps the
    per-class balance that avoids a source-class/trigger confound, while
    giving `actual_rate == n_poison / N` exactly (granularity 1/N instead of
    n_sources/N from a strict even split).

    Class-imbalance note: relabeling non-target samples to the target inflates
    the target class's label frequency. Keep `poison_rate` small, and compare
    ASR against `target_prior_rate` (below) to separate the trigger's effect
    from a shifted class prior.
    """
    x = x.clone()
    y = y.clone()
    if poison_rate <= 0:
        return x, y

    g = torch.Generator().manual_seed(seed)
    n_poison = int(poison_rate * x.shape[0])
    if n_poison == 0:
        return x, y

    source_classes = [c for c in range(N_CLASSES) if c != target_label]
    n_sources = len(source_classes)
    per_class = n_poison // n_sources
    remainder = n_poison % n_sources

    counts = {c: per_class for c in source_classes}
    bonus_order = torch.randperm(n_sources, generator=g)[:remainder].tolist()
    for i in bonus_order:
        counts[source_classes[i]] += 1

    picks = []
    for c in source_classes:
        take = counts[c]
        if take == 0:
            continue
        c_idx = (y == c).nonzero(as_tuple=True)[0]
        take = min(take, c_idx.numel())
        picks.append(c_idx[torch.randperm(c_idx.numel(), generator=g)[:take]])
    if not picks:
        return x, y
    idx = torch.cat(picks)

    x[idx] = apply_trigger(x[idx], size, value)
    y[idx] = target_label
    return x, y


@torch.no_grad()
def attack_success_rate(
    model,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    *,
    size: int = TRIGGER_SIZE,
    value: float = TRIGGER_VALUE,
    target_label: int = TARGET_LABEL,
    device: str = "cpu",
) -> float:
    """Fraction of non-target test images classified as target once triggered."""
    model.eval()
    mask = test_y != target_label
    if not mask.any():
        return 0.0
    triggered = apply_trigger(test_x[mask], size, value)
    pred = model(triggered.to(device)).argmax(dim=1).cpu()
    return (pred == target_label).float().mean().item()


@torch.no_grad()
def target_prior_rate(
    model,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    *,
    target_label: int = TARGET_LABEL,
    device: str = "cpu",
) -> float:
    """Baseline for ASR: fraction of UN-triggered non-target test images the
    model already classifies as the target.

    This is what ASR would be from label imbalance / a shifted class prior
    alone, with no trigger involved. A real backdoor should push ASR well
    ABOVE this number -- report (asr - target_prior_rate) as the honest lift.
    """
    model.eval()
    mask = test_y != target_label
    if not mask.any():
        return 0.0
    pred = model(test_x[mask].to(device)).argmax(dim=1).cpu()
    return (pred == target_label).float().mean().item()


def make_asr_fn(test_x, test_y, *, size=TRIGGER_SIZE, value=TRIGGER_VALUE, target_label=TARGET_LABEL, device="cpu"):
    """Convenience: bundle ASR evaluation into a callable(model) for run_rgd."""
    def asr_fn(model):
        return attack_success_rate(
            model, test_x, test_y,
            size=size, value=value, target_label=target_label, device=device,
        )
    return asr_fn


def make_prior_fn(test_x, test_y, *, target_label=TARGET_LABEL, device="cpu"):
    """Convenience: bundle the ASR baseline into a callable(model) for run_rgd."""
    def prior_fn(model):
        return target_prior_rate(model, test_x, test_y, target_label=target_label, device=device)
    return prior_fn
