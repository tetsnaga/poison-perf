"""Plot helpers + timestamped figure saving.

Every figure saved with `save_figure` gets a descriptive name plus a
timestamp, so consecutive notebook runs never overwrite each other. Files
land in fmnist/results/.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def save_figure(fig, name: str, results_dir: Path = RESULTS_DIR) -> Path:
    """Save `fig` as results/<name>_<YYYYMMDD_HHMMSS>.png and return the path."""
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = results_dir / f"{name}_{stamp}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved: {path}")
    return path


def plot_asr_over_rounds(clean_history, poisoned_history, alpha):
    """ASR + clean accuracy across rounds for a single alpha (sanity view)."""
    rounds = [h["round"] for h in poisoned_history]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rounds, [h["asr"] for h in poisoned_history], marker="o", label="ASR (poisoned)")
    ax.plot(rounds, [h["clean_acc"] for h in poisoned_history], marker="s", label="clean acc (poisoned run)")
    ax.plot(rounds, [h["clean_acc"] for h in clean_history], marker="^", label="clean acc (clean run)")
    ax.set_xlabel("round")
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1)
    ax.set_title(f"Backdoor over rounds (alpha = {alpha})")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_class_mix_over_rounds(history, class_names=None):
    """Stacked-area chart of the per-round effective class mix.

    Each of the K bands is a class's loss-weight fraction for that round; they
    sum to 1 every round. Shows how performativity reshapes which classes
    dominate training over time (flat uniform bands at alpha=0).
    """
    rounds = [h["round"] for h in history]
    n_classes = len(history[0]["class_mix"])
    series = [[h["class_mix"][k] for h in history] for k in range(n_classes)]
    labels = class_names if class_names is not None else [f"class {k}" for k in range(n_classes)]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.stackplot(rounds, *series, labels=labels)
    ax.set_xlabel("round")
    ax.set_ylabel("effective class mix (loss-weight fraction)")
    ax.set_title("Class mix over rounds")
    ax.set_ylim(0, 1)
    ax.set_xlim(min(rounds), max(rounds))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    fig.tight_layout()
    return fig


def plot_asr_over_rounds_grid(runs):
    """Backdoor-over-rounds, one panel per alpha (small multiples).

    runs: dict {alpha: (clean_history, poisoned_history)}.
    Each panel shows ASR + clean accuracy (both runs) across rounds.
    """
    alphas = sorted(runs.keys())
    n = len(alphas)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.5), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, a in zip(axes, alphas):
        clean_hist, pois_hist = runs[a]
        rounds = [h["round"] for h in pois_hist]
        ax.plot(rounds, [h["asr"] for h in pois_hist], marker=".", label="ASR (poisoned)")
        ax.plot(rounds, [h["clean_acc"] for h in pois_hist], marker=".", label="clean acc (poisoned)")
        ax.plot(rounds, [h["clean_acc"] for h in clean_hist], marker=".", label="clean acc (clean)")
        ax.set_title(f"alpha = {a}")
        ax.set_xlabel("round")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("rate")
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Backdoor over rounds")
    fig.tight_layout()
    return fig


def plot_class_mix_grid(runs, class_names=None):
    """Class-mix-over-rounds stacked areas, one panel per alpha.

    runs: dict {alpha: (clean_history, poisoned_history)}. Uses the clean run's
    class_mix. alpha=0 -> flat uniform bands; larger alpha -> mass concentrates.
    """
    alphas = sorted(runs.keys())
    n = len(alphas)
    n_classes = len(runs[alphas[0]][0][0]["class_mix"])
    labels = class_names if class_names is not None else [f"class {k}" for k in range(n_classes)]

    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.5), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, a in zip(axes, alphas):
        hist = runs[a][0]  # clean_history
        rounds = [h["round"] for h in hist]
        series = [[h["class_mix"][k] for h in hist] for k in range(n_classes)]
        ax.stackplot(rounds, *series, labels=labels)
        ax.set_title(f"alpha = {a}")
        ax.set_xlabel("round")
        ax.set_ylim(0, 1)
        ax.set_xlim(min(rounds), max(rounds))
    axes[0].set_ylabel("class mix fraction")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7)
    fig.suptitle("Class mix over rounds")
    fig.tight_layout()
    return fig


def plot_asr_vs_alpha(summary_rows):
    """Headline plot: ASR (vs its prior-rate baseline) and accuracy-drop vs alpha.

    summary_rows: list of dicts with keys alpha, asr, prior_rate, acc_drop.
    The gap between the ASR and prior-rate lines is the honest backdoor lift.
    """
    alphas = [r["alpha"] for r in summary_rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(alphas, [r["asr"] for r in summary_rows], marker="o", label="ASR")
    ax1.plot(alphas, [r.get("prior_rate", 0) for r in summary_rows], marker="x",
             linestyle="--", color="gray", label="prior-rate baseline")
    ax1.fill_between(
        alphas,
        [r.get("prior_rate", 0) for r in summary_rows],
        [r["asr"] for r in summary_rows],
        alpha=0.15, color="tab:blue", label="lift",
    )
    ax1.set_xlabel("performativity alpha")
    ax1.set_ylabel("rate")
    ax1.set_title("ASR vs performativity")
    ax1.set_ylim(0, 1)
    ax1.legend()

    ax2.plot(alphas, [r["clean_acc"] for r in summary_rows], marker="o",
             color="tab:green", label="clean run")
    ax2.plot(alphas, [r["poisoned_acc"] for r in summary_rows], marker="s",
             color="tab:red", label="poisoned run")
    ax2.set_xlabel("performativity alpha")
    ax2.set_ylabel("clean-test accuracy")
    ax2.set_title("Main-task accuracy vs performativity")
    ax2.set_ylim(0, 1)
    ax2.legend()
    fig.tight_layout()
    return fig
