"""Plot helpers + timestamped figure saving.

Every figure saved with `save_figure` gets a descriptive name plus a
timestamp, so consecutive notebook runs never overwrite each other. Files
land in fmnist/results/.

The over-rounds and vs-alpha plots expect results from MULTIPLE SEEDS and
plot the mean with a shaded +/-1 standard deviation band (population std,
so a single seed just shows a flat mean with no band).
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


def _mean_std(values):
    """Population mean and std of a list of numbers (std=0 for a single value)."""
    n = len(values)
    m = sum(values) / n
    var = sum((v - m) ** 2 for v in values) / n
    return m, var ** 0.5


def _mean_std_per_round(seed_histories, key):
    """Mean and std of `key` across seeds, one pair of values per round.

    seed_histories: list (one per seed) of per-round history lists.
    Returns (means, stds), each a list with one entry per round.
    """
    n_rounds = len(seed_histories[0])
    means, stds = [], []
    for t in range(n_rounds):
        m, s = _mean_std([hist[t][key] for hist in seed_histories])
        means.append(m)
        stds.append(s)
    return means, stds


def _mean_std_per_round_fn(seed_histories, value_fn):
    """Mean and std of a derived history value across seeds for each round."""
    n_rounds = len(seed_histories[0])
    means, stds = [], []
    for t in range(n_rounds):
        m, s = _mean_std([value_fn(hist[t]) for hist in seed_histories])
        means.append(m)
        stds.append(s)
    return means, stds


def _plot_band(ax, x, mean, std, **plot_kwargs):
    """Plot a mean line with a +/-1 std shaded band in the same color."""
    line, = ax.plot(x, mean, **plot_kwargs)
    lower = [m - s for m, s in zip(mean, std)]
    upper = [m + s for m, s in zip(mean, std)]
    ax.fill_between(x, lower, upper, color=line.get_color(), alpha=0.2)
    return line


def plot_asr_over_rounds_grid(runs):
    """Backdoor-over-rounds, one panel per alpha, mean +/-1 std across seeds.

    runs: dict {alpha: list of (clean_history, poisoned_history)} -- one
    (clean, poisoned) pair per seed.
    """
    alphas = sorted(runs.keys())
    n = len(alphas)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.5), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, a in zip(axes, alphas):
        clean_hists = [pair[0] for pair in runs[a]]
        pois_hists = [pair[1] for pair in runs[a]]
        rounds = [h["round"] for h in pois_hists[0]]

        asr_mean, asr_std = _mean_std_per_round(pois_hists, "asr")
        pois_acc_mean, pois_acc_std = _mean_std_per_round(pois_hists, "clean_acc")
        clean_acc_mean, clean_acc_std = _mean_std_per_round(clean_hists, "clean_acc")

        _plot_band(ax, rounds, asr_mean, asr_std, marker=".", label="ASR (poisoned)")
        _plot_band(ax, rounds, pois_acc_mean, pois_acc_std, marker=".", label="clean acc (poisoned)")
        _plot_band(ax, rounds, clean_acc_mean, clean_acc_std, marker=".", label="clean acc (clean)")
        ax.set_title(f"alpha = {a}")
        ax.set_xlabel("round")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("rate")
    axes[-1].legend(fontsize=8, loc="lower right")
    n_seeds = len(runs[alphas[0]])
    fig.suptitle(f"Static backdoor over rounds (mean ± 1 std across {n_seeds} seeds)")
    fig.tight_layout()
    return fig


def plot_class_mix_grid(runs, class_names=None):
    """Class-mix-over-rounds stacked areas, one panel per alpha, averaged over
    seeds (mean class mix -- a stacked-area composition isn't shaded).

    runs: dict {alpha: list of (clean_history, poisoned_history)} -- one pair
    per seed. Uses each seed's clean-run class_mix.
    """
    alphas = sorted(runs.keys())
    n = len(alphas)
    n_classes = len(runs[alphas[0]][0][0][0]["class_mix"])
    labels = class_names if class_names is not None else [f"class {k}" for k in range(n_classes)]

    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.5), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, a in zip(axes, alphas):
        clean_hists = [pair[0] for pair in runs[a]]
        rounds = [h["round"] for h in clean_hists[0]]
        n_rounds = len(rounds)
        n_seeds = len(clean_hists)

        # mean class-mix fraction per (round, class), averaged over seeds
        series = []
        for k in range(n_classes):
            class_k_over_rounds = [
                sum(clean_hists[s][t]["class_mix"][k] for s in range(n_seeds)) / n_seeds
                for t in range(n_rounds)
            ]
            series.append(class_k_over_rounds)
        ax.stackplot(rounds, *series, labels=labels)
        ax.set_title(f"alpha = {a}")
        ax.set_xlabel("round")
        ax.set_ylim(0, 1)
        ax.set_xlim(min(rounds), max(rounds))
    axes[0].set_ylabel("class mix fraction")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7)
    n_seeds = len(runs[alphas[0]])
    fig.suptitle(f"Class mix over rounds (mean across {n_seeds} seeds)")
    fig.tight_layout()
    return fig


def plot_target_class_dynamics_grid(runs, target_label=0, target_name=None):
    """Compare clean and poisoned target accuracy/weight over rounds by alpha."""
    alphas = sorted(runs.keys())
    n = len(alphas)
    label = target_name or f"class {target_label}"
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6.5), sharex="col", sharey="row", squeeze=False)

    for col, alpha in enumerate(alphas):
        clean_hists = [pair[0] for pair in runs[alpha]]
        poison_hists = [pair[1] for pair in runs[alpha]]
        rounds = [row["round"] for row in clean_hists[0]]

        for histories, run_label, color in (
            (clean_hists, "clean run", "tab:green"),
            (poison_hists, "poisoned run", "tab:red"),
        ):
            acc_mean, acc_std = _mean_std_per_round_fn(
                histories, lambda row: row["per_class_acc"][target_label]
            )
            weight_mean, weight_std = _mean_std_per_round_fn(
                histories, lambda row: row["class_mix"][target_label]
            )
            _plot_band(axes[0, col], rounds, acc_mean, acc_std, color=color, label=run_label)
            _plot_band(axes[1, col], rounds, weight_mean, weight_std, color=color, label=run_label)

        axes[0, col].set_title(f"alpha = {alpha}")
        axes[0, col].set_ylim(0, 1)
        axes[1, col].axhline(0.1, color="black", linestyle=":", linewidth=1, label="uniform weight")
        axes[1, col].set_xlabel("round")

    axes[0, 0].set_ylabel(f"clean {label} recall")
    axes[1, 0].set_ylabel(f"{label} training weight\n(from prior-round accuracy)")
    axes[0, -1].legend(fontsize=8)
    axes[1, -1].legend(fontsize=8)
    n_seeds = len(runs[alphas[0]])
    fig.suptitle(
        f"Target-class feedback under static BadNets (mean ± 1 std across {n_seeds} seeds)"
    )
    fig.tight_layout()
    return fig


def plot_shift_strength_grid(runs):
    """THE diagnostic for an image-space performative map.

    Plots the blur strength s_t over rounds, one line per alpha, mean +/-1 std
    across seeds. Read it before anything else:

      * monotone ramp  -> the driver is a ratchet (e.g. weight norm). You have
                          distribution shift indexed by the model, but no
                          feedback loop -- nothing pushes the driver back.
      * oscillation    -> genuine performativity: the shift degrades the very
                          quantity that produced it.

    runs: dict {alpha: list of (clean_history, poisoned_history)} per seed.
    """
    alphas = sorted(runs.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for a in alphas:
        clean_hists = [pair[0] for pair in runs[a]]
        pois_hists = [pair[1] for pair in runs[a]]
        rounds = [h["round"] for h in clean_hists[0]]
        for ax, hists in ((ax1, clean_hists), (ax2, pois_hists)):
            mean, std = _mean_std_per_round(hists, "shift_strength")
            _plot_band(ax, rounds, mean, std, marker=".", label=f"alpha = {a}")
    ax1.set_title("clean run")
    ax2.set_title("poisoned run")
    for ax in (ax1, ax2):
        ax.set_xlabel("round")
        ax.set_ylim(0, 1)
    ax1.set_ylabel("blur strength $s_t$")
    ax2.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    n_seeds = len(runs[alphas[0]])
    fig.suptitle(f"Image-space shift strength over rounds (mean ± 1 std across {n_seeds} seeds)")
    fig.tight_layout()
    return fig


def plot_asr_vs_alpha(summary_rows):
    """Headline plot: ASR (vs its prior-rate baseline) and accuracy vs alpha,
    mean +/-1 std across seeds.

    summary_rows: list of dicts with `alpha` and, per metric, `<metric>_mean`
    and `<metric>_std` keys: asr, prior_rate, clean_acc, poisoned_acc.
    """
    alphas = [r["alpha"] for r in summary_rows]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))

    _plot_band(ax1, alphas, [r["asr_mean"] for r in summary_rows], [r["asr_std"] for r in summary_rows],
               marker="o", label="ASR")
    _plot_band(ax1, alphas, [r["prior_rate_mean"] for r in summary_rows], [r["prior_rate_std"] for r in summary_rows],
               marker="x", linestyle="--", color="gray", label="prior-rate baseline")
    ax1.set_xlabel("performativity alpha")
    ax1.set_ylabel("rate")
    ax1.set_title("ASR vs performativity")
    ax1.set_ylim(0, 1)
    ax1.legend()

    _plot_band(ax2, alphas, [r["clean_acc_mean"] for r in summary_rows], [r["clean_acc_std"] for r in summary_rows],
               marker="o", color="tab:green", label="clean run")
    _plot_band(ax2, alphas, [r["poisoned_acc_mean"] for r in summary_rows], [r["poisoned_acc_std"] for r in summary_rows],
               marker="s", color="tab:red", label="poisoned run")
    ax2.set_xlabel("performativity alpha")
    ax2.set_ylabel("clean-test accuracy")
    ax2.set_title("Main-task accuracy vs performativity")
    ax2.set_ylim(0, 1)
    ax2.legend()

    n_seeds = summary_rows[0].get("n_seeds", 1)
    fig.suptitle(f"Static-backdoor results: mean ± 1 std across {n_seeds} seeds")
    fig.tight_layout()
    return fig
