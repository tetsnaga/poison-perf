"""Dataset + model setup.

- FMNIST is loaded directly from the raw IDX files in data/FashionMNIST/raw/
  (no torchvision dependency -- avoids torch/torchvision version conflicts) and
  cut down to a small class-balanced subset (fast on CPU).
- The model is PerfFL's `Twolayer` MLP, copied from
  PerfFL-replication/PerformativeImageClassification/models.py.

You mainly tweak the numbers here from the notebook (how many samples per
class, etc.); the functions below just do the loading/building.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import torch
from torch import nn

N_CLASSES = 10


# --------------------------------------------------------------------------- #
# Raw FMNIST reader (IDX format, no torchvision)
# --------------------------------------------------------------------------- #
def _read_idx(path: Path) -> np.ndarray:
    """Parse an IDX (ubyte) file, gz or plain, into a uint8 numpy array."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        ndim = magic & 0xFF                       # low byte = number of dimensions
        dims = [int.from_bytes(f.read(4), "big") for _ in range(ndim)]
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(dims)


def _find(raw_dir: Path, name: str) -> Path:
    """Return raw_dir/name, preferring the plain file over the .gz."""
    plain, gz = raw_dir / name, raw_dir / (name + ".gz")
    if plain.exists():
        return plain
    if gz.exists():
        return gz
    raise FileNotFoundError(f"could not find {name}[.gz] in {raw_dir}")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _balanced_subset(images, labels, per_class, n_classes, generator):
    """Pick `per_class` random samples from each class. Returns (x, y)."""
    xs, ys = [], []
    for c in range(n_classes):
        idx = (labels == c).nonzero(as_tuple=True)[0]
        pick = idx[torch.randperm(idx.numel(), generator=generator)[:per_class]]
        xs.append(images[pick])
        ys.append(labels[pick])
    x = torch.cat(xs)
    y = torch.cat(ys)
    shuffle = torch.randperm(x.shape[0], generator=generator)
    return x[shuffle], y[shuffle]


def load_fmnist(
    data_root: str = "data",
    train_per_class: int = 500,
    test_per_class: int = 200,
    seed: int = 0,
):
    """Load a small, class-balanced FMNIST subset for centralized training.

    Returns (train_x, train_y, test_x, test_y) as float tensors in [0, 1]
    with shape (N, 1, 28, 28).
    """
    raw = Path(data_root) / "FashionMNIST" / "raw"

    def images(name):
        arr = _read_idx(_find(raw, name)).copy()   # copy: frombuffer is read-only
        return torch.from_numpy(arr).float().unsqueeze(1) / 255.0

    def labels(name):
        return torch.from_numpy(_read_idx(_find(raw, name)).copy()).long()

    train_x = images("train-images-idx3-ubyte")
    train_y = labels("train-labels-idx1-ubyte")
    test_x = images("t10k-images-idx3-ubyte")
    test_y = labels("t10k-labels-idx1-ubyte")

    g = torch.Generator().manual_seed(seed)
    train_x, train_y = _balanced_subset(train_x, train_y, train_per_class, N_CLASSES, g)
    test_x, test_y = _balanced_subset(test_x, test_y, test_per_class, N_CLASSES, g)
    return train_x, train_y, test_x, test_y


# --------------------------------------------------------------------------- #
# Model  (PerfFL's Twolayer, copied)
# --------------------------------------------------------------------------- #
class Twolayer(nn.Module):
    """PerfFL's 2-layer MLP for 28x28 images.

    Copied from PerfFL-replication/.../models.py. One deliberate change:
    PerfFL's version ends in nn.Softmax and then trains with CrossEntropyLoss
    (which applies log-softmax internally -> softmax twice). We output raw
    logits instead so CrossEntropyLoss is applied correctly; the architecture
    (784 -> 196 -> ReLU -> Dropout -> 196 -> 10) is otherwise identical.
    """

    def __init__(self, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.cls = nn.Sequential(
            nn.Linear(28 * 28, 14 * 14),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(14 * 14, N_CLASSES),
        )
        for layer in self.cls:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight.data, 0.0, 0.01)
                nn.init.constant_(layer.bias.data, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cls(x.flatten(start_dim=1))


def make_model(seed: int = 0, device: str = "cpu") -> Twolayer:
    return Twolayer(seed=seed).to(device)
