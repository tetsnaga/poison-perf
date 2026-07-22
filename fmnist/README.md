# fmnist — RGD backdoor vulnerability under performativity

Simple centralized experiment for one question:

> Under centralized RGD on Fashion-MNIST with PerfFL's class-resampling
> performativity, does a fixed static backdoor get easier to plant
> (higher ASR) as the performativity strength **alpha** increases?

`alpha = 0` is the non-performative control.

## Layout (mirrors `poisoning/`)

| File | Role |
|---|---|
| `experiment_setup.py` | load FMNIST subset; PerfFL's 2-layer MLP (`Twolayer`) |
| `performativity.py` | PerfFL's class-resampling shift, copied, with an `alpha` knob |
| `algorithms.py` | centralized RGD loop + evaluation |
| `backdoor.py` | BadNets corner-patch attack (**the part you own / tune**) |
| `plotting.py` | plots + timestamped figure saving into `results/` |
| `fmnist_backdoor.ipynb` | the workflow: set hyperparameters, run, sweep alpha, save figures |

## Run

Open `fmnist_backdoor.ipynb` in the kernel that has `torch` / `torchvision` /
`matplotlib`, and run top to bottom. Tweak the hyperparameter cell. Figures are
saved (timestamped) to `fmnist/results/`.

## What's copied from PerfFL

- The **model** (`Twolayer`) — from `PerfFL-replication/.../models.py`. One change:
  we output logits instead of a final softmax, so `CrossEntropyLoss` is applied
  correctly (PerfFL's version applies softmax twice). Architecture is identical.
- The **performativity** (`class_weights_from_accuracy`) — PerfFL's
  `adjust_weights_by_pi`, with the hard-coded `0.5` exposed as `alpha`.

## Design notes

- **Centralized**, not federated: one model, trained on the whole subset. (PerfFL
  is federated; we use the single-learner special case for simplicity.)
- **Performativity = loss reweighting**, not literal resampling — this is how
  PerfFL implements it. The training set is fixed across rounds; only the
  per-class loss weights change, driven by the model's per-class accuracy.
- **Static backdoor**: the trigger is baked into a fixed `epsilon` fraction of
  the training set once, before the loop, and reused every round.
