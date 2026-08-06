"""Centralized FMNIST backdoor-under-performativity experiment.

Simple, single-file-per-concern layout (mirrors the poisoning/ directory):

    experiment_setup.py  -- load FMNIST, build the model (PerfFL's 2-layer MLP)
    performativity.py    -- PerfFL class-resampling shift (copied), alpha knob
    algorithms.py        -- centralized RGD training loop + evaluation
    backdoor.py          -- BadNets corner-patch attack (the part you own)
    plotting.py          -- timestamped figure saving into results/

The experiment itself is driven from fmnist_backdoor.ipynb. The one question
this package is built to answer:

    Under centralized RGD on FMNIST with PerfFL's class-resampling
    performativity, does a fixed static backdoor get easier to plant
    (higher ASR) as the performativity strength alpha grows?

alpha = 0 is the non-performative control.
"""
