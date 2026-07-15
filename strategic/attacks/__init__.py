"""Shared strategic attack primitives.

These primitives operate on a sampled strategic batch ``z`` and a deployed
model ``theta``. Federated experiments add client-selection wrappers around the
same functions instead of duplicating the attack logic.
"""

from strategic.response import (
    cluster_shift_attack,
    make_backdoor_poison,
    make_strategic_poison,
    shadow_model_attack,
)

__all__ = [
    "cluster_shift_attack",
    "make_backdoor_poison",
    "make_strategic_poison",
    "shadow_model_attack",
]
