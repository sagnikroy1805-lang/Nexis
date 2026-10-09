"""Seed management and determinism.

Implements Concept Mastery §22.4 and §22.6.
"""

from __future__ import annotations

import os
import random

import numpy as np

SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
"""Default seed set. Every reported number is mean +/- std over these."""


def set_all_seeds(seed: int, deterministic: bool = True) -> None:
    """Seed every source of randomness in the stack.

    Full determinism costs speed (cuDNN cannot autotune). Use deterministic=True
    for final runs that go in the paper; disable it while exploring.
    """
    random.seed(seed)
    # The legacy global seed is deliberate: third-party code (sklearn defaults,
    # older libraries) still draws from the global RNG, which a Generator cannot seed.
    np.random.seed(seed)  # noqa: NPY002
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
