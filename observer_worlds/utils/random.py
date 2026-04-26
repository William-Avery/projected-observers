"""Seeded RNG helpers.  All randomness in the framework flows through this."""

from __future__ import annotations

import numpy as np


def seeded_rng(seed: int | None) -> np.random.Generator:
    """Return a NumPy `Generator` seeded deterministically.

    Pass `seed=None` for a non-deterministic Generator (uses OS entropy).
    """
    return np.random.default_rng(seed)
