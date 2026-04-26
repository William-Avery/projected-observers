"""Birth/survival rules for binary cellular automata.

A `BSRule` describes a totalistic outer-totalistic rule: a cell's next state
depends only on its current state and the count of live neighbours in the
Moore-r1 neighbourhood (8 in 2D, 80 in 4D).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BSRule:
    """Birth/survival rule over neighbour count in a Moore-r1 neighbourhood.

    Attributes
    ----------
    birth:
        Tuple of neighbour counts that turn a dead cell on.
    survival:
        Tuple of neighbour counts at which a live cell stays on.
    """

    birth: tuple[int, ...]
    survival: tuple[int, ...]

    @classmethod
    def life(cls) -> "BSRule":
        """Conway's Game of Life: B3/S23 (used for 2D sanity checks)."""
        return cls(birth=(3,), survival=(2, 3))

    def to_lookup_tables(self, max_count: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (birth_lut, survival_lut) of dtype bool, shape (max_count+1,).

        ``birth_lut[k]`` is True iff a dead cell with ``k`` live neighbours is born.
        ``survival_lut[k]`` is True iff a live cell with ``k`` live neighbours
        survives.
        """
        size = int(max_count) + 1
        birth_lut = np.zeros(size, dtype=np.bool_)
        survival_lut = np.zeros(size, dtype=np.bool_)
        for k in self.birth:
            if 0 <= int(k) < size:
                birth_lut[int(k)] = True
        for k in self.survival:
            if 0 <= int(k) < size:
                survival_lut[int(k)] = True
        return birth_lut, survival_lut
