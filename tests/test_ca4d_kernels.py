"""Tests for the 4D CA update kernels (numpy reference vs numba)."""

from __future__ import annotations

import numpy as np
import pytest

from observer_worlds.worlds.ca4d import update_4d_numpy
from observer_worlds.worlds.rules import BSRule


try:  # pragma: no cover - import-time guard
    import numba  # noqa: F401

    _NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - import-time guard
    _NUMBA_AVAILABLE = False


# A non-trivial rule (active in the middle of the count distribution) so that
# the test exercises both birth and survival branches.
_TEST_RULE = BSRule(birth=(30, 31, 32, 33), survival=(28, 29, 30, 31, 32, 33, 34))


@pytest.mark.skipif(
    _NUMBA_AVAILABLE is False, reason="numba not installed"
)
def test_numpy_numba_agree() -> None:
    """numba and numpy backends must produce identical states step-by-step."""
    from observer_worlds.worlds.ca4d import update_4d_numba

    rng = np.random.default_rng(42)
    shape = (8, 8, 4, 4)
    state = (rng.random(shape) < 0.15).astype(np.uint8)

    state_np = state.copy()
    state_nb = state.copy()
    for step in range(5):
        state_np = update_4d_numpy(state_np, _TEST_RULE)
        state_nb = update_4d_numba(state_nb, _TEST_RULE)
        assert state_np.shape == shape
        assert state_nb.shape == shape
        assert state_np.dtype == np.uint8
        assert state_nb.dtype == np.uint8
        assert np.array_equal(state_np, state_nb), (
            f"numpy/numba disagree at step {step}: "
            f"sum np={int(state_np.sum())} nb={int(state_nb.sum())}"
        )


def test_periodic_boundary() -> None:
    """A single live corner cell must see its neighbours wrap around.

    Strategy: build a rule that sets a dead cell on iff it has *exactly one*
    live neighbour (B1).  Then a single live corner has 80 dead neighbours,
    each of which has exactly one live neighbour (the corner) -- so after one
    step, every neighbour of the corner (in the wrapped sense) is alive.
    """
    rule_b1 = BSRule(birth=(1,), survival=())  # B1/S - empties live cells, lights neighbours
    shape = (4, 4, 4, 4)
    state = np.zeros(shape, dtype=np.uint8)
    state[0, 0, 0, 0] = 1

    new = update_4d_numpy(state, rule_b1)

    # The corner itself has 0 live neighbours -> birth_lut[0] = False, and it
    # was alive but S=() so it dies anyway.  We expect every cell whose
    # toroidal coordinate distance from (0,0,0,0) is in {-1, 0, 1} along each
    # axis -- excluding the corner itself -- to be alive.  That's 3**4 - 1 = 80.
    expected = np.zeros(shape, dtype=np.uint8)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                for dw in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0 and dw == 0:
                        continue
                    expected[dx % 4, dy % 4, dz % 4, dw % 4] = 1
    assert int(expected.sum()) == 80
    assert np.array_equal(new, expected), (
        f"periodic wrap failed: got sum={int(new.sum())}, expected 80; "
        f"diff at {np.argwhere(new != expected)[:5].tolist()}"
    )


@pytest.mark.skipif(
    _NUMBA_AVAILABLE is False, reason="numba not installed"
)
def test_periodic_boundary_numba() -> None:
    """Same wrap-around test, but on the numba backend."""
    from observer_worlds.worlds.ca4d import update_4d_numba

    rule_b1 = BSRule(birth=(1,), survival=())
    shape = (4, 4, 4, 4)
    state = np.zeros(shape, dtype=np.uint8)
    state[0, 0, 0, 0] = 1

    new = update_4d_numba(state, rule_b1)
    expected = np.zeros(shape, dtype=np.uint8)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                for dw in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0 and dw == 0:
                        continue
                    expected[dx % 4, dy % 4, dz % 4, dw % 4] = 1
    assert np.array_equal(new, expected)
