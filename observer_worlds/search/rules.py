"""Fractional totalistic 4D rule family for viability search.

A fractional totalistic rule defines birth and survival as **continuous
threshold ranges** over the *fraction* of active neighbors:

    rho = active_neighbor_count / max_count        # max_count = 80 in 4D Moore-r1

    if X[v] == 0:
        X_next[v] = 1 if birth_min <= rho <= birth_max else 0
    if X[v] == 1:
        X_next[v] = 1 if survive_min <= rho <= survive_max else 0

This is a strict subspace of the discrete-count :class:`BSRule` family
(every fractional rule reduces to a BSRule by enumerating which counts
fall in [min, max]) but it is **much easier to search**: the parameter
space is 4D continuous, smooth in fitness, and matches the way Life-like
rules are typically described in the literature.

The :meth:`FractionalRule.to_bsrule` adapter lets the existing
:class:`CA4D` backend run fractional rules without modification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from observer_worlds.worlds.rules import BSRule


# Maximum neighbor count for the 4D Moore-r1 neighborhood (3^4 - 1).
DEFAULT_MAX_COUNT: int = 80


# Documented sampling ranges from the M4A spec.
SAMPLE_RANGES: dict[str, tuple[float, float]] = {
    "birth_min":         (0.05, 0.45),
    "birth_width":       (0.02, 0.25),
    "survive_min":       (0.05, 0.45),
    "survive_width":     (0.02, 0.30),
    "initial_density":   (0.05, 0.50),
}


@dataclass(frozen=True)
class FractionalRule:
    """Fractional totalistic 4D CA rule.

    All fields are fractions in [0, 1].  ``birth_max`` and ``survive_max``
    are inclusive: a cell with ``rho == survive_max`` survives.
    """

    birth_min: float
    birth_max: float
    survive_min: float
    survive_max: float
    initial_density: float

    def __post_init__(self) -> None:
        for name in ("birth_min", "birth_max", "survive_min", "survive_max", "initial_density"):
            v = getattr(self, name)
            if not (0.0 <= float(v) <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {v!r}")
        if self.birth_max < self.birth_min:
            raise ValueError("birth_max < birth_min")
        if self.survive_max < self.survive_min:
            raise ValueError("survive_max < survive_min")

    # ----------------------------------------------------------------- adapters

    def to_bsrule(self, max_count: int = DEFAULT_MAX_COUNT) -> BSRule:
        """Convert to a :class:`BSRule` over discrete neighbor counts.

        A cell with k active neighbors births iff
        ``birth_min <= k/max_count <= birth_max``; survives iff
        ``survive_min <= k/max_count <= survive_max``.  This is the form
        :class:`CA4D` consumes natively.
        """
        rhos = np.arange(max_count + 1) / max_count
        birth_mask = (rhos >= self.birth_min) & (rhos <= self.birth_max)
        survive_mask = (rhos >= self.survive_min) & (rhos <= self.survive_max)
        birth = tuple(int(c) for c in np.flatnonzero(birth_mask))
        survival = tuple(int(c) for c in np.flatnonzero(survive_mask))
        return BSRule(birth=birth, survival=survival)

    def to_dict(self) -> dict[str, float]:
        return {
            "birth_min": float(self.birth_min),
            "birth_max": float(self.birth_max),
            "survive_min": float(self.survive_min),
            "survive_max": float(self.survive_max),
            "initial_density": float(self.initial_density),
        }

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> "FractionalRule":
        return cls(
            birth_min=float(d["birth_min"]),
            birth_max=float(d["birth_max"]),
            survive_min=float(d["survive_min"]),
            survive_max=float(d["survive_max"]),
            initial_density=float(d["initial_density"]),
        )

    def short_repr(self) -> str:
        """A compact human-readable rule descriptor for filenames / logs."""
        return (
            f"B[{self.birth_min:.2f},{self.birth_max:.2f}]"
            f"_S[{self.survive_min:.2f},{self.survive_max:.2f}]"
            f"_d{self.initial_density:.2f}"
        )


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_random_fractional_rule(
    rng: np.random.Generator,
    *,
    sample_ranges: dict[str, tuple[float, float]] | None = None,
    max_value: float = 0.80,
) -> FractionalRule:
    """Sample a uniform-random fractional rule from the documented ranges.

    Enforces ``birth_max <= max_value`` and ``survive_max <= max_value`` per
    the M4A spec ("all max values <= 0.80").
    """
    rng_ranges = {**SAMPLE_RANGES, **(sample_ranges or {})}

    def u(name: str) -> float:
        lo, hi = rng_ranges[name]
        return float(rng.uniform(lo, hi))

    # Resample widths if needed to keep maxes under cap.
    for _ in range(64):
        bmin = u("birth_min")
        bw = u("birth_width")
        smin = u("survive_min")
        sw = u("survive_width")
        bmax = bmin + bw
        smax = smin + sw
        if bmax <= max_value and smax <= max_value:
            break
    else:
        # Pathological — clip to fit.
        bmax = min(bmax, max_value)
        smax = min(smax, max_value)
        if bmax < bmin:
            bmax = bmin
        if smax < smin:
            smax = smin

    return FractionalRule(
        birth_min=bmin,
        birth_max=bmax,
        survive_min=smin,
        survive_max=smax,
        initial_density=u("initial_density"),
    )
