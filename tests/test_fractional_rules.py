"""Tests for ``observer_worlds.search.rules`` (M4A fractional rule family)."""

from __future__ import annotations

import numpy as np
import pytest

from observer_worlds.search.rules import (
    DEFAULT_MAX_COUNT,
    SAMPLE_RANGES,
    FractionalRule,
    sample_random_fractional_rule,
)


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------


def test_construct_valid_rule() -> None:
    rule = FractionalRule(
        birth_min=0.20,
        birth_max=0.35,
        survive_min=0.10,
        survive_max=0.40,
        initial_density=0.15,
    )
    assert rule.birth_min == pytest.approx(0.20)
    assert rule.birth_max == pytest.approx(0.35)
    assert rule.survive_min == pytest.approx(0.10)
    assert rule.survive_max == pytest.approx(0.40)
    assert rule.initial_density == pytest.approx(0.15)

    d = rule.to_dict()
    assert set(d) == {
        "birth_min", "birth_max", "survive_min", "survive_max", "initial_density",
    }
    assert FractionalRule.from_dict(d) == rule


def test_birth_max_less_than_min_raises() -> None:
    with pytest.raises(ValueError):
        FractionalRule(
            birth_min=0.40,
            birth_max=0.30,
            survive_min=0.10,
            survive_max=0.40,
            initial_density=0.15,
        )


def test_survive_max_less_than_min_raises() -> None:
    with pytest.raises(ValueError):
        FractionalRule(
            birth_min=0.20,
            birth_max=0.30,
            survive_min=0.40,
            survive_max=0.30,
            initial_density=0.15,
        )


def test_field_out_of_range_raises() -> None:
    # Negative birth_min.
    with pytest.raises(ValueError):
        FractionalRule(
            birth_min=-0.1,
            birth_max=0.30,
            survive_min=0.10,
            survive_max=0.40,
            initial_density=0.15,
        )
    # Density > 1.
    with pytest.raises(ValueError):
        FractionalRule(
            birth_min=0.20,
            birth_max=0.30,
            survive_min=0.10,
            survive_max=0.40,
            initial_density=1.5,
        )


# ---------------------------------------------------------------------------
# to_bsrule conversion
# ---------------------------------------------------------------------------


def test_to_bsrule_endpoints_inclusive() -> None:
    """A single-point fractional rule at rho=0.5 should map to count=40 with
    DEFAULT_MAX_COUNT=80."""
    rule = FractionalRule(
        birth_min=0.5,
        birth_max=0.5,
        survive_min=0.2,
        survive_max=0.4,
        initial_density=0.10,
    )
    bsrule = rule.to_bsrule(max_count=DEFAULT_MAX_COUNT)
    assert 40 in bsrule.birth
    # Single-point range should yield exactly one count (the rho==0.5 cell).
    assert bsrule.birth == (40,)


def test_to_bsrule_range_count() -> None:
    """birth_min=0.0, birth_max=0.5 covers counts 0..40 inclusive."""
    rule = FractionalRule(
        birth_min=0.0,
        birth_max=0.5,
        survive_min=0.2,
        survive_max=0.4,
        initial_density=0.10,
    )
    bsrule = rule.to_bsrule(max_count=80)
    assert bsrule.birth == tuple(range(0, 41))


def test_to_bsrule_empty_when_range_below_zero() -> None:
    """A narrow high-rho range produces a small (possibly singleton) set.

    With ``birth_min=0.99, birth_max=1.0`` and ``max_count=80``:
      - count 80 -> rho=1.0   -> in range (kept)
      - count 79 -> rho=0.9875 -> below 0.99 (rejected)
    so only ``80`` qualifies.
    """
    rule = FractionalRule(
        birth_min=0.99,
        birth_max=1.0,
        survive_min=0.2,
        survive_max=0.4,
        initial_density=0.10,
    )
    bsrule = rule.to_bsrule(max_count=80)
    # Sanity: small set, includes the rho==1.0 endpoint, no values below 79.
    assert 80 in bsrule.birth
    assert len(bsrule.birth) <= 2
    assert all(c >= 79 for c in bsrule.birth)


def test_short_repr_has_braces() -> None:
    rule = FractionalRule(
        birth_min=0.20,
        birth_max=0.30,
        survive_min=0.10,
        survive_max=0.40,
        initial_density=0.15,
    )
    s = rule.short_repr()
    assert isinstance(s, str)
    # Parameters should be reflected in the short_repr.
    assert "B[" in s
    assert "S[" in s
    assert "0.20" in s
    assert "0.30" in s
    assert "0.10" in s
    assert "0.40" in s
    assert "0.15" in s


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


def test_sampler_respects_default_ranges() -> None:
    rng = np.random.default_rng(0)
    bmin_lo, bmin_hi = SAMPLE_RANGES["birth_min"]
    smin_lo, smin_hi = SAMPLE_RANGES["survive_min"]
    dens_lo, dens_hi = SAMPLE_RANGES["initial_density"]

    for _ in range(100):
        rule = sample_random_fractional_rule(rng)
        assert bmin_lo <= rule.birth_min <= bmin_hi
        assert smin_lo <= rule.survive_min <= smin_hi
        assert dens_lo <= rule.initial_density <= dens_hi
        # The sampler caps maxes at 0.80 by default.
        assert rule.birth_max <= 0.80 + 1e-12
        assert rule.survive_max <= 0.80 + 1e-12
        # Internal consistency: max >= min.
        assert rule.birth_max >= rule.birth_min
        assert rule.survive_max >= rule.survive_min


def test_sampler_deterministic_with_seed() -> None:
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    seq_a = [sample_random_fractional_rule(rng_a) for _ in range(5)]
    seq_b = [sample_random_fractional_rule(rng_b) for _ in range(5)]
    assert seq_a == seq_b


def test_sampler_custom_ranges() -> None:
    """A custom ``sample_ranges`` override that pins ``birth_min`` to a
    1-point range produces rules with that exact birth_min."""
    rng = np.random.default_rng(7)
    pinned = {"birth_min": (0.30, 0.30)}
    for _ in range(20):
        rule = sample_random_fractional_rule(rng, sample_ranges=pinned)
        assert rule.birth_min == pytest.approx(0.30)
