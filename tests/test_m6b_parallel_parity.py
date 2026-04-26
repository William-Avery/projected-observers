"""Parity test: parallel run_m6b_replication produces field-identical
M6BRow lists to the serial implementation for fixed seeds and a tiny grid."""
from __future__ import annotations

from dataclasses import asdict

from observer_worlds.experiments._m6b_replication import run_m6b_replication
from observer_worlds.search import FractionalRule


def _tiny_rules() -> list[tuple[FractionalRule, str, str]]:
    """Two structurally distinct deterministic rules."""
    r0 = FractionalRule(
        birth_min=0.20, birth_max=0.32,
        survive_min=0.18, survive_max=0.40,
        initial_density=0.30,
    )
    r1 = FractionalRule(
        birth_min=0.22, birth_max=0.30,
        survive_min=0.16, survive_max=0.42,
        initial_density=0.30,
    )
    return [(r0, "rule0", "test"), (r1, "rule1", "test")]


def _rows_to_dicts(rows):
    """Drop fields that legitimately vary across runs (none expected
    for M6BRow, but be defensive)."""
    return [asdict(r) for r in rows]


def test_run_m6b_replication_parallel_matches_serial(tmp_path):
    rules = _tiny_rules()
    seeds = [1000, 1001]
    common = dict(
        rules=rules, seeds=seeds,
        grid_shape=(8, 8, 4, 4),
        timesteps=20,
        max_candidates_per_mode=2,
        horizons=[3, 5],
        n_replicates=2,
        backend="numba",
        include_per_step_shuffled=True,
    )
    serial = run_m6b_replication(
        **common,
        workdir_for_zarr=str(tmp_path / "serial"),
        n_workers=1,
    )
    parallel = run_m6b_replication(
        **common,
        workdir_for_zarr=str(tmp_path / "parallel"),
        n_workers=2,
    )

    sort_key = lambda d: (
        d["rule_id"], d["seed"], d["condition"],
        d["candidate_selection_mode"], d["candidate_id"],
        d["intervention_type"], d["replicate"], d["horizon"],
    )
    serial_rows = sorted(_rows_to_dicts(serial), key=sort_key)
    parallel_rows = sorted(_rows_to_dicts(parallel), key=sort_key)
    assert serial_rows == parallel_rows
