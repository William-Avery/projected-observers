"""Parity test: parallel run_m6c_taxonomy produces field-identical
M6CRow lists to the serial implementation."""
from __future__ import annotations

from dataclasses import asdict

from observer_worlds.experiments._m6c_taxonomy import run_m6c_taxonomy
from observer_worlds.search import FractionalRule


def _tiny_rules() -> list[tuple[FractionalRule, str, str]]:
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


def test_run_m6c_taxonomy_parallel_matches_serial(tmp_path):
    rules = _tiny_rules()
    seeds = [1000, 1001]
    common = dict(
        rules=rules, seeds=seeds,
        grid_shape=(8, 8, 4, 4),
        timesteps=20,
        max_candidates=2,
        horizons=[3, 5],
        n_replicates=2,
        backend="numba",
    )
    serial = run_m6c_taxonomy(
        **common,
        workdir=str(tmp_path / "serial"),
        n_workers=1,
    )
    parallel = run_m6c_taxonomy(
        **common,
        workdir=str(tmp_path / "parallel"),
        n_workers=2,
    )

    # Sort by stable identity fields. Read M6CRow definition in
    # observer_worlds/experiments/_m6c_taxonomy.py to confirm the field
    # names; adjust the sort key tuple if needed.
    def key(d):
        return tuple(d.get(k) for k in (
            "rule_id", "seed", "candidate_id", "snapshot_t", "horizon",
        ))

    s_dicts = sorted([asdict(r) for r in serial], key=key)
    p_dicts = sorted([asdict(r) for r in parallel], key=key)
    assert s_dicts == p_dicts
