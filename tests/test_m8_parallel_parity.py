"""Parity test: parallel run_m8_mechanism_discovery produces field-identical
M8CandidateResult lists to the serial implementation."""
from __future__ import annotations

from dataclasses import asdict

from observer_worlds.experiments._m8_mechanism import run_m8_mechanism_discovery
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


def _strip_for_compare(result_list):
    """M8CandidateResult contains nested numpy arrays inside ResponseMap and
    others, which don't compare cleanly. Compare only scalar identity fields
    plus the mechanism label."""
    out = []
    for r in result_list:
        out.append({
            "rule_id": r.rule_id,
            "rule_source": r.rule_source,
            "seed": r.seed,
            "candidate_id": r.candidate_id,
            "snapshot_t": r.snapshot_t,
            "mechanism_label": r.mechanism.label,
            "mechanism_confidence": r.mechanism.confidence,
            "near_threshold_fraction": r.near_threshold_fraction,
            "first_visible_effect_time": r.timing.first_visible_effect_time,
            "boundary_response_fraction": r.response_map.boundary_response_fraction,
            "interior_response_fraction": r.response_map.interior_response_fraction,
        })
    out.sort(key=lambda d: (d["rule_id"], d["seed"], d["candidate_id"]))
    return out


def test_run_m8_mechanism_discovery_parallel_matches_serial(tmp_path):
    rules = _tiny_rules()
    seeds = [1000, 1001]
    common = dict(
        rules=rules, seeds=seeds,
        grid_shape=(8, 8, 4, 4),
        timesteps=20,
        max_candidates=2,
        horizons=[3, 5, 8],
        n_replicates=2,
        backend="numba",
    )
    serial = run_m8_mechanism_discovery(
        **common,
        workdir=tmp_path / "serial",
        n_workers=1,
    )
    parallel = run_m8_mechanism_discovery(
        **common,
        workdir=tmp_path / "parallel",
        n_workers=2,
    )
    assert _strip_for_compare(serial) == _strip_for_compare(parallel)
