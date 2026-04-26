"""Parity test: parallel run_sweep produces field-identical PairedRecords
to the serial implementation for fixed seeds and a tiny grid."""
from __future__ import annotations

from dataclasses import asdict

from observer_worlds.experiments._m4b_sweep import run_sweep
from observer_worlds.search import FractionalRule
from observer_worlds.utils.config import DetectionConfig
from observer_worlds.worlds import BSRule


def _tiny_rules(n: int) -> list[FractionalRule]:
    """Fixed deterministic FractionalRule fixtures."""
    return [
        FractionalRule(
            birth_min=0.20, birth_max=0.30,
            survive_min=0.18, survive_max=0.40,
            initial_density=0.30,
        )
        for _ in range(n)
    ]


def _records_to_dict(records):
    """Strip frames_for_video (numpy arrays don't dict-compare cleanly) and
    return a list of nested dicts."""
    out = []
    for rec in records:
        d = {
            "rule_idx": rec.rule_idx,
            "seed": rec.seed,
            "rule_dict": rec.rule_dict,
        }
        for cond in ("coherent_4d", "shuffled_4d", "matched_2d"):
            cond_res = getattr(rec, cond)
            cd = asdict(cond_res)
            cd.pop("frames_for_video", None)
            cd.pop("sim_time_seconds", None)  # timing varies
            cd.pop("metric_time_seconds", None)
            d[cond] = cd
        out.append(d)
    return out


def test_run_sweep_parallel_matches_serial():
    rules = _tiny_rules(2)
    seeds = [1000, 1001]
    common = dict(
        rules=rules, seeds=seeds,
        grid_shape_4d=(8, 8, 4, 4),
        grid_shape_2d=(8, 8),
        timesteps=20,
        initial_density_2d=0.30,
        detection_config=DetectionConfig(),
        backend="numba",
        rollout_steps=2,
        rule_2d=BSRule.life(),
        video_frames_kept=0,
        snapshots_per_run=1,
    )
    serial = run_sweep(**common, n_workers=1)
    parallel = run_sweep(**common, n_workers=2)
    assert _records_to_dict(serial) == _records_to_dict(parallel)
