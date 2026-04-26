"""Unit tests for the cross-run analysis tool."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from observer_worlds.analysis.summarize_results import (
    load_run,
    load_runs,
    plot_baseline_comparison,
    plot_observer_score_histogram,
    plot_score_vs_age,
    summary_table,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


_OBSERVER_COLUMNS = [
    "track_id",
    "combined",
    "n_components_used",
    "time_raw",
    "memory_raw",
    "selfhood_raw",
    "causality_raw",
    "resilience_raw",
    "time_z",
    "memory_z",
    "selfhood_z",
    "causality_z",
    "resilience_z",
    "boundary_predictability",
    "extra_env_given_boundary",
    "persistence",
    "boundedness",
    "sensory_fraction",
    "active_fraction",
]

_CANDIDATE_COLUMNS = [
    "track_id",
    "age",
    "length",
    "is_candidate",
    "boundedness",
    "internal_variation",
    "mean_area",
    "max_area",
    "reasons",
]


def _stub_config(label: str, kind: str) -> dict:
    nx = ny = 16
    nz = nw = 1 if kind == "2d" else 4
    return {
        "world": {
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "nw": nw,
            "timesteps": 40,
            "initial_density": 0.15,
            "rule_birth": [30, 31, 32, 33],
            "rule_survival": [28, 29, 30, 31, 32, 33, 34],
            "backend": "numpy",
        },
        "projection": {"method": "mean_threshold", "theta": 0.5},
        "detection": {
            "connectivity": 1,
            "min_area": 3,
            "boundary_dilation": 1,
            "environment_dilation": 4,
            "iou_threshold": 0.3,
            "centroid_distance_threshold": 5.0,
            "max_gap": 2,
            "min_age": 10,
            "max_area_fraction": 0.5,
        },
        "output": {
            "output_root": "outputs",
            "save_4d_snapshots": False,
            "snapshot_interval": 10,
            "gif_fps": 10,
            "gif_max_frames": 200,
            "save_gif": False,
        },
        "seed": 0,
        "label": label,
    }


def _summary_md(label: str, kind: str) -> str:
    return (
        f"# Run summary — {label} ({kind})\n"
        "\n"
        f"- Run dir: `outputs/{label}`\n"
        f"- World kind: **{kind}**\n"
        "- Grid: (16, 16, 4, 4)\n"
        "- Timesteps: 40\n"
        "- Seed: 0\n"
    )


def _make_fake_run(
    tmp_path: Path,
    label: str,
    kind: str,
    n_candidates: int,
    scores: list[float] | None = None,
) -> Path:
    """Create a minimal run directory layout under ``tmp_path/<label>``.

    - ``summary.md`` contains a ``World kind: **<kind>**`` line.
    - ``config.json`` is a stub satisfying the loader.
    - ``data/observer_scores.csv`` has one row per candidate. ``scores`` (if
      provided) supplies the ``combined`` column; otherwise scores increment
      from 0.0.
    - ``data/candidates.csv`` has ``n_candidates`` rows with ``is_candidate=True``
      and synthetic ages.
    - ``data/tracks.csv`` is a header-only file (n_tracks==0) — keeping the
      fixture cheap; tests asserting on ``n_tracks`` should account for this.
    """
    run_dir = tmp_path / label
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "summary.md").write_text(_summary_md(label, kind))
    (run_dir / "config.json").write_text(json.dumps(_stub_config(label, kind), indent=2))

    if scores is None:
        scores = [float(i) for i in range(n_candidates)]
    if len(scores) != n_candidates:
        raise ValueError("len(scores) must equal n_candidates")

    # observer_scores.csv: write n_candidates rows with combined=score, all
    # other numeric columns left empty. The loader treats empty cells as None.
    obs_lines = [",".join(_OBSERVER_COLUMNS)]
    for i, s in enumerate(scores):
        cells = [""] * len(_OBSERVER_COLUMNS)
        cells[_OBSERVER_COLUMNS.index("track_id")] = str(i)
        cells[_OBSERVER_COLUMNS.index("combined")] = f"{s:.6f}"
        cells[_OBSERVER_COLUMNS.index("n_components_used")] = "3"
        obs_lines.append(",".join(cells))
    (data_dir / "observer_scores.csv").write_text("\n".join(obs_lines) + "\n")

    # candidates.csv: n_candidates "True" rows + one extra non-candidate.
    cand_lines = [",".join(_CANDIDATE_COLUMNS)]
    for i in range(n_candidates):
        cand_lines.append(
            f"{i},{20 + i},{20 + i},True,0.5,0.5,10.0,12.0,"
        )
    cand_lines.append("999,5,5,False,0.1,0.1,3.0,4.0,too_short")
    (data_dir / "candidates.csv").write_text("\n".join(cand_lines) + "\n")

    # tracks.csv: header only (cheap — n_tracks ends up as 0).
    (data_dir / "tracks.csv").write_text(
        "track_id,frame,centroid_y,centroid_x,area,bbox_rmin,bbox_cmin,"
        "bbox_rmax,bbox_cmax,interior_count,boundary_count,env_count\n"
    )
    return run_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_load_run_extracts_world_kind(tmp_path: Path):
    run_dir = _make_fake_run(tmp_path, "r0", "4d", n_candidates=0)

    rs = load_run(run_dir)

    assert rs.world_kind == "4d"
    assert rs.n_candidates == 0
    assert rs.label == "r0"
    assert rs.observer_scores == []
    assert rs.candidate_ages == []
    assert rs.run_dir == run_dir


def test_load_run_unknown_world_kind(tmp_path: Path):
    run_dir = _make_fake_run(tmp_path, "rx", "4d", n_candidates=0)
    # Overwrite summary.md with one that lacks the World kind marker.
    (run_dir / "summary.md").write_text("# nothing here\n")

    rs = load_run(run_dir)
    assert rs.world_kind == "unknown"


def test_load_run_parses_observer_scores_floats_and_nones(tmp_path: Path):
    run_dir = _make_fake_run(
        tmp_path, "r1", "4d", n_candidates=1, scores=[0.42]
    )

    rs = load_run(run_dir)
    assert len(rs.observer_scores) == 1
    row = rs.observer_scores[0]
    assert row["combined"] == pytest.approx(0.42)
    # An empty-cell column should become None.
    assert row["time_raw"] is None


def test_load_runs_skips_incomplete(tmp_path: Path):
    _make_fake_run(tmp_path, "complete_run", "4d", n_candidates=1)

    bad_dir = tmp_path / "incomplete_run"
    bad_dir.mkdir()
    (bad_dir / "summary.md").write_text(_summary_md("incomplete_run", "4d"))
    # No config.json, no observer_scores.csv — should be skipped.

    runs = load_runs(tmp_path)
    assert len(runs) == 1
    assert runs[0].label == "complete_run"


def test_load_runs_sorted_by_dirname(tmp_path: Path):
    _make_fake_run(tmp_path, "b_run", "4d", n_candidates=0)
    _make_fake_run(tmp_path, "a_run", "2d", n_candidates=0)
    _make_fake_run(tmp_path, "c_run", "shuffled_4d", n_candidates=0)

    runs = load_runs(tmp_path)
    assert [r.run_dir.name for r in runs] == ["a_run", "b_run", "c_run"]


def test_summary_table_groups_by_kind(tmp_path: Path):
    _make_fake_run(tmp_path, "r4d_a", "4d", n_candidates=2, scores=[1.0, 2.0])
    _make_fake_run(tmp_path, "r4d_b", "4d", n_candidates=2, scores=[3.0, 4.0])
    _make_fake_run(tmp_path, "r2d_a", "2d", n_candidates=2, scores=[0.5, 1.5])

    runs = load_runs(tmp_path)
    table = summary_table(runs)

    assert "4d" in table
    assert "2d" in table
    assert table["4d"]["n_runs"] == 2
    assert table["4d"]["n_candidates"] == 4
    assert table["2d"]["n_runs"] == 1
    assert table["2d"]["n_candidates"] == 2
    assert table["4d"]["mean_score"] == pytest.approx(2.5)
    assert table["4d"]["max_score"] == pytest.approx(4.0)
    assert table["2d"]["mean_score"] == pytest.approx(1.0)


def test_summary_table_handles_kind_without_scores(tmp_path: Path):
    # Single 4d run with zero candidates: scores should be NaN, n_runs=1.
    _make_fake_run(tmp_path, "empty_4d", "4d", n_candidates=0)
    runs = load_runs(tmp_path)
    table = summary_table(runs)
    import math

    assert table["4d"]["n_runs"] == 1
    assert table["4d"]["n_candidates"] == 0
    assert math.isnan(table["4d"]["mean_score"])  # type: ignore[arg-type]
    assert math.isnan(table["4d"]["mean_age"])  # type: ignore[arg-type]


def test_plots_render_to_files(tmp_path: Path):
    _make_fake_run(tmp_path, "r4d", "4d", n_candidates=3, scores=[0.1, 0.5, 0.9])
    _make_fake_run(tmp_path, "r2d", "2d", n_candidates=2, scores=[-0.2, 0.3])
    runs = load_runs(tmp_path)

    out_dir = tmp_path / "_summary"
    out_dir.mkdir()
    hist = out_dir / "hist.png"
    scatter = out_dir / "scatter.png"
    box = out_dir / "box.png"

    plot_observer_score_histogram(runs, hist)
    plot_score_vs_age(runs, scatter)
    plot_baseline_comparison(runs, box)

    for p in (hist, scatter, box):
        assert p.exists(), p
        assert p.stat().st_size > 0, p


def test_plots_render_with_no_runs(tmp_path: Path):
    out_dir = tmp_path / "_summary"
    out_dir.mkdir()
    hist = out_dir / "hist.png"
    scatter = out_dir / "scatter.png"
    box = out_dir / "box.png"

    plot_observer_score_histogram([], hist)
    plot_score_vs_age([], scatter)
    plot_baseline_comparison([], box)

    for p in (hist, scatter, box):
        assert p.exists()
        assert p.stat().st_size > 0


def test_plots_render_with_empty_kind_panel(tmp_path: Path):
    # 4d has candidates, 2d does not — histogram should still render.
    _make_fake_run(tmp_path, "r4d", "4d", n_candidates=2, scores=[0.0, 1.0])
    _make_fake_run(tmp_path, "r2d", "2d", n_candidates=0)
    runs = load_runs(tmp_path)

    out = tmp_path / "hist.png"
    plot_observer_score_histogram(runs, out)
    assert out.exists() and out.stat().st_size > 0
