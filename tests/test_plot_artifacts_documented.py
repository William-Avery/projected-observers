"""Lightweight check that plots referenced in `docs/M8_plot_inventory.md`
exist in at least one local M8E-style run directory under `outputs/`.

The check is best-effort: `outputs/` is gitignored and may be empty on
CI or a fresh clone. In that case the test skips rather than fails.
The point is to catch documentation drift on a developer's machine
where the artifacts are present.

The test never renders or opens an image.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
INVENTORY = REPO / "docs" / "M8_plot_inventory.md"
OUTPUTS = REPO / "outputs"


def _extract_plot_filenames(md_text: str) -> set[str]:
    """Pull names like `plots/m8f_within_class_hce_ci.png` out of the
    inventory's headings and tables."""
    return set(re.findall(r"plots/([A-Za-z0-9_]+\.png)", md_text))


def _find_m8e_like_dirs() -> list[Path]:
    """Return the local run dirs that look like M8E or M8 mechanism
    discovery output (contain a plots/ subdir with at least one .png)."""
    if not OUTPUTS.exists():
        return []
    dirs = []
    for d in OUTPUTS.iterdir():
        if not d.is_dir():
            continue
        plots_dir = d / "plots"
        if not plots_dir.exists():
            continue
        if any(p.suffix == ".png" for p in plots_dir.iterdir()):
            dirs.append(d)
    return dirs


def test_inventory_doc_exists():
    assert INVENTORY.exists(), (
        f"plot inventory missing at {INVENTORY}"
    )


def test_inventory_lists_plot_filenames():
    """Sanity check that the inventory actually names some plot files —
    so an empty doc doesn't silently pass the artifact-existence check
    by listing nothing to look for."""
    names = _extract_plot_filenames(INVENTORY.read_text(encoding="utf-8"))
    assert len(names) >= 5, (
        f"plot inventory should reference at least 5 plot files; "
        f"found {len(names)}: {sorted(names)}"
    )


def test_inventory_plots_present_in_at_least_one_local_run():
    """For every plot filename mentioned in the inventory, check that
    the file exists in at least one local run directory. Skips
    gracefully on a fresh clone where outputs/ is empty."""
    run_dirs = _find_m8e_like_dirs()
    if not run_dirs:
        pytest.skip(
            "no local M8-style run directory under outputs/; "
            "skipping artifact existence check (outputs/ is gitignored)"
        )

    inventory_names = _extract_plot_filenames(
        INVENTORY.read_text(encoding="utf-8")
    )
    if not inventory_names:
        pytest.skip("plot inventory references no .png files")

    missing: dict[str, list[str]] = {}
    for d in run_dirs:
        plots = {p.name for p in (d / "plots").iterdir() if p.suffix == ".png"}
        gap = inventory_names - plots
        if gap:
            missing[str(d.name)] = sorted(gap)

    # Pass if at least one local run has every documented plot.
    have_all = [d for d in run_dirs
                if not (inventory_names - {p.name for p in (d / "plots").iterdir()
                                           if p.suffix == ".png"})]
    assert have_all, (
        f"no local run directory under outputs/ contains every plot "
        f"referenced in {INVENTORY.relative_to(REPO)}. Missing per dir:\n"
        + "\n".join(f"  {k}: {v}" for k, v in missing.items())
    )
