"""Configuration dataclasses.

These dataclasses are the single source of truth for all run-time defaults.
Every experiment script reads a `RunConfig` (optionally loaded from JSON) and
threads sub-configs into the relevant subsystems.

The defaults here are the documented "sensible defaults" referenced in the
README.  They may not produce interesting dynamics on the first run; rule
search (M4) is the principled way to discover rules with persistent
structures.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------


@dataclass
class WorldConfig:
    """Parameters for the 4D bulk cellular automaton."""

    nx: int = 64
    ny: int = 64
    nz: int = 8
    nw: int = 8
    timesteps: int = 200
    initial_density: float = 0.15
    # Birth/survival sets over neighbour count in the 4D Moore-r1 neighbourhood
    # (max 80 neighbours).  Defaults are heuristic and likely produce trivial
    # dynamics; rule search is the proper way to find interesting rules.
    rule_birth: tuple[int, ...] = (30, 31, 32, 33)
    rule_survival: tuple[int, ...] = (28, 29, 30, 31, 32, 33, 34)
    backend: str = "numba"  # "numba" | "numpy"

    def __post_init__(self) -> None:
        if self.backend not in {"numba", "numpy"}:
            raise ValueError(f"backend must be 'numba' or 'numpy', got {self.backend!r}")
        if not (0.0 <= self.initial_density <= 1.0):
            raise ValueError(f"initial_density must be in [0,1], got {self.initial_density}")
        # Coerce to tuple in case someone passes a list from JSON.
        self.rule_birth = tuple(int(x) for x in self.rule_birth)
        self.rule_survival = tuple(int(x) for x in self.rule_survival)

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (self.nx, self.ny, self.nz, self.nw)


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


@dataclass
class ProjectionConfig:
    """Parameters for the 4D -> 2D projection."""

    method: str = "mean_threshold"  # mean_threshold | sum | parity | max
    theta: float = 0.5

    def __post_init__(self) -> None:
        valid = {"mean_threshold", "sum", "parity", "max"}
        if self.method not in valid:
            raise ValueError(f"projection method must be in {valid}, got {self.method!r}")


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


@dataclass
class DetectionConfig:
    """Parameters for connected-component detection and tracking."""

    connectivity: int = 1  # scipy.ndimage convention; 1 == 4-connectivity in 2D
    min_area: int = 3
    boundary_dilation: int = 1
    environment_dilation: int = 4  # outer radius; (env = dilation-by-env minus dilation-by-bnd)
    iou_threshold: float = 0.3
    centroid_distance_threshold: float = 5.0
    max_gap: int = 2  # frames a track may go unmatched before retirement
    min_age: int = 10  # candidate filter
    max_area_fraction: float = 0.5  # reject "whole grid" structures


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


@dataclass
class OutputConfig:
    """Parameters for run output."""

    output_root: str = "outputs"
    save_4d_snapshots: bool = False
    snapshot_interval: int = 50
    gif_fps: int = 10
    gif_max_frames: int = 200  # cap to keep gifs small
    save_gif: bool = True


# ---------------------------------------------------------------------------
# Top-level run config
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    """Top-level configuration for a single experiment run."""

    world: WorldConfig = field(default_factory=WorldConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    seed: int = 0
    label: str = "run"  # short label used in summary.md

    # ---- (de)serialization ------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=_json_default)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunConfig":
        return cls(
            world=WorldConfig(**data.get("world", {})),
            projection=ProjectionConfig(**data.get("projection", {})),
            detection=DetectionConfig(**data.get("detection", {})),
            output=OutputConfig(**data.get("output", {})),
            seed=int(data.get("seed", 0)),
            label=str(data.get("label", "run")),
        )

    @classmethod
    def from_json(cls, s: str) -> "RunConfig":
        return cls.from_dict(json.loads(s))

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        return cls.from_json(Path(path).read_text())


def _json_default(obj: Any) -> Any:
    """JSON encoder fallback for dataclasses, tuples, and Paths."""
    if is_dataclass(obj):
        return {f.name: getattr(obj, f.name) for f in fields(obj)}
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serializable: {type(obj).__name__}")
