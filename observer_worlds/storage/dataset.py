from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import zarr

from observer_worlds.utils.config import RunConfig

if TYPE_CHECKING:  # pragma: no cover - type-hint only imports
    from observer_worlds.detection.tracking import Track
    from observer_worlds.metrics.persistence import CandidateScore


class ZarrRunStore:
    """Per-run storage backed by zarr v2.

    Layout produced under ``run_dir``::

        run_dir/
          config.json
          summary.md
          data/
            states.zarr/
              frames_2d              # (T, Nx, Ny) uint8
              snapshots_4d/          # group; only created if save_4d_snapshots
                t000050              # (Nx, Ny, Nz, Nw) uint8
            tracks.csv
            candidates.csv
          frames/                    # gif lives here (created by analysis/videos)
          plots/                     # png plots (created by analysis/plots)
    """

    def __init__(
        self,
        run_dir: str | Path,
        timesteps: int,
        shape_2d: tuple[int, int],
        save_4d_snapshots: bool = False,
        shape_4d: tuple[int, int, int, int] | None = None,
    ) -> None:
        self._run_dir = Path(run_dir)
        self._data_dir = self._run_dir / "data"
        self._frames_dir = self._run_dir / "frames"
        self._plots_dir = self._run_dir / "plots"

        for d in (self._run_dir, self._data_dir, self._frames_dir, self._plots_dir):
            d.mkdir(parents=True, exist_ok=True)

        self._timesteps = int(timesteps)
        self._shape_2d = (int(shape_2d[0]), int(shape_2d[1]))
        self._save_4d_snapshots = bool(save_4d_snapshots)
        self._shape_4d = tuple(shape_4d) if shape_4d is not None else None

        if self._save_4d_snapshots and self._shape_4d is None:
            raise ValueError("shape_4d must be provided when save_4d_snapshots=True")

        # Pre-allocate the zarr store.
        self._zarr_path = self._data_dir / "states.zarr"
        self._root = zarr.open(str(self._zarr_path), mode="w")

        nx, ny = self._shape_2d
        self._frames_2d = self._root.create_dataset(
            "frames_2d",
            shape=(self._timesteps, nx, ny),
            chunks=(1, nx, ny),
            dtype="uint8",
        )

        if self._save_4d_snapshots:
            self._snapshots_group = self._root.create_group("snapshots_4d")
        else:
            self._snapshots_group = None

    # ----------------------------------------------------------------- props

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    @property
    def frames_dir(self) -> Path:
        return self._frames_dir

    @property
    def plots_dir(self) -> Path:
        return self._plots_dir

    # ------------------------------------------------------------- writers

    def write_frame_2d(self, t: int, frame: np.ndarray) -> None:
        if frame.shape != self._shape_2d:
            raise ValueError(
                f"frame shape {frame.shape!r} does not match expected "
                f"{self._shape_2d!r}"
            )
        self._frames_2d[t] = frame.astype(np.uint8, copy=False)

    def write_snapshot_4d(self, t: int, state: np.ndarray) -> None:
        if not self._save_4d_snapshots or self._snapshots_group is None:
            raise RuntimeError(
                "ZarrRunStore was not created with save_4d_snapshots=True"
            )
        if self._shape_4d is not None and state.shape != self._shape_4d:
            raise ValueError(
                f"snapshot shape {state.shape!r} does not match expected "
                f"{self._shape_4d!r}"
            )
        name = f"t{int(t):06d}"
        # Overwrite if it already exists.
        if name in self._snapshots_group:
            del self._snapshots_group[name]
        arr = self._snapshots_group.create_dataset(
            name,
            shape=state.shape,
            chunks=state.shape,
            dtype="uint8",
        )
        arr[...] = state.astype(np.uint8, copy=False)

    def write_config_json(self, config: RunConfig) -> None:
        (self._run_dir / "config.json").write_text(config.to_json())

    def write_tracks_csv(self, tracks: list) -> None:
        path = self._data_dir / "tracks.csv"
        columns = [
            "track_id",
            "frame",
            "centroid_y",
            "centroid_x",
            "area",
            "bbox_rmin",
            "bbox_cmin",
            "bbox_rmax",
            "bbox_cmax",
            "interior_count",
            "boundary_count",
            "env_count",
        ]
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            for tr in tracks:
                frames = list(tr.frames)
                for i, frame in enumerate(frames):
                    centroid_y, centroid_x = tr.centroid_history[i]
                    area = int(tr.area_history[i])
                    rmin, cmin, rmax, cmax = tr.bbox_history[i]
                    interior_count = int(np.asarray(tr.interior_history[i]).sum())
                    boundary_count = int(np.asarray(tr.boundary_history[i]).sum())
                    env_count = int(np.asarray(tr.env_history[i]).sum())
                    writer.writerow(
                        [
                            int(tr.track_id),
                            int(frame),
                            float(centroid_y),
                            float(centroid_x),
                            area,
                            int(rmin),
                            int(cmin),
                            int(rmax),
                            int(cmax),
                            interior_count,
                            boundary_count,
                            env_count,
                        ]
                    )

    def write_candidates_csv(self, candidates: list) -> None:
        path = self._data_dir / "candidates.csv"
        columns = [
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
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            for c in candidates:
                reasons = getattr(c, "reasons", None)
                if reasons is None:
                    reasons_str = ""
                elif isinstance(reasons, str):
                    reasons_str = reasons
                else:
                    reasons_str = ";".join(str(r) for r in reasons)
                writer.writerow(
                    [
                        int(c.track_id),
                        int(c.age),
                        int(c.length),
                        bool(c.is_candidate),
                        float(c.boundedness),
                        float(c.internal_variation),
                        float(c.mean_area),
                        float(c.max_area),
                        reasons_str,
                    ]
                )

    def write_summary_md(self, content: str) -> None:
        (self._run_dir / "summary.md").write_text(content)

    # ------------------------------------------------------------- helpers

    @staticmethod
    def make_run_dir(output_root: str | Path, label: str = "run") -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = Path(output_root) / f"{label}_{stamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    # -------------------------------------------------------------- readers

    def read_frames_2d(self) -> np.ndarray:
        return np.asarray(self._frames_2d[:])

    def list_snapshots(self) -> list[int]:
        """Return the sorted list of timestep indices for which a 4D snapshot exists."""
        if self._snapshots_group is None:
            return []
        out: list[int] = []
        for name in self._snapshots_group:
            if name.startswith("t"):
                try:
                    out.append(int(name[1:]))
                except ValueError:
                    continue
        return sorted(out)

    def read_snapshot_4d(self, t: int) -> np.ndarray:
        """Return the 4D snapshot at timestep ``t`` as a numpy uint8 array."""
        if self._snapshots_group is None:
            raise RuntimeError(
                "ZarrRunStore was not created with save_4d_snapshots=True"
            )
        name = f"t{int(t):06d}"
        if name not in self._snapshots_group:
            raise KeyError(f"no snapshot at t={t}")
        return np.asarray(self._snapshots_group[name][:])
