from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import imageio.v2 as imageio
import numpy as np

try:  # PIL is pulled in by matplotlib; tolerate its absence.
    from PIL import Image, ImageDraw

    _HAVE_PIL = True
except Exception:  # pragma: no cover - defensive fallback
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]
    _HAVE_PIL = False

if TYPE_CHECKING:  # pragma: no cover
    from observer_worlds.detection.tracking import Track


# Colors as RGB tuples.
_INACTIVE_GRAY = (40, 40, 40)
_ACTIVE_WHITE = (240, 240, 240)
_CANDIDATE_COLOR = (220, 50, 50)  # red
_NORMAL_COLOR = (240, 220, 60)  # yellow


def _binary_to_rgb(frame: np.ndarray, upsample: int) -> np.ndarray:
    """Convert a binary {0,1} frame to an upscaled RGB array (Nx*u, Ny*u, 3)."""
    binary = (np.asarray(frame) > 0).astype(np.uint8)
    rgb = np.empty(binary.shape + (3,), dtype=np.uint8)
    rgb[binary == 0] = _INACTIVE_GRAY
    rgb[binary == 1] = _ACTIVE_WHITE
    if upsample > 1:
        rgb = np.kron(rgb, np.ones((upsample, upsample, 1), dtype=np.uint8))
    return rgb


def _draw_overlays(
    rgb: np.ndarray,
    tracks: list,
    t: int,
    upsample: int,
    candidate_track_ids: set[int] | None,
) -> np.ndarray:
    if not _HAVE_PIL or not tracks:
        return rgb

    img = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(img)
    cand_ids = candidate_track_ids if candidate_track_ids is not None else set()

    for tr in tracks:
        frames = list(tr.frames)
        if t not in frames:
            continue
        i = frames.index(t)
        cy, cx = tr.centroid_history[i]
        rmin, cmin, rmax, cmax = tr.bbox_history[i]

        color = _CANDIDATE_COLOR if int(tr.track_id) in cand_ids else _NORMAL_COLOR

        # Scale to pixel coords.  bbox is (rmin, cmin, rmax, cmax) in (row, col).
        x0 = int(cmin) * upsample
        y0 = int(rmin) * upsample
        x1 = int(cmax) * upsample - 1
        y1 = int(rmax) * upsample - 1
        if x1 < x0:
            x1 = x0
        if y1 < y0:
            y1 = y0
        draw.rectangle([x0, y0, x1, y1], outline=color, width=1)

        # Centroid dot (pixel coords: x = col * u, y = row * u).
        px = int(round(float(cx) * upsample))
        py = int(round(float(cy) * upsample))
        r = max(1, upsample // 2)
        draw.ellipse([px - r, py - r, px + r, py + r], fill=color)

        # Track id label.
        label = str(int(tr.track_id))
        tx = px + r + 1
        ty = max(0, py - r - 8)
        draw.text((tx, ty), label, fill=color)

    return np.asarray(img)


def write_projected_gif(
    frames: np.ndarray,
    tracks: list | None,
    out_path: str | Path,
    fps: int = 10,
    max_frames: int = 200,
    upsample: int = 4,
    candidate_track_ids: set[int] | None = None,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if frames.ndim != 3:
        raise ValueError(
            f"write_projected_gif expects (T, Nx, Ny) frames, got shape "
            f"{frames.shape!r}"
        )

    total = int(frames.shape[0])
    n = min(total, int(max_frames))
    if n <= 0:
        # Nothing to write; emit a 1-frame placeholder.
        nx = max(int(frames.shape[1]), 1)
        ny = max(int(frames.shape[2]), 1)
        placeholder = np.zeros((nx * upsample, ny * upsample, 3), dtype=np.uint8)
        imageio.mimsave(str(out_path), [placeholder], fps=fps, loop=0)
        return

    rendered: list[np.ndarray] = []
    for t in range(n):
        rgb = _binary_to_rgb(frames[t], upsample=upsample)
        if tracks is not None:
            rgb = _draw_overlays(
                rgb,
                tracks=tracks,
                t=t,
                upsample=upsample,
                candidate_track_ids=candidate_track_ids,
            )
        rendered.append(rgb)

    imageio.mimsave(str(out_path), rendered, fps=fps, loop=0)
