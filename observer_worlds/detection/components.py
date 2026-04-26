"""Connected-component extraction with interior/boundary/environment shells.

Given a 2D bool/uint8 frame, :func:`extract_components` returns one
:class:`Component` per connected blob (subject to ``config.min_area``).  Each
component carries full-grid bool masks for the blob itself, its interior, its
boundary shell, and a surrounding environment shell -- wasteful in memory but
trivial to consume downstream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.ndimage as ndi

from observer_worlds.utils.config import DetectionConfig


@dataclass
class Component:
    """A single connected component in one 2D frame.

    ``mask``, ``interior_mask``, ``boundary_mask``, ``environment_mask`` are
    all full-grid bool arrays of shape ``(Nx, Ny)``.  This is wasteful for
    memory but makes downstream IoU and per-cell indexing trivial -- fine at
    this scale.
    """

    component_id: int
    frame: int
    mask: np.ndarray
    interior_mask: np.ndarray
    boundary_mask: np.ndarray
    environment_mask: np.ndarray
    area: int
    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]
    perimeter: float
    # Active cells in the env shell (env_mask & frame).  Distinct from the
    # geometric shell size (``env_mask.sum()``) -- this is the actual sensory
    # quantity available to the candidate.  Defaults to 0 for callers that
    # don't pass the frame.
    env_active_count: int = 0


def extract_components(
    frame: np.ndarray,
    frame_idx: int,
    config: DetectionConfig,
) -> list[Component]:
    """Detect connected components and build interior/boundary/environment shells.

    See module docstring for details.  Components with ``area <
    config.min_area`` are dropped.  The returned ``component_id`` is a
    frame-local integer starting at 0.
    """
    if frame.ndim != 2:
        raise ValueError(
            f"extract_components expects a 2D frame, got {frame.ndim}D "
            f"(shape={frame.shape!r})"
        )

    binary = frame.astype(bool)
    structure = ndi.generate_binary_structure(2, config.connectivity)
    labels, n_labels = ndi.label(binary, structure=structure)

    components: list[Component] = []
    next_id = 0
    for lbl in range(1, n_labels + 1):
        mask = labels == lbl
        area = int(mask.sum())
        if area < config.min_area:
            continue

        # Interior: erode by boundary_dilation iterations.  When erosion
        # collapses the blob entirely, interior is just empty (a tiny blob is
        # all boundary).
        if config.boundary_dilation > 0:
            interior_mask = ndi.binary_erosion(
                mask, structure=structure, iterations=config.boundary_dilation
            )
        else:
            interior_mask = mask.copy()

        boundary_mask = mask & ~interior_mask

        # Environment: shell of width (env_dilation - boundary_dilation)
        # outside the component itself.  We dilate by env_dilation to get the
        # outer boundary, dilate by boundary_dilation to define the "inner
        # offset", then take the shell strictly outside ``mask``.
        if config.environment_dilation > 0:
            dilated_outer = ndi.binary_dilation(
                mask, structure=structure, iterations=config.environment_dilation
            )
        else:
            dilated_outer = mask.copy()

        if config.boundary_dilation > 0:
            dilated_inner = ndi.binary_dilation(
                mask, structure=structure, iterations=config.boundary_dilation
            )
        else:
            dilated_inner = mask.copy()

        environment_mask = dilated_outer & ~dilated_inner & ~mask

        # Centroid: mean of the active cell coordinates.
        rows, cols = np.where(mask)
        centroid = (float(rows.mean()), float(cols.mean()))

        rmin, rmax = int(rows.min()), int(rows.max()) + 1
        cmin, cmax = int(cols.min()), int(cols.max()) + 1
        bbox = (rmin, cmin, rmax, cmax)

        # Perimeter: cheap surrogate -- count boundary cells.
        perimeter = float(int(boundary_mask.sum()))

        # Active env cells = cells in the env shell that are active in the
        # underlying frame (the candidate's "sensory" view).
        env_active_count = int((environment_mask & binary).sum())

        components.append(
            Component(
                component_id=next_id,
                frame=frame_idx,
                mask=mask,
                interior_mask=interior_mask,
                boundary_mask=boundary_mask,
                environment_mask=environment_mask,
                area=area,
                centroid=centroid,
                bbox=bbox,
                perimeter=perimeter,
                env_active_count=env_active_count,
            )
        )
        next_id += 1

    return components
