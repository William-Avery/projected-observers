"""Candidate morphology classification for M8B.

M8 saw most candidates at <10 cells. At that size erosion gives an
empty interior, and the shell-mask code falls back to using the entire
mask as the boundary — boundary and interior become indistinguishable.
M8B requires explicit morphology gating before any boundary-vs-interior
mechanism claim is made.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.ndimage as ndi


MORPHOLOGY_CLASSES: tuple[str, ...] = (
    "very_thick_candidate",
    "thick_candidate",
    "thin_candidate",
    "degenerate",
)


@dataclass
class MorphologyResult:
    morphology_class: str
    area: int
    erosion1_interior_size: int
    erosion2_interior_size: int
    boundary_size: int
    environment_size: int
    can_separate_boundary_from_interior: bool
    can_classify_environment_coupled: bool


def classify_morphology(
    mask: np.ndarray,
    *,
    env_dilation: int = 3,
    min_thick_area: int = 25,
    min_very_thick_area: int = 50,
) -> MorphologyResult:
    """Classify a 2D candidate mask into one of four morphology classes.

    The contract these classes guarantee, used by the M8B mechanism
    classifier:

    - ``very_thick_candidate``: erosion(r=2) leaves a non-empty interior;
      boundary ring (mask XOR erosion(r=1)) is at least 2 cells thick on
      average; environment shell is non-empty. Boundary, interior, and
      environment can all be probed independently.
    - ``thick_candidate``: erosion(r=1) leaves a non-empty interior;
      boundary distinguishable; environment shell non-empty. Suitable
      for a clean boundary-vs-interior decomposition but the boundary
      ring may be thin.
    - ``thin_candidate``: erosion gives an empty interior. Candidate-
      locality can still be measured; boundary-vs-interior cannot.
    - ``degenerate``: empty mask, single cell, or otherwise unusable.
    """
    if mask is None:
        return MorphologyResult(
            morphology_class="degenerate", area=0,
            erosion1_interior_size=0, erosion2_interior_size=0,
            boundary_size=0, environment_size=0,
            can_separate_boundary_from_interior=False,
            can_classify_environment_coupled=False,
        )

    mask = np.asarray(mask).astype(bool)
    area = int(mask.sum())
    if area <= 1:
        return MorphologyResult(
            morphology_class="degenerate", area=area,
            erosion1_interior_size=0, erosion2_interior_size=0,
            boundary_size=0, environment_size=0,
            can_separate_boundary_from_interior=False,
            can_classify_environment_coupled=False,
        )

    interior_e1 = ndi.binary_erosion(mask, iterations=1)
    interior_e2 = ndi.binary_erosion(mask, iterations=2)
    boundary = mask & ~interior_e1
    outer_dilation = ndi.binary_dilation(mask, iterations=env_dilation)
    inner_dilation = ndi.binary_dilation(mask, iterations=1)
    environment = outer_dilation & ~inner_dilation & ~mask

    e1_size = int(interior_e1.sum())
    e2_size = int(interior_e2.sum())
    bnd_size = int(boundary.sum())
    env_size = int(environment.sum())

    if (area >= min_very_thick_area
            and e2_size > 0 and bnd_size > 0 and env_size > 0):
        cls = "very_thick_candidate"
        can_sep = True
        can_env = True
    elif (area >= min_thick_area
          and e1_size > 0 and bnd_size > 0 and env_size > 0):
        cls = "thick_candidate"
        can_sep = True
        can_env = True
    elif e1_size > 0 and bnd_size > 0:
        cls = "thin_candidate"
        can_sep = False
        can_env = env_size > 0
    else:
        cls = "thin_candidate"
        can_sep = False
        can_env = env_size > 0

    return MorphologyResult(
        morphology_class=cls, area=area,
        erosion1_interior_size=e1_size,
        erosion2_interior_size=e2_size,
        boundary_size=bnd_size, environment_size=env_size,
        can_separate_boundary_from_interior=can_sep,
        can_classify_environment_coupled=can_env,
    )


def shell_masks_strict(
    mask: np.ndarray, *,
    erosion_radius: int = 1,
    env_dilation: int = 3,
) -> dict:
    """Return a dict of all spatial regions used by M8B region-aware
    response maps.

    Unlike M8's ``_shell_masks``, this never falls back to "boundary =
    interior_mask" for thin candidates — if interior is empty the
    boundary ring is also defined to be empty (callers must use the
    morphology gate to decide whether to ignore the result).
    """
    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        z = np.zeros_like(mask)
        return {"interior": z, "boundary": z, "environment": z, "whole": z}
    eroded = ndi.binary_erosion(mask, iterations=erosion_radius)
    boundary = mask & ~eroded
    interior = eroded
    outer = ndi.binary_dilation(mask, iterations=env_dilation)
    inner = ndi.binary_dilation(mask, iterations=1)
    environment = outer & ~inner & ~mask
    return {
        "interior": interior,
        "boundary": boundary,
        "environment": environment,
        "whole": mask,
    }


def far_mask(
    mask: np.ndarray, *, translation: tuple[int, int] | None = None,
) -> np.ndarray:
    """Construct a far-region mask: the candidate translated to the
    antipode of the (periodic) grid. Same shape and area as input."""
    Nx, Ny = mask.shape
    if translation is None:
        translation = (Nx // 2, Ny // 2)
    return np.roll(mask, shift=translation, axis=(0, 1)).astype(bool)
