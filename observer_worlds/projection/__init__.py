"""Projection-suite package.

Exposes :class:`ProjectionSuite`, the registry that follow-up topic 1
(projection robustness) uses to evaluate HCE under multiple observation
maps. Lower-level projection implementations live in
``observer_worlds.worlds.projection``; this package adds:

* a registry interface so new projections can be added without touching
  experiment runners,
* per-projection metadata (does it support a threshold-margin? does it
  return uint8 / int / float?),
* skeleton implementations for projections beyond the existing four
  (``random_linear`` and ``multi_channel``).
"""
from __future__ import annotations

from .projection_suite import (
    ProjectionSpec,
    ProjectionSuite,
    default_suite,
)
from .invisible_perturbations import make_projection_invisible_perturbation
from .visible_perturbations import make_projection_visible_perturbation

__all__ = [
    "ProjectionSpec", "ProjectionSuite", "default_suite",
    "make_projection_invisible_perturbation",
    "make_projection_visible_perturbation",
]
