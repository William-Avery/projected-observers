"""Combined observer-likeness score.

The single per-track scalar produced by this module is a weighted sum of five
component scores: time, memory, selfhood, causality, resilience.  Components
that could not be computed for a track (because the track was too short, or
because no 4D snapshot was available) contribute 0 to the sum and have their
weights effectively redistributed by the normalization step.

This is **not** a consciousness score.  It is a functional observer-likeness
score that ranks tracks by how many primitive observer-like properties they
exhibit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from observer_worlds.metrics.causality_score import CausalityResult
    from observer_worlds.metrics.memory_score import MemoryScoreResult
    from observer_worlds.metrics.resilience_score import ResilienceResult
    from observer_worlds.metrics.selfhood_score import SelfhoodScoreResult
    from observer_worlds.metrics.time_score import TimeScoreResult


# Default weights — equal contribution from each component after per-run
# z-normalization across candidates.  A user can override via the `weights`
# argument; missing keys fall back to these defaults.
DEFAULT_WEIGHTS: dict[str, float] = {
    "time": 1.0,
    "memory": 1.0,
    "selfhood": 1.0,
    "causality": 1.0,
    "resilience": 1.0,
}


@dataclass
class ObserverScore:
    """Per-track combined observer-likeness score.

    Each ``*_raw`` field is the score as returned by the corresponding
    metric function (None if the metric was not computed).  The
    ``*_normalized`` fields are z-scored across the population in
    :func:`compute_observer_scores`, then clipped to a sensible range for
    summing.  ``combined`` is the weighted sum of the normalized components
    (weighted by ``weights``).
    """

    track_id: int

    # Raw (unormalized) scores from each component metric, or None if missing.
    time_raw: float | None
    memory_raw: float | None
    selfhood_raw: float | None
    causality_raw: float | None
    resilience_raw: float | None

    # Per-population z-normalized values (None if the raw was None).
    time_normalized: float | None
    memory_normalized: float | None
    selfhood_normalized: float | None
    causality_normalized: float | None
    resilience_normalized: float | None

    # Effective weights actually used for this track's combined score
    # (zeroed where the raw score was missing).
    weights_used: dict[str, float] = field(default_factory=dict)

    combined: float = 0.0
    n_components_used: int = 0


# ---------------------------------------------------------------------------
# Per-track score assembly (without normalization)
# ---------------------------------------------------------------------------


def collect_raw_scores(
    *,
    track_id: int,
    time: "TimeScoreResult | None" = None,
    memory: "MemoryScoreResult | None" = None,
    selfhood: "SelfhoodScoreResult | None" = None,
    causality: "CausalityResult | None" = None,
    resilience: "ResilienceResult | None" = None,
) -> dict[str, float | None]:
    """Pull the raw scalar score out of each result, or None if invalid/missing.

    Returns a dict with keys 'time', 'memory', 'selfhood', 'causality',
    'resilience'.
    """
    return {
        "track_id": track_id,
        "time": time.time_score if (time is not None and time.valid) else None,
        "memory": (
            memory.memory_score if (memory is not None and memory.valid) else None
        ),
        "selfhood": (
            selfhood.selfhood_score
            if (selfhood is not None and selfhood.valid)
            else None
        ),
        "causality": (
            causality.causality_score
            if (causality is not None and causality.valid)
            else None
        ),
        "resilience": (
            resilience.resilience_score
            if (resilience is not None and resilience.valid)
            else None
        ),
    }


# ---------------------------------------------------------------------------
# Population normalization + combine
# ---------------------------------------------------------------------------


def _zscore_with_missing(
    values: list[float | None], clip: float = 3.0
) -> list[float | None]:
    """Z-score a list that may contain None.  Non-None entries are
    standardized over the non-None population; the result is clipped to
    +/- ``clip``.  None entries pass through.

    If the non-None population has zero variance (e.g. only one valid
    track, or all identical), all non-None outputs are 0.0.
    """
    finite = [v for v in values if v is not None and np.isfinite(v)]
    if len(finite) < 2:
        return [0.0 if v is not None and np.isfinite(v) else None for v in values]
    arr = np.asarray(finite, dtype=np.float64)
    mu = float(arr.mean())
    sd = float(arr.std())
    if sd < 1e-12:
        return [0.0 if v is not None and np.isfinite(v) else None for v in values]
    out: list[float | None] = []
    for v in values:
        if v is None or not np.isfinite(v):
            out.append(None)
        else:
            z = (v - mu) / sd
            out.append(float(np.clip(z, -clip, clip)))
    return out


def compute_observer_scores(
    raw_per_track: list[dict[str, float | None]],
    *,
    weights: dict[str, float] | None = None,
) -> list[ObserverScore]:
    """Combine per-track raw scores into normalized :class:`ObserverScore` s.

    `raw_per_track` is a list of dicts as returned by :func:`collect_raw_scores`,
    one per track.  Each dict has a 'track_id' int and component keys.

    Normalization: each component is z-scored across the population of valid
    tracks (None entries skipped); z-scores are clipped to +/- 3.

    Combine: for each track, the combined score is the weighted average of the
    normalized component scores that are not None.  Missing components have
    their weight effectively redistributed -- we divide by the sum of weights
    actually used.  If no components are available, combined is 0.0.
    """
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    components = ("time", "memory", "selfhood", "causality", "resilience")

    n = len(raw_per_track)
    if n == 0:
        return []

    # Per-component lists for z-scoring.
    cols = {c: [r[c] for r in raw_per_track] for c in components}
    norms = {c: _zscore_with_missing(cols[c]) for c in components}

    out: list[ObserverScore] = []
    for i, r in enumerate(raw_per_track):
        weights_used: dict[str, float] = {}
        weighted_sum = 0.0
        weight_total = 0.0
        for c in components:
            z = norms[c][i]
            if z is None:
                weights_used[c] = 0.0
                continue
            w = weights[c]
            weights_used[c] = w
            weighted_sum += w * z
            weight_total += w
        combined = weighted_sum / weight_total if weight_total > 0 else 0.0
        out.append(
            ObserverScore(
                track_id=int(r["track_id"]),
                time_raw=r["time"],
                memory_raw=r["memory"],
                selfhood_raw=r["selfhood"],
                causality_raw=r["causality"],
                resilience_raw=r["resilience"],
                time_normalized=norms["time"][i],
                memory_normalized=norms["memory"][i],
                selfhood_normalized=norms["selfhood"][i],
                causality_normalized=norms["causality"][i],
                resilience_normalized=norms["resilience"][i],
                weights_used=weights_used,
                combined=float(combined),
                n_components_used=sum(1 for w in weights_used.values() if w > 0),
            )
        )
    return out
