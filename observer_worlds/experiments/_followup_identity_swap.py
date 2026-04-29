"""Workhorse module for Follow-up Topic 2 — hidden identity swap.

For a candidate pair ``(A, B)`` whose 2D projections look similar but
whose hidden 4D substrates differ, we construct a **hybrid state**:
A's substrate, but with B's hidden fibres grafted into the cells of
A's mask (and vice versa for the symmetric pair direction).

Per-cell construction rule
--------------------------

Pairs come from two independent rollouts of the **same rule** with
**different seeds** (cross-substrate). Both rollouts produce a 4D
state at each candidate's peak frame. The hybrid for the pair
direction "A_with_B_hidden" is built like this:

* A's centroid (in 2D) lives at ``c_A``; B's at ``c_B``. Translation
  ``Δ = c_B - c_A`` maps A's mask cells to B's frame.
* For each ``(x, y)`` in A's mask:
  * the corresponding location in B's frame is
    ``(x', y') = (x, y) + Δ``
  * if ``(x', y')`` falls inside B's mask **and** the per-cell
    projected value matches (``Y_alpha[x, y] == Y_beta[x', y']``), the
    hybrid takes B's hidden fibre at ``(x', y')`` (i.e., the full
    ``(z, w)`` plane of state_beta at that location)
  * otherwise the hybrid keeps A's own fibre (no-swap-here, so the
    projection is preserved automatically at that cell)
* Cells outside A's mask are left as state_alpha's fibres.

Because every projection in the registered suite is **cell-independent
in (x, y)** — i.e., projection at ``(x, y)`` depends only on the
``(z, w)`` fibre at that location — the per-cell rule above guarantees
``Y(hybrid_A) == Y(state_alpha)`` exactly within A's mask, regardless
of which projection is used downstream.

If no swap-eligible cell pairs are found (e.g., the per-cell projected
values diverge everywhere), the hybrid equals the original alpha state
and the pair is reported invalid. The runner records the count of
swap-eligible cells (``n_cells_swapped``) per pair so the strength of
the intervention is auditable.

Smoke-quality similarity metric
-------------------------------

For Stage 3 smoke we use a center-anchored cropped-bbox L1 similarity
between binary projected masks at each horizon. Production work
(Stage 5+) can swap in a richer trajectory metric (centroid distance,
shape IoU with translation alignment, observer-score continuation).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from observer_worlds.experiments._followup_projection import (
    CandidateRef, _bbox_mask, binarize_for_detection, detect_candidates,
    initial_4d_state, project_stream, run_substrate,
)
from observer_worlds.projection import default_suite
from observer_worlds.utils.config import DetectionConfig


# ---------------------------------------------------------------------------
# Candidate matching
# ---------------------------------------------------------------------------


@dataclass
class CandidateInCell:
    """A candidate plus the substrate context needed to make a hybrid.

    Stage 5C2 perf fix: the full ``state_stream`` (`(T+1, Nx, Ny, Nz, Nw)`,
    ~65 MB at production grid) is no longer carried by default.
    Instead, ``horizon_projected_frames`` keeps just the projected 2D
    frames at ``peak_frame + h`` for each requested horizon (~32 KB
    total per candidate at production grid). The ~1000× reduction
    keeps joblib IPC cheap during cross-source production sweeps.

    Callers that want the full state stream for debugging may pass
    ``return_state_stream_debug=True`` to
    :func:`discover_candidates_for_cell`; in that case ``state_stream``
    is populated and ``measure_pair`` will prefer it over a fresh
    rollout. The default-``None`` path is the audited production path.
    """
    cell_id: str                 # e.g. "M7_HCE_optimized_rank01|6000"
    rule_id: str
    rule_source: str
    seed: int
    cand: CandidateRef
    state_at_peak: np.ndarray    # 4D state at peak_frame (small: one frame)
    horizon_projected_frames: dict[int, np.ndarray] | None = None
    """Pre-projected 2D frames at ``peak_frame + h`` for h in horizons.
    None when discovery did not pre-project (e.g. legacy callers)."""
    state_stream: np.ndarray | None = None
    """Full 4D state stream; populated only with the debug flag."""


SUPPORTED_MATCH_MODES = (
    "same_area",
    "feature_nearest",
    "morphology_nearest",
)
UNSUPPORTED_MATCH_MODES = (
    "observer_score_bin",
    "mechanism_class",
    "strict_projection_equal",
)


def _morphology_features(c: CandidateInCell) -> np.ndarray:
    """Shape descriptor for matching.

    Returns ``[area, perimeter, bbox_aspect, compactness]``.
      * area: number of ON cells in the peak mask
      * perimeter: count of boundary cells (peak_mask AND NOT eroded)
      * bbox_aspect: bbox_height / bbox_width
      * compactness: 4·π·area / perimeter² (1.0 for perfect disc;
        smaller for elongated shapes)
    """
    import scipy.ndimage as ndi
    m = c.cand.peak_mask.astype(bool)
    area = int(m.sum())
    if area == 0:
        return np.array([0.0, 0.0, 1.0, 0.0])
    eroded = ndi.binary_erosion(m, iterations=1)
    boundary = m & ~eroded
    perim = int(boundary.sum())
    rmin, cmin, rmax, cmax = c.cand.peak_bbox
    h = max(1, rmax - rmin + 1)
    w = max(1, cmax - cmin + 1)
    aspect = h / w
    compact = (4.0 * np.pi * area) / (perim ** 2) if perim > 0 else 0.0
    return np.array([float(area), float(perim), float(aspect),
                     float(compact)])


def _basic_features(c: CandidateInCell) -> np.ndarray:
    """Stage-3 ``feature_nearest`` features: ``[area, lifetime, peak_frame]``."""
    area = float(int(c.cand.peak_mask.sum()))
    return np.array([area, float(c.cand.lifetime),
                     float(c.cand.peak_frame)], dtype=np.float64)


def _crop_to_bbox(mask: np.ndarray) -> np.ndarray:
    """Return the bounding-box crop of a 2D bool/uint8 mask."""
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0:
        return np.zeros((0, 0), dtype=bool)
    return mask[rows.min():rows.max() + 1, cols.min():cols.max() + 1]


def _translation_aligned_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """IoU between two masks after centring each in a common bbox.

    Both masks are cropped to their bounding boxes, padded into a
    common ``(max_h, max_w)`` frame at the centre, and IoU is computed
    on the result. Robust to translation; sensitive to shape and area.
    """
    a = _crop_to_bbox(mask_a.astype(bool))
    b = _crop_to_bbox(mask_b.astype(bool))
    if a.size == 0 or b.size == 0:
        return 0.0
    th = max(a.shape[0], b.shape[0])
    tw = max(a.shape[1], b.shape[1])
    out_a = np.zeros((th, tw), dtype=bool)
    out_b = np.zeros((th, tw), dtype=bool)
    pa = ((th - a.shape[0]) // 2, (tw - a.shape[1]) // 2)
    pb = ((th - b.shape[0]) // 2, (tw - b.shape[1]) // 2)
    out_a[pa[0]:pa[0] + a.shape[0], pa[1]:pa[1] + a.shape[1]] = a
    out_b[pb[0]:pb[0] + b.shape[0], pb[1]:pb[1] + b.shape[1]] = b
    inter = float((out_a & out_b).sum())
    union = float((out_a | out_b).sum())
    return float(inter / union) if union > 0 else 0.0


def _area_ratio(area_a: float, area_b: float) -> float:
    """1.0 = identical area; lower = more dissimilar. ``[0, 1]``."""
    a = max(0.0, float(area_a)); b = max(0.0, float(area_b))
    if a == 0 and b == 0:
        return 1.0
    if a == 0 or b == 0:
        return 0.0
    return min(a, b) / max(a, b)


def _bbox_aspect_similarity(c1: CandidateInCell, c2: CandidateInCell) -> float:
    """1.0 = same aspect ratio; geometric-mean-style similarity."""
    def aspect(c):
        rmin, cmin, rmax, cmax = c.cand.peak_bbox
        h = max(1, rmax - rmin + 1); w = max(1, cmax - cmin + 1)
        return h / w
    a1 = aspect(c1); a2 = aspect(c2)
    return float(min(a1, a2) / max(a1, a2)) if max(a1, a2) > 0 else 1.0


def compute_visible_similarity(a: CandidateInCell, b: CandidateInCell) -> dict:
    """Visible-shape similarity for one candidate pair.

    Returns a dict with the IoU plus the ``area_ratio`` and
    ``bbox_aspect`` components, plus a single ``combined`` score that
    averages them. The combined score is the canonical scalar used
    against the ``min_visible_similarity`` threshold.
    """
    iou = _translation_aligned_iou(a.cand.peak_mask, b.cand.peak_mask)
    ar = _area_ratio(int(a.cand.peak_mask.sum()),
                      int(b.cand.peak_mask.sum()))
    asp = _bbox_aspect_similarity(a, b)
    combined = float((iou + ar + asp) / 3.0)
    return {
        "translation_aligned_iou": float(iou),
        "area_ratio": float(ar),
        "bbox_aspect_similarity": float(asp),
        "combined": combined,
    }


def find_candidate_pairs(
    candidates: list[CandidateInCell], *,
    match_mode: str, max_pairs: int,
    same_rule_only: bool = True,
    different_cell: bool = True,
) -> list[tuple[CandidateInCell, CandidateInCell, dict]]:
    """Pair candidates ``(A, B)`` for the swap intervention.

    * ``same_rule_only`` keeps the dynamics fixed. Cross-rule pairs
      would conflate "different hidden state" with "different rule";
      out of scope here.
    * ``different_cell`` requires A and B to come from different
      ``(rule_id, seed)`` runs so their substrates are independent.
    * ``max_pairs`` caps the number of pairs returned. Pairs are sorted
      by match-quality (lower distance is better).

    The returned ``meta`` dict for each pair always includes
    ``visible_similarity`` (the canonical combined score from
    :func:`compute_visible_similarity`) plus its components, so
    downstream gating on a min-visible-similarity threshold is well-
    defined regardless of match mode.
    """
    if match_mode in UNSUPPORTED_MATCH_MODES:
        raise NotImplementedError(
            f"matching mode {match_mode!r} is not implemented; "
            f"supported modes: {SUPPORTED_MATCH_MODES}"
        )
    if match_mode not in SUPPORTED_MATCH_MODES:
        raise ValueError(
            f"unknown matching mode {match_mode!r}; "
            f"supported: {SUPPORTED_MATCH_MODES}"
        )
    if len(candidates) < 2:
        return []

    if match_mode == "morphology_nearest":
        feats = [_morphology_features(c) for c in candidates]
        # Normalize by std to put features on comparable scales.
        F = np.stack(feats, axis=0)
        sds = F.std(axis=0)
        sds[sds < 1e-12] = 1.0
        feats_norm = F / sds
    elif match_mode == "feature_nearest":
        feats_norm = [_basic_features(c) for c in candidates]
    else:
        feats_norm = None

    pair_records: list[tuple[float, int, int, dict]] = []
    for i, a in enumerate(candidates):
        for j, b in enumerate(candidates):
            if i >= j:
                continue
            if same_rule_only and a.rule_id != b.rule_id:
                continue
            if different_cell and (a.cell_id == b.cell_id):
                continue
            area_a = float(int(a.cand.peak_mask.sum()))
            area_b = float(int(b.cand.peak_mask.sum()))
            if match_mode == "same_area":
                d = abs(area_a - area_b)
            elif match_mode == "feature_nearest":
                d = float(np.linalg.norm(feats_norm[i] - feats_norm[j]))
            elif match_mode == "morphology_nearest":
                d = float(np.linalg.norm(feats_norm[i] - feats_norm[j]))
            else:
                continue
            vs = compute_visible_similarity(a, b)
            pair_records.append((
                d, i, j,
                {
                    "match_distance": float(d),
                    "visible_similarity": float(vs["combined"]),
                    "translation_aligned_iou": float(
                        vs["translation_aligned_iou"]),
                    "area_ratio": float(vs["area_ratio"]),
                    "bbox_aspect_similarity": float(
                        vs["bbox_aspect_similarity"]),
                    "area_a": float(area_a),
                    "area_b": float(area_b),
                },
            ))
    pair_records.sort(key=lambda x: x[0])
    out: list[tuple[CandidateInCell, CandidateInCell, dict]] = []
    for _, i, j, meta in pair_records[: int(max_pairs)]:
        out.append((candidates[i], candidates[j], meta))
    return out


# ---------------------------------------------------------------------------
# Hybrid construction
# ---------------------------------------------------------------------------


@dataclass
class HybridConstruction:
    hybrid_state: np.ndarray
    n_cells_swapped: int
    n_cells_in_mask: int
    projection_preservation_error: float
    accepted: bool
    invalid_reason: str | None
    translation: tuple[int, int]   # (dx, dy) applied: B's frame coord = A's coord + translation


def _project_with_suite(projection_name: str, state_4d: np.ndarray) -> np.ndarray:
    return default_suite().project(projection_name, state_4d)


def _diff_norm(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    if diff.ndim == 3:
        diff = diff.mean(axis=-1)
    return float(diff.mean())


def construct_hybrid(
    state_alpha: np.ndarray, state_beta: np.ndarray,
    mask_a: np.ndarray, mask_b: np.ndarray,
    centroid_a: tuple[float, float], centroid_b: tuple[float, float],
    *, projection_name: str,
    tolerance: float = 1e-6,
) -> HybridConstruction:
    """Graft B's hidden fibres into A's region of state_alpha under
    the per-cell-projection-preservation rule.

    Returns the hybrid plus an audit struct. ``hybrid_state`` equals
    ``state_alpha`` exactly outside A's mask. Inside A's mask, cells
    are either swapped from B or left at A's value, never anything
    else.
    """
    if state_alpha.shape != state_beta.shape:
        return HybridConstruction(
            hybrid_state=state_alpha.copy(),
            n_cells_swapped=0,
            n_cells_in_mask=int(mask_a.sum()),
            projection_preservation_error=0.0,
            accepted=False,
            invalid_reason=(
                f"alpha shape {state_alpha.shape} != beta shape {state_beta.shape}"
            ),
            translation=(0, 0),
        )
    Nx, Ny = state_alpha.shape[:2]
    proj_alpha = _project_with_suite(projection_name, state_alpha)
    proj_beta = _project_with_suite(projection_name, state_beta)

    # Translation from A's frame to B's frame.
    dx = int(round(centroid_b[0] - centroid_a[0]))
    dy = int(round(centroid_b[1] - centroid_a[1]))

    hybrid = state_alpha.copy()
    n_swap = 0
    n_in_mask = int(mask_a.sum())
    for x, y in np.argwhere(mask_a):
        x_b, y_b = int(x) + dx, int(y) + dy
        if x_b < 0 or x_b >= Nx or y_b < 0 or y_b >= Ny:
            continue
        if not mask_b[x_b, y_b]:
            continue
        # Per-cell projection check. Works for binary, count, continuous,
        # and multi-channel because projections are cell-independent in (x, y).
        if proj_alpha.ndim == 3:  # multi-channel
            same = bool(np.array_equal(
                proj_alpha[x, y], proj_beta[x_b, y_b],
            ))
        else:
            v_alpha = proj_alpha[x, y]
            v_beta = proj_beta[x_b, y_b]
            if proj_alpha.dtype.kind in "iu":
                same = bool(int(v_alpha) == int(v_beta))
            else:
                same = bool(abs(float(v_alpha) - float(v_beta)) <= tolerance)
        if not same:
            continue
        hybrid[x, y, :, :] = state_beta[x_b, y_b, :, :]
        n_swap += 1

    # Verify projection preservation on the hybrid (full-grid).
    proj_hybrid = _project_with_suite(projection_name, hybrid)
    delta = _diff_norm(proj_hybrid, proj_alpha)
    accepted = (delta <= tolerance) and (n_swap > 0)
    invalid_reason = None
    if n_swap == 0:
        invalid_reason = (
            "no swap-eligible cells (no overlap of masks under translation, "
            "or per-cell projected values disagreed everywhere)"
        )
        accepted = False
    elif delta > tolerance:
        invalid_reason = (
            f"projection preservation failed (delta={delta:.4g} > "
            f"tolerance={tolerance:.4g})"
        )
        accepted = False
    return HybridConstruction(
        hybrid_state=hybrid,
        n_cells_swapped=n_swap,
        n_cells_in_mask=n_in_mask,
        projection_preservation_error=float(delta),
        accepted=bool(accepted),
        invalid_reason=invalid_reason,
        translation=(dx, dy),
    )


# ---------------------------------------------------------------------------
# Similarity metric (smoke)
# ---------------------------------------------------------------------------


def _crop_bbox(arr_2d: np.ndarray, bbox: tuple[int, int, int, int]
               ) -> np.ndarray:
    rmin, cmin, rmax, cmax = bbox
    return arr_2d[rmin:rmax + 1, cmin:cmax + 1]


def _center_pad(a: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Pad ``a`` to ``target_shape`` centred (zero pad)."""
    out = np.zeros(target_shape, dtype=a.dtype)
    ah, aw = a.shape
    th, tw = target_shape
    ph = max(0, (th - ah) // 2)
    pw = max(0, (tw - aw) // 2)
    out[ph: ph + ah, pw: pw + aw] = a[: min(ah, th), : min(aw, tw)]
    return out


def _binary_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """1 − mean|a − b| over the union of bboxes.

    Both are expected to be uint8 / bool 2D. They're center-aligned to
    a common bounding box first.
    """
    ah, aw = a.shape
    bh, bw = b.shape
    th = max(ah, bh); tw = max(aw, bw)
    a_pad = _center_pad((a > 0).astype(np.uint8), (th, tw))
    b_pad = _center_pad((b > 0).astype(np.uint8), (th, tw))
    diff = float(np.abs(a_pad.astype(np.int16) - b_pad.astype(np.int16)).mean())
    return 1.0 - diff


# ---------------------------------------------------------------------------
# Per-pair measurement
# ---------------------------------------------------------------------------


@dataclass
class IdentityPairResult:
    pair_id: int
    rule_id: str
    rule_source: str
    seed_a: int
    seed_b: int
    projection_name: str
    candidate_a_id: int
    candidate_b_id: int
    match_mode: str
    match_distance: float
    visible_similarity: float
    area_a: int
    area_b: int
    hidden_distance: float                  # mean abs diff of fibres in the swapped region
    n_cells_in_mask_a: int
    n_cells_swapped_a: int
    projection_preservation_error_a: float
    valid_swap_a: bool
    invalid_reason_a: str | None
    n_cells_in_mask_b: int
    n_cells_swapped_b: int
    projection_preservation_error_b: float
    valid_swap_b: bool
    invalid_reason_b: str | None
    horizons: tuple[int, ...]
    host_similarity_a_per_h: list[float]    # one entry per horizon
    donor_similarity_a_per_h: list[float]
    host_similarity_b_per_h: list[float]
    donor_similarity_b_per_h: list[float]
    hidden_identity_pull_a_per_h: list[float]   # donor - host
    hidden_identity_pull_b_per_h: list[float]


def _hidden_distance(state_a: np.ndarray, state_b: np.ndarray,
                      mask_a: np.ndarray, mask_b: np.ndarray,
                      translation: tuple[int, int]) -> float:
    """Mean abs diff of the (z, w) fibres in cells that overlap under
    the translation. Useful as an audit of how different the hidden
    state actually was."""
    Nx, Ny = state_a.shape[:2]
    dx, dy = translation
    diffs = []
    for x, y in np.argwhere(mask_a):
        xb, yb = int(x) + dx, int(y) + dy
        if 0 <= xb < Nx and 0 <= yb < Ny and mask_b[xb, yb]:
            diffs.append(float(np.abs(
                state_a[x, y].astype(np.int16) - state_b[xb, yb].astype(np.int16)
            ).mean()))
    return float(np.mean(diffs)) if diffs else 0.0


def measure_pair(
    *, pair_id: int, A: CandidateInCell, B: CandidateInCell,
    match_meta: dict, match_mode: str,
    projection_name: str,
    horizons: Sequence[int],
    rule_bs,
    backend: str,
    min_visible_similarity: float = 0.0,
) -> IdentityPairResult:
    """Run the swap intervention for one candidate pair."""
    suite = default_suite()
    visible_sim = float(match_meta.get("visible_similarity", 0.0))
    horizons_t = tuple(int(h) for h in horizons)

    # Quality gate: pairs whose visible similarity is below the
    # threshold are marked invalid before any rollout cost is paid.
    if visible_sim < float(min_visible_similarity):
        reason = (
            f"visible_similarity_too_low "
            f"(combined_visible_similarity={visible_sim:.3f} < "
            f"threshold={min_visible_similarity:.3f})"
        )
        return IdentityPairResult(
            pair_id=pair_id, rule_id=A.rule_id, rule_source=A.rule_source,
            seed_a=int(A.seed), seed_b=int(B.seed),
            projection_name=projection_name,
            candidate_a_id=A.cand.candidate_id,
            candidate_b_id=B.cand.candidate_id,
            match_mode=match_mode,
            match_distance=float(match_meta.get("match_distance", 0.0)),
            visible_similarity=visible_sim,
            area_a=int(A.cand.peak_mask.sum()),
            area_b=int(B.cand.peak_mask.sum()),
            hidden_distance=0.0,
            n_cells_in_mask_a=int(A.cand.peak_mask.sum()),
            n_cells_swapped_a=0,
            projection_preservation_error_a=0.0,
            valid_swap_a=False, invalid_reason_a=reason,
            n_cells_in_mask_b=int(B.cand.peak_mask.sum()),
            n_cells_swapped_b=0,
            projection_preservation_error_b=0.0,
            valid_swap_b=False, invalid_reason_b=reason,
            horizons=horizons_t,
            host_similarity_a_per_h=[None] * len(horizons_t),
            donor_similarity_a_per_h=[None] * len(horizons_t),
            host_similarity_b_per_h=[None] * len(horizons_t),
            donor_similarity_b_per_h=[None] * len(horizons_t),
            hidden_identity_pull_a_per_h=[None] * len(horizons_t),
            hidden_identity_pull_b_per_h=[None] * len(horizons_t),
        )

    a_centroid = (
        float(np.mean(np.where(A.cand.peak_mask)[0])),
        float(np.mean(np.where(A.cand.peak_mask)[1])),
    ) if A.cand.peak_mask.any() else (0.0, 0.0)
    b_centroid = (
        float(np.mean(np.where(B.cand.peak_mask)[0])),
        float(np.mean(np.where(B.cand.peak_mask)[1])),
    ) if B.cand.peak_mask.any() else (0.0, 0.0)

    # Build hybrid_A and hybrid_B.
    hyb_a = construct_hybrid(
        A.state_at_peak, B.state_at_peak,
        A.cand.peak_mask.astype(bool), B.cand.peak_mask.astype(bool),
        a_centroid, b_centroid,
        projection_name=projection_name,
    )
    hyb_b = construct_hybrid(
        B.state_at_peak, A.state_at_peak,
        B.cand.peak_mask.astype(bool), A.cand.peak_mask.astype(bool),
        b_centroid, a_centroid,
        projection_name=projection_name,
    )

    hidden_dist = _hidden_distance(
        A.state_at_peak, B.state_at_peak,
        A.cand.peak_mask.astype(bool), B.cand.peak_mask.astype(bool),
        hyb_a.translation,
    )

    max_h = max(horizons_t) if horizons_t else 0

    host_sim_a, donor_sim_a, pull_a = [], [], []
    host_sim_b, donor_sim_b, pull_b = [], [], []

    # If neither hybrid is valid we cannot measure identity pull.
    if not hyb_a.accepted and not hyb_b.accepted:
        host_sim_a = donor_sim_a = pull_a = [None] * len(horizons_t)
        host_sim_b = donor_sim_b = pull_b = [None] * len(horizons_t)
    else:
        # Run hybrids forward.
        from observer_worlds.experiments._followup_projection import (
            _rollout_perturbed,
        )

        def _project_at(state_4d):
            p = suite.project(projection_name, state_4d)
            return binarize_for_detection(
                p, suite.get(projection_name).output_kind,
            )

        for h in horizons_t:
            # Original alpha future at A.peak + h. Three paths in
            # priority order (Stage 5C2):
            #   1. horizon_projected_frames[h] if discovery pre-projected
            #      (production path; tiny IPC; ~32 KB per candidate).
            #   2. state_stream[t] if the debug flag was set during
            #      discovery (legacy / debug only).
            #   3. fresh rollout from state_at_peak as a fallback
            #      (correct but pays an extra rollout per (pair × h)).
            host_a_pre = (A.horizon_projected_frames or {}).get(h)
            if host_a_pre is not None:
                host_a_proj = host_a_pre
            elif A.state_stream is not None:
                t_a = A.cand.peak_frame + h
                avail_a = A.state_stream.shape[0] - 1
                state_a_future = (A.state_stream[t_a] if t_a <= avail_a
                                  else _rollout_perturbed(
                                      rule_bs, A.state_at_peak, h,
                                      backend=backend))
                host_a_proj = _project_at(state_a_future)
            else:
                state_a_future = _rollout_perturbed(
                    rule_bs, A.state_at_peak, h, backend=backend,
                )
                host_a_proj = _project_at(state_a_future)
            host_b_pre = (B.horizon_projected_frames or {}).get(h)
            if host_b_pre is not None:
                host_b_proj = host_b_pre
            elif B.state_stream is not None:
                t_b = B.cand.peak_frame + h
                avail_b = B.state_stream.shape[0] - 1
                state_b_future = (B.state_stream[t_b] if t_b <= avail_b
                                  else _rollout_perturbed(
                                      rule_bs, B.state_at_peak, h,
                                      backend=backend))
                host_b_proj = _project_at(state_b_future)
            else:
                state_b_future = _rollout_perturbed(
                    rule_bs, B.state_at_peak, h, backend=backend,
                )
                host_b_proj = _project_at(state_b_future)

            host_a_crop = _crop_bbox(host_a_proj, A.cand.peak_bbox)
            host_b_crop = _crop_bbox(host_b_proj, B.cand.peak_bbox)

            # A direction
            if hyb_a.accepted:
                state_hyb_a = _rollout_perturbed(
                    rule_bs, hyb_a.hybrid_state, h, backend=backend,
                )
                hyb_a_proj = _project_at(state_hyb_a)
                hyb_a_crop = _crop_bbox(hyb_a_proj, A.cand.peak_bbox)
                ha = _binary_similarity(hyb_a_crop, host_a_crop)
                da = _binary_similarity(hyb_a_crop, host_b_crop)
                host_sim_a.append(ha); donor_sim_a.append(da)
                pull_a.append(da - ha)
            else:
                host_sim_a.append(None); donor_sim_a.append(None)
                pull_a.append(None)
            # B direction
            if hyb_b.accepted:
                state_hyb_b = _rollout_perturbed(
                    rule_bs, hyb_b.hybrid_state, h, backend=backend,
                )
                hyb_b_proj = _project_at(state_hyb_b)
                hyb_b_crop = _crop_bbox(hyb_b_proj, B.cand.peak_bbox)
                hb = _binary_similarity(hyb_b_crop, host_b_crop)
                db = _binary_similarity(hyb_b_crop, host_a_crop)
                host_sim_b.append(hb); donor_sim_b.append(db)
                pull_b.append(db - hb)
            else:
                host_sim_b.append(None); donor_sim_b.append(None)
                pull_b.append(None)

    return IdentityPairResult(
        pair_id=pair_id, rule_id=A.rule_id, rule_source=A.rule_source,
        seed_a=int(A.seed), seed_b=int(B.seed),
        projection_name=projection_name,
        candidate_a_id=A.cand.candidate_id,
        candidate_b_id=B.cand.candidate_id,
        match_mode=match_mode,
        match_distance=float(match_meta["match_distance"]),
        visible_similarity=float(match_meta["visible_similarity"]),
        area_a=int(A.cand.peak_mask.sum()),
        area_b=int(B.cand.peak_mask.sum()),
        hidden_distance=float(hidden_dist),
        n_cells_in_mask_a=hyb_a.n_cells_in_mask,
        n_cells_swapped_a=hyb_a.n_cells_swapped,
        projection_preservation_error_a=hyb_a.projection_preservation_error,
        valid_swap_a=hyb_a.accepted,
        invalid_reason_a=hyb_a.invalid_reason,
        n_cells_in_mask_b=hyb_b.n_cells_in_mask,
        n_cells_swapped_b=hyb_b.n_cells_swapped,
        projection_preservation_error_b=hyb_b.projection_preservation_error,
        valid_swap_b=hyb_b.accepted,
        invalid_reason_b=hyb_b.invalid_reason,
        horizons=horizons_t,
        host_similarity_a_per_h=host_sim_a,
        donor_similarity_a_per_h=donor_sim_a,
        host_similarity_b_per_h=host_sim_b,
        donor_similarity_b_per_h=donor_sim_b,
        hidden_identity_pull_a_per_h=pull_a,
        hidden_identity_pull_b_per_h=pull_b,
    )


# ---------------------------------------------------------------------------
# Top-level (rule × seeds) discovery + pairing + measurement
# ---------------------------------------------------------------------------


def discover_candidates_for_cell(
    *, rule_bs, rule_id: str, rule_source: str, seed: int,
    grid_shape, timesteps: int, backend: str,
    projection_name: str, max_candidates: int,
    initial_density: float = 0.5,
    detection_config: DetectionConfig | None = None,
    horizons: tuple[int, ...] | None = None,
    return_state_stream_debug: bool = False,
) -> list[CandidateInCell]:
    """Run substrate, project, detect, return CandidateInCell records.

    Stage 5C2 perf fix: by default, only the candidate's
    ``state_at_peak`` plus pre-projected 2D frames at
    ``peak_frame + h`` for each ``h in horizons`` are stored on each
    record (~64 KB per candidate at production grid). The full 4D
    state stream stays local to the worker and is GC'd before the
    record is IPC'd back to the parent.

    Pass ``return_state_stream_debug=True`` to retain the full state
    stream on every record (legacy behaviour; only useful for
    debugging — joblib IPC of a 65 MB array per candidate scales badly).
    """
    det_cfg = detection_config or DetectionConfig()
    state0 = initial_4d_state(grid_shape, initial_density, seed=seed)
    stream = run_substrate(rule_bs, state0, timesteps, backend=backend)
    suite = default_suite()
    proj_stream = project_stream(suite, projection_name, stream)
    output_kind = suite.get(projection_name).output_kind
    binary_frames = np.stack([
        binarize_for_detection(proj_stream[t], output_kind)
        for t in range(proj_stream.shape[0])
    ], axis=0)
    cand_refs = detect_candidates(
        binary_frames, det_cfg=det_cfg, max_candidates=max_candidates,
    )
    cell_id = f"{rule_id}|{seed}"
    horizons_t = tuple(int(h) for h in (horizons or ()))
    avail = stream.shape[0] - 1
    out = []
    for cr in cand_refs:
        # Pre-project the host's future at each requested horizon. Use
        # binarize_for_detection so the host_similarity comparison sees
        # the same kind of array as the hybrid rollouts produce.
        per_h: dict[int, np.ndarray] = {}
        for h in horizons_t:
            t = int(cr.peak_frame) + int(h)
            if 0 <= t <= avail:
                per_h[int(h)] = binarize_for_detection(
                    suite.project(projection_name, stream[t]), output_kind,
                )
        record = CandidateInCell(
            cell_id=cell_id, rule_id=rule_id, rule_source=rule_source,
            seed=int(seed), cand=cr,
            state_at_peak=stream[cr.peak_frame].copy(),
            horizon_projected_frames=per_h if per_h else None,
            state_stream=stream if return_state_stream_debug else None,
        )
        out.append(record)
    return out
