"""M6C — hidden-state feature extraction.

For each (x,y) column under a candidate's interior at intervention
snapshot t, computes structural features of the z,w fiber. The
candidate-level aggregates (mean, std, min, max, heterogeneity, spatial
coherence) summarize the whole footprint, and a small set of
**projection-threshold sensitivity** features lets us check whether HCE
is just a projection-mechanics artifact ("column close to threshold →
small bit-flip flips projection") or genuinely reflects hidden
organization.

The feature set is deliberately interpretable: every row in the
returned DataFrame-shaped dict is one column or one candidate, with
named columns matching the spec.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.ndimage as ndi


# Threshold for "near threshold" classification. With theta=0.5 and 64
# z,w cells, a column has active_fraction in {0/64, 1/64, ..., 64/64}.
# A `threshold_margin = abs(active_fraction - 0.5)` of 0.10 corresponds
# to ~6 cells away from the 32-cell midpoint.
NEAR_THRESHOLD_MARGIN: float = 0.10


# ---------------------------------------------------------------------------
# Per-column features
# ---------------------------------------------------------------------------


def _binary_entropy(p: float) -> float:
    """Shannon entropy of a Bernoulli(p) in bits."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))


def _spatial_autocorrelation_2d(grid: np.ndarray) -> float:
    """Lag-1 spatial autocorrelation of a 2D bool/uint8 array.

    Computed as Pearson correlation between the cell value and the mean
    of its 4-neighbours (with periodic wrap). Returns 0 if the grid is
    constant (no variance).
    """
    g = grid.astype(np.float64)
    if g.std() < 1e-12:
        return 0.0
    # Roll-based mean of the 4 neighbours.
    nb = (np.roll(g, 1, 0) + np.roll(g, -1, 0)
          + np.roll(g, 1, 1) + np.roll(g, -1, 1)) / 4.0
    if nb.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(g.ravel(), nb.ravel())[0, 1])


def column_features(
    fiber: np.ndarray,
    *,
    theta: float = 0.5,
) -> dict:
    """Extract per-column features from one (z,w) fiber.

    ``fiber``: 2D bool/uint8 array of shape (Nz, Nw).

    Returned dict keys (all spec'd names):
      active_count, active_fraction, distance_to_projection_threshold,
      hidden_entropy, hidden_n_components, hidden_largest_component,
      hidden_spatial_autocorrelation, hidden_perimeter,
      hidden_compactness, hidden_parity, hidden_variance,
      threshold_margin, projection_value
    """
    n_total = fiber.size
    active_count = int(fiber.sum())
    active_frac = float(active_count) / n_total
    proj_value = 1 if active_frac > theta else 0
    margin = abs(active_frac - theta)

    # Connected components within the (z,w) fiber (4-connectivity).
    bool_fiber = fiber.astype(bool)
    if bool_fiber.any():
        structure = ndi.generate_binary_structure(2, 1)
        labels, n_comp = ndi.label(bool_fiber, structure=structure)
        comp_sizes = np.bincount(labels.ravel())[1:]  # skip background
        largest = int(comp_sizes.max()) if comp_sizes.size else 0
    else:
        n_comp, largest = 0, 0

    # Hidden perimeter: count edges between active and inactive cells.
    # Uses XOR with rolled neighbours, summed over both axes.
    if 0 < active_count < n_total:
        diff_z = np.bitwise_xor(bool_fiber, np.roll(bool_fiber, 1, axis=0))
        diff_w = np.bitwise_xor(bool_fiber, np.roll(bool_fiber, 1, axis=1))
        perimeter = int(diff_z.sum() + diff_w.sum())
    else:
        perimeter = 0

    # Compactness: 4*pi*area / perimeter^2 (unitless, in [0,1] roughly).
    if perimeter > 0:
        compactness = 4.0 * np.pi * active_count / (perimeter ** 2)
    else:
        compactness = 1.0 if active_count > 0 else 0.0

    return {
        "active_count": active_count,
        "active_fraction": active_frac,
        "projection_value": proj_value,
        "threshold_margin": float(margin),
        "distance_to_projection_threshold": float(margin),
        "hidden_entropy": _binary_entropy(active_frac),
        "hidden_n_components": int(n_comp),
        "hidden_largest_component": largest,
        "hidden_spatial_autocorrelation": _spatial_autocorrelation_2d(bool_fiber),
        "hidden_perimeter": perimeter,
        "hidden_compactness": float(compactness),
        "hidden_parity": int(active_count % 2),
        "hidden_variance": float(bool_fiber.astype(np.float64).var()),
    }


# ---------------------------------------------------------------------------
# Candidate-level aggregates
# ---------------------------------------------------------------------------


def _hidden_heterogeneity_across_columns(fibers: np.ndarray) -> float:
    """Mean pairwise Hamming distance (normalized) between columns under
    the candidate.

    ``fibers``: 2D array of shape (n_columns, n_fiber_cells), bool.
    Returns 0 if fewer than 2 columns.
    """
    n_cols, n_cells = fibers.shape
    if n_cols < 2:
        return 0.0
    # Pairwise XOR sums = number of differing cells per pair.
    # Vectorize: flatten to (n_cols, n_cells) and compute pairwise diffs.
    # For small candidates (~10-20 columns) this is cheap.
    a = fibers.astype(np.uint8)
    A = a[:, None, :]   # (n, 1, cells)
    B = a[None, :, :]   # (1, n, cells)
    diffs = np.bitwise_xor(A, B).sum(axis=-1)
    # Average upper triangle, normalize by n_cells.
    iu = np.triu_indices(n_cols, k=1)
    pair_dists = diffs[iu] / n_cells
    return float(pair_dists.mean()) if pair_dists.size else 0.0


def _hidden_connectedness_across_adjacent_columns(
    state_4d: np.ndarray,
    interior_mask: np.ndarray,
) -> float:
    """For each pair of (x,y) interior cells that are 2D-adjacent
    (sharing an edge), compute mean correlation of their z,w fibers.
    Returns 0 if no adjacent pairs.
    """
    coords = np.argwhere(interior_mask)
    if coords.shape[0] < 2:
        return 0.0
    coord_set = {(int(r), int(c)) for r, c in coords}
    correlations = []
    Nx, Ny = interior_mask.shape
    for r, c in coords:
        for dr, dc in ((1, 0), (0, 1)):  # right/down only (avoid double-count)
            r2, c2 = (r + dr) % Nx, (c + dc) % Ny
            if (int(r2), int(c2)) not in coord_set:
                continue
            f1 = state_4d[r, c].ravel().astype(np.float64)
            f2 = state_4d[r2, c2].ravel().astype(np.float64)
            if f1.std() > 1e-12 and f2.std() > 1e-12:
                correlations.append(float(np.corrcoef(f1, f2)[0, 1]))
    return float(np.mean(correlations)) if correlations else 0.0


def candidate_hidden_features(
    state_4d: np.ndarray,
    interior_mask: np.ndarray,
    *,
    theta: float = 0.5,
    near_threshold_margin: float = NEAR_THRESHOLD_MARGIN,
) -> dict:
    """Extract candidate-level hidden features from the 4D state at the
    intervention snapshot.

    Aggregates per-column features across all (x,y) under
    ``interior_mask``, plus computes cross-column quantities.

    Returns a flat dict suitable for joining with HCE measurement rows.
    """
    Nx, Ny, Nz, Nw = state_4d.shape
    coords = np.argwhere(interior_mask)
    if coords.shape[0] == 0:
        return _empty_candidate_features()

    # Per-column features.
    col_feats: list[dict] = []
    fibers_flat: list[np.ndarray] = []
    for r, c in coords:
        fiber = state_4d[r, c]
        col_feats.append(column_features(fiber, theta=theta))
        fibers_flat.append(fiber.ravel())
    fibers_arr = np.array(fibers_flat, dtype=np.uint8)

    # Aggregate scalar features.
    af = np.array([f["active_fraction"] for f in col_feats])
    margins = np.array([f["threshold_margin"] for f in col_feats])
    ents = np.array([f["hidden_entropy"] for f in col_feats])
    n_comps = np.array([f["hidden_n_components"] for f in col_feats])
    largest = np.array([f["hidden_largest_component"] for f in col_feats])
    autocorr = np.array([f["hidden_spatial_autocorrelation"] for f in col_feats])
    perim = np.array([f["hidden_perimeter"] for f in col_feats])
    compact = np.array([f["hidden_compactness"] for f in col_feats])
    parity = np.array([f["hidden_parity"] for f in col_feats])
    variance = np.array([f["hidden_variance"] for f in col_feats])
    proj_vals = np.array([f["projection_value"] for f in col_feats])

    near_thresh_frac = float((margins < near_threshold_margin).mean())
    # Expected projection flip probability under one random hidden bit flip.
    # A flip changes active_count by ±1, so active_fraction shifts by 1/n_total.
    # Projection flips if the new fraction crosses the theta threshold.
    n_fiber_cells = Nz * Nw
    delta = 1.0 / n_fiber_cells
    flip_probs = []
    for f in col_feats:
        af_v = f["active_fraction"]
        # Active cells can flip down: P_down = active_frac
        # Inactive cells can flip up:  P_up   = 1 - active_frac
        # New fraction cross theta iff:
        crosses_down = (af_v > theta) and (af_v - delta) <= theta
        crosses_up   = (af_v <= theta) and (af_v + delta) > theta
        p_flip = (af_v if crosses_down else 0.0) + ((1.0 - af_v) if crosses_up else 0.0)
        flip_probs.append(p_flip)
    flip_probs = np.array(flip_probs)

    return {
        # Column-level aggregates.
        "n_columns": int(coords.shape[0]),
        "mean_active_fraction": float(af.mean()),
        "std_active_fraction": float(af.std()),
        "min_active_fraction": float(af.min()),
        "max_active_fraction": float(af.max()),
        "mean_threshold_margin": float(margins.mean()),
        "min_threshold_margin": float(margins.min()),
        "near_threshold_fraction": near_thresh_frac,
        "mean_hidden_entropy": float(ents.mean()),
        "std_hidden_entropy": float(ents.std()),
        "mean_hidden_n_components": float(n_comps.mean()),
        "mean_hidden_largest_component": float(largest.mean()),
        "mean_hidden_spatial_autocorrelation": float(autocorr.mean()),
        "mean_hidden_perimeter": float(perim.mean()),
        "mean_hidden_compactness": float(compact.mean()),
        "fraction_hidden_parity_odd": float(parity.mean()),
        "mean_hidden_variance": float(variance.mean()),
        "fraction_columns_above_threshold": float(proj_vals.mean()),
        # Cross-column features.
        "hidden_heterogeneity": _hidden_heterogeneity_across_columns(fibers_arr),
        "hidden_connectedness_across_columns": _hidden_connectedness_across_adjacent_columns(
            state_4d, interior_mask
        ),
        # Threshold sensitivity.
        "mean_projection_flip_probability": float(flip_probs.mean()),
        "max_projection_flip_probability": float(flip_probs.max()),
    }


def _empty_candidate_features() -> dict:
    return {k: 0.0 for k in (
        "n_columns", "mean_active_fraction", "std_active_fraction",
        "min_active_fraction", "max_active_fraction",
        "mean_threshold_margin", "min_threshold_margin",
        "near_threshold_fraction", "mean_hidden_entropy",
        "std_hidden_entropy", "mean_hidden_n_components",
        "mean_hidden_largest_component",
        "mean_hidden_spatial_autocorrelation",
        "mean_hidden_perimeter", "mean_hidden_compactness",
        "fraction_hidden_parity_odd", "mean_hidden_variance",
        "fraction_columns_above_threshold",
        "hidden_heterogeneity", "hidden_connectedness_across_columns",
        "mean_projection_flip_probability",
        "max_projection_flip_probability",
    )}


# ---------------------------------------------------------------------------
# Temporal features (require a sequence of snapshots of the same fibers)
# ---------------------------------------------------------------------------


def temporal_hidden_features(
    snapshots_at_t: list[np.ndarray],
    interior_mask: np.ndarray,
    *,
    snapshot_times: list[int],
) -> dict:
    """Compute hidden-temporal features from a list of snapshots in
    chronological order.

    `snapshots_at_t[i]` is the 4D state at `snapshot_times[i]`.

    Returns:
      hidden_temporal_persistence: mean (1 - normalized hamming distance)
        between successive fibers under interior_mask
      hidden_temporal_volatility: mean normalized hamming distance
        between successive fibers (= 1 - persistence)

    Returns zero values if fewer than 2 snapshots.
    """
    if len(snapshots_at_t) < 2:
        return {
            "hidden_temporal_persistence": 0.0,
            "hidden_temporal_volatility": 0.0,
            "n_snapshots_used": len(snapshots_at_t),
        }
    coords = np.argwhere(interior_mask)
    if coords.shape[0] == 0:
        return {
            "hidden_temporal_persistence": 0.0,
            "hidden_temporal_volatility": 0.0,
            "n_snapshots_used": 0,
        }
    Nz = snapshots_at_t[0].shape[2]
    Nw = snapshots_at_t[0].shape[3]
    n_cells = Nz * Nw
    persistences = []
    for i in range(len(snapshots_at_t) - 1):
        a = snapshots_at_t[i]
        b = snapshots_at_t[i + 1]
        diffs = []
        for r, c in coords:
            d = int(np.bitwise_xor(a[r, c], b[r, c]).sum())
            diffs.append(d / n_cells)
        persistences.append(1.0 - float(np.mean(diffs)))
    return {
        "hidden_temporal_persistence": float(np.mean(persistences)),
        "hidden_temporal_volatility": float(1.0 - np.mean(persistences)),
        "n_snapshots_used": len(snapshots_at_t),
    }


# ---------------------------------------------------------------------------
# Convenience: list of all feature names (used by the regression module)
# ---------------------------------------------------------------------------


HIDDEN_FEATURE_NAMES: tuple[str, ...] = (
    "n_columns",
    "mean_active_fraction", "std_active_fraction",
    "min_active_fraction", "max_active_fraction",
    "mean_threshold_margin", "min_threshold_margin",
    "near_threshold_fraction",
    "mean_hidden_entropy", "std_hidden_entropy",
    "mean_hidden_n_components", "mean_hidden_largest_component",
    "mean_hidden_spatial_autocorrelation",
    "mean_hidden_perimeter", "mean_hidden_compactness",
    "fraction_hidden_parity_odd", "mean_hidden_variance",
    "fraction_columns_above_threshold",
    "hidden_heterogeneity", "hidden_connectedness_across_columns",
    "mean_projection_flip_probability", "max_projection_flip_probability",
    "hidden_temporal_persistence", "hidden_temporal_volatility",
)
