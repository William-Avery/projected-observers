"""Memory observer-likeness score.

Functional question: does the candidate's *internal* state add predictive
power for future sensory states beyond the current sensory state alone?

We compare two ridge regressions sharing the same target:

    baseline:    S_t          ->  S_{t+k}
    augmented:   [S_t, I_t]   ->  S_{t+k}

If the candidate's internal state encodes useful "memory" of the past that
the current sensory snapshot does not contain, the augmented model will do
better.  The score

    M = error_S_only - error_S_plus_I

is positive when internal state is informative.

Implementation mirrors :mod:`observer_worlds.metrics.time_score`: ridge
regression with K-fold CV and per-fold standardisation.  X is standardised
separately for each model (the two models have different feature
dimensions); y is standardised the same way for both (per-fold mean/std on
train) so MSEs are directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from observer_worlds.metrics.features import TrackFeatures


@dataclass
class MemoryScoreResult:
    track_id: int
    memory_score: float         # error_S_only - error_S_plus_I (positive = useful memory)
    error_s_only: float         # MSE: S_t -> S_{t+k}
    error_s_plus_i: float       # MSE: (S_t, I_t) -> S_{t+k}
    horizon: int                # k
    n_train: int
    valid: bool
    reason: str


def _contiguous_horizon_indices(frames: np.ndarray, horizon: int) -> np.ndarray:
    """Return indices ``i`` such that frames ``i, i+1, ..., i+horizon`` are
    all consecutive (every diff == 1).
    """
    n = frames.shape[0]
    if n <= horizon:
        return np.empty(0, dtype=np.int64)
    diffs = np.diff(frames)  # length n-1
    # We need diffs[i], diffs[i+1], ..., diffs[i+horizon-1] all == 1.
    # Build a sliding-window all-ones mask over diffs of length `horizon`.
    eq1 = (diffs == 1).astype(np.int64)
    # cumulative-sum trick: window sum == horizon iff all diffs in window are 1.
    csum = np.concatenate([[0], np.cumsum(eq1)])
    window_sum = csum[horizon:] - csum[:-horizon]  # length n-horizon
    return np.flatnonzero(window_sum == horizon).astype(np.int64)


def _fold_mse_with_target_scaler(
    X: np.ndarray,
    y: np.ndarray,
    *,
    cv_splits: int,
    seed: int,
) -> tuple[list[float], bool]:
    """Run K-fold CV, scaling X and y per fold, returning per-fold MSEs.

    Returns ``(mses, ok)`` where ``ok`` is False if any fold had a
    degenerate target.
    """
    n = X.shape[0]
    n_splits = min(cv_splits, n)
    if n_splits < 2:
        return [], False
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_mses: list[float] = []
    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        if y_tr.ndim == 1:
            if float(np.std(y_tr)) < 1e-9:
                return [], False
        else:
            if np.any(np.std(y_tr, axis=0) < 1e-9):
                return [], False
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        X_tr_s = x_scaler.fit_transform(X_tr)
        X_te_s = x_scaler.transform(X_te)
        y_tr_s = y_scaler.fit_transform(y_tr)
        y_te_s = y_scaler.transform(y_te)
        model = Ridge(alpha=1.0)
        model.fit(X_tr_s, y_tr_s)
        pred = model.predict(X_te_s)
        fold_mses.append(float(((pred - y_te_s) ** 2).mean()))
    return fold_mses, True


def compute_memory_score(
    features: TrackFeatures,
    *,
    horizon: int = 1,
    min_samples: int = 8,
    cv_splits: int = 3,
    seed: int = 0,
) -> MemoryScoreResult:
    """Compute M = error(S_t -> S_{t+k}) - error((S_t, I_t) -> S_{t+k}).

    Positive memory score means including the candidate's internal state
    improves prediction of the future sensory state, i.e. the internal state
    contains memory beyond what the current sensory snapshot carries.

    Notes:
      * For ``horizon=k``, we require frames ``i, i+1, ..., i+k`` to all be
        consecutive (no gaps), so ``S_t`` and ``S_{t+k}`` are connected by an
        unbroken chain.
      * Both models share folds (same ``KFold`` shuffle seed) and the same
        target scaling logic, so their MSEs are comparable.  X is
        standardised separately because the augmented model has more
        features.
    """
    track_id = features.track_id
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")

    idx = _contiguous_horizon_indices(features.frames, horizon)
    n = int(idx.shape[0])

    if n < min_samples:
        return MemoryScoreResult(
            track_id=track_id,
            memory_score=float("nan"),
            error_s_only=float("nan"),
            error_s_plus_i=float("nan"),
            horizon=horizon,
            n_train=n,
            valid=False,
            reason="too_short",
        )

    S_t = features.sensory_features[idx]
    I_t = features.internal_features[idx]
    S_target = features.sensory_features[idx + horizon]

    X_baseline = S_t
    X_augmented = np.concatenate([S_t, I_t], axis=1)

    base_mses, base_ok = _fold_mse_with_target_scaler(
        X_baseline, S_target, cv_splits=cv_splits, seed=seed
    )
    aug_mses, aug_ok = _fold_mse_with_target_scaler(
        X_augmented, S_target, cv_splits=cv_splits, seed=seed
    )

    if not base_ok or not aug_ok:
        return MemoryScoreResult(
            track_id=track_id,
            memory_score=float("nan"),
            error_s_only=float("nan"),
            error_s_plus_i=float("nan"),
            horizon=horizon,
            n_train=n,
            valid=False,
            reason="degenerate_target",
        )

    err_s = float(np.mean(base_mses))
    err_si = float(np.mean(aug_mses))

    return MemoryScoreResult(
        track_id=track_id,
        memory_score=float(err_s - err_si),
        error_s_only=err_s,
        error_s_plus_i=err_si,
        horizon=horizon,
        n_train=n,
        valid=True,
        reason="ok",
    )
