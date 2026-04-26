"""Time observer-likeness score.

Functional question: does the candidate's internal state predict *future*
sensory inputs better than *past* sensory inputs?

We frame this as two ridge regressions sharing the same predictor structure:

    forward:  X_t = [I_t, S_t]  ->  S_{t+1}
    backward: X_t = [I_t, S_t]  ->  S_{t-1}

If the candidate is "watching" the world in any temporal sense, the forward
model should out-perform the backward model.  The score

    T = backward_error - forward_error

is positive when the future is easier to predict than the past.  We use
K-fold CV with per-fold standardisation so MSE values are roughly comparable
across tracks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from observer_worlds.metrics.features import TrackFeatures


@dataclass
class TimeScoreResult:
    track_id: int
    time_score: float           # backward_error - forward_error (raw, can be negative)
    forward_error: float        # MSE on (I_t, S_t) -> S_{t+1}
    backward_error: float       # MSE on (I_t, S_t) -> S_{t-1}
    n_train: int                # number of (t, t+1) pairs used
    n_features: int             # dim of [I, S] concatenated
    valid: bool                 # False if n_train < min_samples
    reason: str                 # "ok" | "too_short" | etc


def _fold_mse(
    X: np.ndarray,
    y: np.ndarray,
    *,
    cv_splits: int,
    seed: int,
) -> float | None:
    """Mean-MSE across K folds with per-fold standardisation + Ridge.

    Returns ``None`` if any fold has a degenerate target (zero std), which
    makes the metric uninformative for this track.
    """
    n = X.shape[0]
    n_splits = min(cv_splits, n)
    if n_splits < 2:
        return None
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_mses: list[float] = []
    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        # Per-fold standardisation: fit on train, transform on test.
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        # Reject degenerate targets where standardisation is uninformative.
        if y_tr.ndim == 1:
            if float(np.std(y_tr)) < 1e-9:
                return None
        else:
            if np.any(np.std(y_tr, axis=0) < 1e-9):
                return None
        X_tr_s = x_scaler.fit_transform(X_tr)
        X_te_s = x_scaler.transform(X_te)
        y_tr_s = y_scaler.fit_transform(y_tr)
        y_te_s = y_scaler.transform(y_te)
        model = Ridge(alpha=1.0)
        model.fit(X_tr_s, y_tr_s)
        pred = model.predict(X_te_s)
        fold_mses.append(float(((pred - y_te_s) ** 2).mean()))
    return float(np.mean(fold_mses))


def compute_time_score(
    features: TrackFeatures,
    *,
    min_samples: int = 8,
    cv_splits: int = 3,
    seed: int = 0,
) -> TimeScoreResult:
    """Compute T = backward_error - forward_error using ridge + K-fold CV.

    Higher score means future sensory input is easier to predict from the
    candidate's current internal+sensory state than past sensory input is.

    Implementation:
      1. Use ``features.contiguous_triples()`` to get aligned indices ``i``
         such that frames ``i, i+1, i+2`` are all consecutive.  At each such
         triple, the "anchor" timestep is ``t = i+1``; we predict
         ``S_{t+1} = S_{i+2}`` (forward) and ``S_{t-1} = S_{i}`` (backward)
         from ``X_t = concat(I_t, S_t)`` at index ``i+1``.
      2. If ``len(X) < min_samples``: return invalid result.
      3. Per-fold standardisation of X and y (fit on train, transform test),
         then ``Ridge(alpha=1.0)``; sklearn's Ridge handles multi-output Y
         natively.
      4. MSE on standardised targets so magnitudes are comparable across
         tracks.
    """
    track_id = features.track_id
    triples = features.contiguous_triples()
    n_triples = int(triples.shape[0])

    n_features_dim = (
        features.internal_features.shape[1]
        + features.sensory_features.shape[1]
    )

    if n_triples < min_samples:
        return TimeScoreResult(
            track_id=track_id,
            time_score=float("nan"),
            forward_error=float("nan"),
            backward_error=float("nan"),
            n_train=n_triples,
            n_features=n_features_dim,
            valid=False,
            reason="too_short",
        )

    anchor = triples + 1  # actual t indices

    I_t = features.internal_features[anchor]
    S_t = features.sensory_features[anchor]
    S_next = features.sensory_features[triples + 2]
    S_prev = features.sensory_features[triples]

    X = np.concatenate([I_t, S_t], axis=1)

    fwd = _fold_mse(X, S_next, cv_splits=cv_splits, seed=seed)
    bwd = _fold_mse(X, S_prev, cv_splits=cv_splits, seed=seed)

    if fwd is None or bwd is None:
        return TimeScoreResult(
            track_id=track_id,
            time_score=float("nan"),
            forward_error=float("nan") if fwd is None else float(fwd),
            backward_error=float("nan") if bwd is None else float(bwd),
            n_train=n_triples,
            n_features=n_features_dim,
            valid=False,
            reason="degenerate_target",
        )

    return TimeScoreResult(
        track_id=track_id,
        time_score=float(bwd - fwd),
        forward_error=float(fwd),
        backward_error=float(bwd),
        n_train=n_triples,
        n_features=n_features_dim,
        valid=True,
        reason="ok",
    )
