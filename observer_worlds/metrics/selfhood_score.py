"""Selfhood (Markov-blanket-style) observer-likeness score.

Functional question: does the candidate's *boundary* mediate environment-
internal interaction?  In a Markov-blanket framing, a self-like system has
its internal state predicted well by the boundary, and adding the
environment as a predictor on top of the boundary should not buy you much.

We compute three ridge regressions (with K-fold CV and per-fold
standardisation), reporting predictive performance as :math:`R^2` clipped to
:math:`[0, 1]` (negative :math:`R^2` -- worse than the mean baseline -- is
treated as zero predictive power):

    boundary_predictability   = R^2(B_t -> I_t)
    direct_env_predictability = R^2(E_t -> I_t)
    extra_env_given_boundary  = max(0, R^2((B,E) -> I) - R^2(B -> I))

The selfhood score itself is

    selfhood = boundary_predictability - extra_env_given_boundary

i.e. high when the boundary already explains the internal state and the
environment adds little once the boundary is known.

Two complementary scalar summaries accompany the score:

    persistence: mean cosine similarity between consecutive ``I_t`` vectors,
                 clipped to ``[0, 1]`` (1 == internal state stable).
    boundedness: ``1 / (1 + std(area)/(mean(area)+eps))`` -- mirrors the M1
                 persistence-score formulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from observer_worlds.metrics.features import TrackFeatures


@dataclass
class SelfhoodScoreResult:
    track_id: int
    selfhood_score: float       # boundary_predictability - extra_env_given_boundary
    boundary_predictability: float  # R^2 of B_t -> I_t (clipped to [0,1])
    direct_env_predictability: float  # R^2 of E_t -> I_t (clipped to [0,1])
    extra_env_given_boundary: float   # R^2((B,E) -> I) - R^2(B -> I), clipped to >= 0
    persistence: float          # 1 - mean cosine distance between consecutive I_t (in [0,1])
    boundedness: float          # 1 / (1 + std(area)/(mean(area)+1e-9))
    n_train: int
    valid: bool
    reason: str


def _predictive_r2(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    cv_splits: int,
    seed: int,
) -> float | None:
    """Mean cross-validated :math:`R^2` for predicting standardised ``Y`` from ``X``.

    Returns ``None`` if the target is degenerate on any fold (zero std on the
    training portion); otherwise returns the across-fold mean of
    ``r2_score`` on standardised test targets.  Negative values are *not*
    clipped here -- the caller does that.
    """
    n = X.shape[0]
    n_splits = min(cv_splits, n)
    if n_splits < 2:
        return None
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_r2s: list[float] = []
    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        Y_tr, Y_te = Y[train_idx], Y[test_idx]

        # Reject degenerate training targets -- standardisation would blow up
        # and R^2 is uninformative.
        if Y_tr.ndim == 1:
            if float(np.std(Y_tr)) < 1e-9:
                return None
        else:
            if np.any(np.std(Y_tr, axis=0) < 1e-9):
                return None

        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        X_tr_s = x_scaler.fit_transform(X_tr)
        X_te_s = x_scaler.transform(X_te)
        Y_tr_s = y_scaler.fit_transform(Y_tr)
        Y_te_s = y_scaler.transform(Y_te)

        model = Ridge(alpha=1.0)
        model.fit(X_tr_s, Y_tr_s)
        pred = model.predict(X_te_s)
        # multioutput='uniform_average' is the default; spelled out for clarity.
        fold_r2s.append(
            float(r2_score(Y_te_s, pred, multioutput="uniform_average"))
        )
    return float(np.mean(fold_r2s))


def _persistence(internal: np.ndarray) -> float:
    """Mean cosine similarity between consecutive rows, clipped to ``[0, 1]``.

    We measure stability across the *observed* sequence; gaps in the track
    are tolerated -- the spec asks for "stability" rather than a strictly
    contiguous prediction, so any consecutive pair of observed rows counts.
    """
    if internal.shape[0] < 2:
        return 1.0
    a = internal[:-1]
    b = internal[1:]
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    denom = norm_a * norm_b
    # Where either norm is zero, cosine similarity is undefined; treat as 1.0
    # (two zero vectors are "the same").
    valid = denom > 1e-12
    if not np.any(valid):
        return 1.0
    sims = np.ones(a.shape[0], dtype=np.float64)
    sims[valid] = np.sum(a[valid] * b[valid], axis=1) / denom[valid]
    mean_sim = float(np.mean(sims))
    if mean_sim < 0.0:
        return 0.0
    if mean_sim > 1.0:
        return 1.0
    return mean_sim


def _boundedness(area: np.ndarray) -> float:
    if area.size == 0:
        return 0.0
    mean_area = float(np.mean(area))
    std_area = float(np.std(area))
    return 1.0 / (1.0 + std_area / (mean_area + 1e-9))


def compute_selfhood_score(
    features: TrackFeatures,
    *,
    min_samples: int = 8,
    cv_splits: int = 3,
    seed: int = 0,
) -> SelfhoodScoreResult:
    """Compute the Markov-blanket-style selfhood components.

    See module docstring for the full functional question.  This metric is
    per-row -- there is no temporal pairing -- so we use every observed row
    rather than filtering for contiguous pairs.
    """
    track_id = features.track_id
    n = features.n_obs

    persistence = _persistence(features.internal_features)
    boundedness = _boundedness(features.area)

    if n < min_samples:
        return SelfhoodScoreResult(
            track_id=track_id,
            selfhood_score=float("nan"),
            boundary_predictability=float("nan"),
            direct_env_predictability=float("nan"),
            extra_env_given_boundary=float("nan"),
            persistence=persistence,
            boundedness=boundedness,
            n_train=n,
            valid=False,
            reason="too_short",
        )

    X_B = features.boundary_features
    X_E = features.sensory_features
    X_BE = np.concatenate([X_B, X_E], axis=1)
    Y = features.internal_features

    r2_b = _predictive_r2(X_B, Y, cv_splits=cv_splits, seed=seed)
    r2_e = _predictive_r2(X_E, Y, cv_splits=cv_splits, seed=seed)
    r2_be = _predictive_r2(X_BE, Y, cv_splits=cv_splits, seed=seed)

    if r2_b is None or r2_e is None or r2_be is None:
        return SelfhoodScoreResult(
            track_id=track_id,
            selfhood_score=float("nan"),
            boundary_predictability=float("nan") if r2_b is None else float(max(0.0, min(1.0, r2_b))),
            direct_env_predictability=float("nan") if r2_e is None else float(max(0.0, min(1.0, r2_e))),
            extra_env_given_boundary=float("nan"),
            persistence=persistence,
            boundedness=boundedness,
            n_train=n,
            valid=False,
            reason="degenerate_target",
        )

    # Clip raw R^2 to [0, 1].  Negative R^2 means the model is worse than
    # predicting the mean -- treat that as no predictive power.  >1 cannot
    # happen in theory but clip defensively.
    bp = float(max(0.0, min(1.0, r2_b)))
    de = float(max(0.0, min(1.0, r2_e)))
    be = float(max(0.0, min(1.0, r2_be)))

    extra = max(0.0, be - bp)
    selfhood = bp - extra

    return SelfhoodScoreResult(
        track_id=track_id,
        selfhood_score=float(selfhood),
        boundary_predictability=bp,
        direct_env_predictability=de,
        extra_env_given_boundary=float(extra),
        persistence=persistence,
        boundedness=boundedness,
        n_train=n,
        valid=True,
        reason="ok",
    )
