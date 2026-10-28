from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Maximum one-hot levels per categorical feature; rarer levels encode as all zeros.
HYBRID_MAX_LEVELS = 20
# Fitting rows are subsampled above this size to bound IRLS cost.
HYBRID_MAX_FIT_ROWS = 50_000
_L2_PENALTY = 1.0
_MAX_ITER = 50
_TOL = 1e-6


def build_feature_spec(
    values: dict[str, pd.Series],
    column_types: dict[str, str],
    indicators: dict[str, np.ndarray],
    sentinels: dict[str, str | None],
) -> list[dict[str, Any]]:
    """Describe the design-matrix columns used by the hybrid missingness models.

    *values* holds completed (filled) per-column values; numeric and datetime
    columns become standardized features, categorical columns become one-hot
    features over their most frequent non-sentinel levels. *indicators* holds
    the real missingness indicators of columns whose missingness is also
    observable in generated data (flags or sentinels).
    """
    spec: list[dict[str, Any]] = []
    for column, series in values.items():
        if column_types[column] == "categorical":
            observed = series.astype(str)
            sentinel = sentinels.get(column)
            if sentinel is not None:
                observed = observed[observed != sentinel]
            for level in observed.value_counts().index[:HYBRID_MAX_LEVELS]:
                spec.append({"kind": "category", "column": column, "level": str(level)})
            continue

        arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        std = float(np.nanstd(arr)) if arr.size else 0.0
        if not np.isfinite(std) or std == 0.0:
            continue
        spec.append(
            {"kind": "numeric", "column": column, "mean": float(np.nanmean(arr)), "std": std}
        )

    for column in indicators:
        spec.append({"kind": "missing", "column": column})
    return spec


def design_matrix(
    spec: list[dict[str, Any]],
    values: dict[str, pd.Series],
    indicators: dict[str, np.ndarray],
    n_rows: int,
) -> np.ndarray:
    """Evaluate *spec* on completed values and missingness indicators."""
    matrix = np.zeros((n_rows, len(spec)), dtype=float)
    as_text: dict[str, np.ndarray] = {}
    for index, feature in enumerate(spec):
        column = feature["column"]
        if feature["kind"] == "missing":
            matrix[:, index] = indicators[column].astype(float)
        elif feature["kind"] == "numeric":
            arr = pd.to_numeric(values[column], errors="coerce").to_numpy(dtype=float)
            arr = np.where(np.isnan(arr), feature["mean"], arr)
            matrix[:, index] = (arr - feature["mean"]) / feature["std"]
        else:
            if column not in as_text:
                as_text[column] = values[column].astype(str).to_numpy()
            matrix[:, index] = as_text[column] == feature["level"]
    return matrix


def fit_logistic(x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Fit an L2-penalized logistic regression by iteratively reweighted least squares.

    Returns ``(intercept, coefficients)``. The intercept is not penalized.
    """
    n_rows, n_features = x.shape
    xb = np.hstack([np.ones((n_rows, 1)), x])
    rate = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    weights = np.zeros(n_features + 1)
    weights[0] = np.log(rate / (1 - rate))
    penalty = np.full(n_features + 1, _L2_PENALTY)
    penalty[0] = 0.0

    for _ in range(_MAX_ITER):
        p = _sigmoid(xb @ weights)
        gradient = xb.T @ (p - y) + penalty * weights
        hessian = xb.T @ (xb * (p * (1 - p))[:, None]) + np.diag(penalty)
        step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        weights -= step
        if np.max(np.abs(step)) < _TOL:
            break
    return float(weights[0]), weights[1:]


def predict_proba(x: np.ndarray, intercept: float, coefficients: np.ndarray) -> np.ndarray:
    return _sigmoid(intercept + x @ coefficients)


def weighted_sample_without_replacement(
    probabilities: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw *k* row indices without replacement, proportional to *probabilities*.

    Uses the Gumbel-top-k trick, which is equivalent to successive weighted
    sampling without replacement.
    """
    log_weights = np.log(np.clip(probabilities, 1e-12, None))
    keys = log_weights + rng.gumbel(size=probabilities.shape[0])
    return np.argpartition(-keys, k - 1)[:k] if k else np.empty(0, dtype=int)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))
