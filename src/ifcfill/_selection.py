"""Choose a missingness reconstruction mode from the observed data alone.

This module is optional and needs scikit-learn::

    pip install "ifcfill[select]"

The mode that best reconstructs missingness depends on how the missingness
arose, which is rarely known. :func:`select_mode` compares the modes that need
no generator (``"hybrid"``, ``"joint"``, ``"marginal"``) on the observed table
itself, by cross-validation, and returns the one that loses the least
dependence of missingness on the data, as measured by
:func:`missingness_dependence_gap`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._io import load_to_dataframe
from ._transformer import IFCTransformer
from ._types import ColType, infer_col_type

SELECTABLE_MODES = ("hybrid", "joint", "marginal")
MIN_DEPENDENCE_CASES = 5
_MISSING_TEXT = "__missing__"
_UNIX_EPOCH = pd.Timestamp("1970-01-01")


@dataclass(frozen=True)
class ModeSelection:
    """Result of :func:`select_mode`.

    Attributes
    ----------
    mode:
        The selected ``missingness_restore`` value for
        :meth:`IFCTransformer.inverse_transform`.
    gaps:
        Mean validation dependence gap of every candidate mode. Lower is
        better; ``NaN`` means no column had enough missing and observed
        values to score.
    """

    mode: str
    gaps: dict[str, float]


def select_mode(
    data: str | Path | pd.DataFrame,
    *,
    col_types: dict[str, ColType] | None = None,
    modes: Sequence[str] = SELECTABLE_MODES,
    n_folds: int = 3,
    categorical_columns: Sequence[str] | None = None,
    random_state: int | None = None,
    **transformer_params: Any,
) -> ModeSelection:
    """Select the reconstruction mode that best retains missingness dependence.

    *data* is split into *n_folds* folds. In each rotation, an
    :class:`IFCTransformer` is fitted on one fold, the next fold is completed
    with :meth:`~IFCTransformer.transform` and reconstructed with every
    candidate mode, and the :func:`missingness_dependence_gap` of each
    reconstruction is computed with the fitting fold as the reference and a
    third, unseen fold as the evaluation rows. The mode with the smallest gap
    averaged over the rotations is selected. No synthetic data are needed.

    ``"generated"`` cannot be selected this way, because its missingness comes
    from the generator and is known only after synthesis.

    Parameters
    ----------
    data:
        The observed, incomplete table, as a DataFrame or a CSV file path.
    col_types:
        Optional per-column type overrides, as for :class:`IFCTransformer`.
        Types of the remaining columns are inferred once on the full *data*,
        so every fold uses the same types.
    modes:
        Candidate modes, a subset of ``("hybrid", "joint", "marginal")``.
    n_folds:
        Number of folds; at least 3, because each rotation uses one fold
        to fit, one to reconstruct, and one to evaluate.
    categorical_columns:
        Columns the gap's classifiers treat as categorical (one-hot encoded).
        Defaults to the columns typed ``"categorical"``. A binary outcome
        declared ``"integer"`` can be listed here to treat it as a category.
    random_state:
        Seed for the fold split, the mask assignment, and the classifiers.
    **transformer_params:
        Further keyword arguments for :class:`IFCTransformer`, such as
        ``cat_fill`` or ``int_fill``.

    Returns
    -------
    ModeSelection
        The selected mode and the mean validation gap of every candidate.

    Raises
    ------
    ImportError
        If scikit-learn is not installed.
    ValueError
        If a candidate mode is not selectable, *n_folds* is below 3, or every
        candidate's gap is ``NaN``.

    Examples
    --------
    >>> from ifcfill import IFCTransformer, select_mode
    >>> choice = select_mode(real, random_state=0)  # doctest: +SKIP
    >>> tf = IFCTransformer().fit(real)  # doctest: +SKIP
    >>> released = tf.inverse_transform(
    ...     synthetic, missingness_restore=choice.mode, random_state=0
    ... )  # doctest: +SKIP
    """
    _require_sklearn()
    modes = tuple(modes)
    unknown = [mode for mode in modes if mode not in SELECTABLE_MODES]
    if unknown or not modes:
        raise ValueError(
            f"Cannot select among {modes!r}. Candidate modes must be a non-empty "
            f"subset of {SELECTABLE_MODES}; 'generated' needs synthetic data."
        )
    if n_folds < 3:
        raise ValueError(f"n_folds must be at least 3, got {n_folds}.")

    observed = load_to_dataframe(data)
    columns = list(observed.columns)
    resolved_types: dict[str, ColType] = {
        column: (col_types or {}).get(column) or infer_col_type(observed[column])
        for column in columns
    }
    if categorical_columns is None:
        categorical_columns = [c for c, kind in resolved_types.items() if kind == "categorical"]
    datetime_columns = [c for c, kind in resolved_types.items() if kind == "datetime"]

    order = np.random.default_rng(random_state).permutation(len(observed))
    folds = [observed.iloc[part].reset_index(drop=True) for part in np.array_split(order, n_folds)]
    gaps: dict[str, list[float]] = {mode: [] for mode in modes}
    for rotation in range(n_folds):
        fit, reconstruct, evaluate = (folds[(rotation + k) % n_folds] for k in range(3))
        transformer = IFCTransformer(col_types=resolved_types, **transformer_params).fit(fit)
        completed = transformer.transform(reconstruct)
        for mode in modes:
            restored = transformer.inverse_transform(
                completed.copy(), missingness_restore=mode, random_state=random_state
            )
            gaps[mode].append(
                missingness_dependence_gap(
                    fit,
                    restored[columns],
                    evaluate,
                    categorical_columns=categorical_columns,
                    datetime_columns=datetime_columns,
                    random_state=random_state,
                )
            )

    mean_gaps = {mode: _nanmean(values) for mode, values in gaps.items()}
    scored = {mode: gap for mode, gap in mean_gaps.items() if not np.isnan(gap)}
    if not scored:
        raise ValueError(
            "No column has enough missing and observed values in every fold "
            f"(at least {MIN_DEPENDENCE_CASES} of each) to compare the modes."
        )
    return ModeSelection(mode=min(scored, key=scored.get), gaps=mean_gaps)


def missingness_dependence_gap(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    categorical_columns: Sequence[str] = (),
    datetime_columns: Sequence[str] = (),
    random_state: int | None = None,
) -> float:
    """Return how much missingness--value dependence is lost in *comparison*.

    For every column ``j`` that is partly missing in *reference*, one
    gradient-boosted tree classifier predicting the indicator ``M_j`` from the
    remaining columns is fitted on *reference* and another on *comparison*.
    Both are scored on *evaluation*, unseen rows from the reference's
    missingness process, and the result is the mean of
    ``AUC_reference - AUC_comparison`` over ``j``. Zero means the comparison
    teaches the same dependence of missingness on the other variables;
    positive values mean dependence is lost. A comparison with no variation in
    ``M_j`` (for example a fully imputed table) scores an AUC of 0.5.

    Parameters
    ----------
    reference:
        Real data whose missingness dependence is the target.
    comparison:
        Data to assess, for example a reconstructed or synthetic table.
    evaluation:
        Real rows unseen by both models, from the same missingness process.
    categorical_columns:
        Columns one-hot encoded with an explicit missing level; all others
        are treated as numeric and median-filled with missingness indicators.
    datetime_columns:
        Columns converted to days since the Unix epoch.
    random_state:
        Seed for the classifiers.

    Returns
    -------
    float
        Mean gap over the scored columns. Columns with fewer than five
        missing or five observed values in *reference* or *evaluation* are
        skipped, and ``NaN`` is returned when none remain.
    """
    _require_sklearn()
    from sklearn.metrics import roc_auc_score

    if not set(reference.columns) == set(comparison.columns) == set(evaluation.columns):
        raise ValueError("All DataFrames must contain the same column names")
    columns = list(reference.columns)
    reference, comparison, evaluation = (
        _with_datetime_days(frame[columns], datetime_columns)
        for frame in (reference, comparison, evaluation)
    )

    gaps = []
    for column in columns:
        reference_missing = reference[column].isna().to_numpy(dtype=int)
        evaluation_missing = evaluation[column].isna().to_numpy(dtype=int)
        if min(_minority_count(reference_missing), _minority_count(evaluation_missing)) < (
            MIN_DEPENDENCE_CASES
        ):
            continue

        features = [other for other in columns if other != column]
        aucs = []
        for training, training_missing in (
            (reference, reference_missing),
            (comparison, comparison[column].isna().to_numpy(dtype=int)),
        ):
            if training_missing.min() == training_missing.max():
                aucs.append(0.5)
                continue
            model = _missingness_model(features, categorical_columns, random_state)
            model.fit(training[features], training_missing)
            scores = model.predict_proba(evaluation[features])[:, 1]
            aucs.append(float(roc_auc_score(evaluation_missing, scores)))
        gaps.append(aucs[0] - aucs[1])

    return float(np.mean(gaps)) if gaps else float("nan")


def _require_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "Mode selection needs scikit-learn. Install it with "
            "'pip install \"ifcfill[select]\"'."
        ) from error


def _nanmean(values: list[float]) -> float:
    finite = [value for value in values if not np.isnan(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _minority_count(indicator: np.ndarray) -> int:
    return int(min(indicator.sum(), len(indicator) - indicator.sum()))


def _with_datetime_days(frame: pd.DataFrame, datetime_columns: Sequence[str]) -> pd.DataFrame:
    converted = frame.copy()
    for column in datetime_columns:
        converted[column] = _datetime_days(converted[column])
    return converted


def _datetime_days(series: pd.Series) -> pd.Series:
    """Convert dates, or numeric day offsets, to days since the Unix epoch."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == series.notna().sum():
        return numeric.astype(float)
    converted = pd.to_datetime(series, errors="coerce")
    return ((converted - _UNIX_EPOCH) / pd.Timedelta(days=1)).astype(float)


def _missingness_model(
    features: list[str],
    categorical_columns: Sequence[str],
    random_state: int | None,
) -> Any:
    """Tree model whose inputs keep other variables' missingness visible.

    A nonlinear learner is used so that the gap does not share the model
    family of IFCFill-Hybrid, whose missingness placement is logistic.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

    categorical = [column for column in features if column in categorical_columns]
    numeric = [column for column in features if column not in categorical_columns]
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                make_pipeline(
                    FunctionTransformer(_as_float),
                    SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                ),
                numeric,
            ),
            (
                "categorical",
                make_pipeline(
                    FunctionTransformer(_as_missing_aware_text),
                    OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=False),
                ),
                categorical,
            ),
        ]
    )
    return make_pipeline(preprocessor, HistGradientBoostingClassifier(random_state=random_state))


def _as_float(frame: pd.DataFrame) -> np.ndarray:
    return frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, na_value=np.nan)


def _as_missing_aware_text(frame: pd.DataFrame) -> np.ndarray:
    return frame.astype(object).where(frame.notna(), _MISSING_TEXT).astype(str).to_numpy()
