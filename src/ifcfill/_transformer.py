from __future__ import annotations

import json
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from ._encoding import LabelCategoryEncoder
from ._impute import compute_fill_categorical, compute_fill_float, compute_fill_integer
from ._io import load_to_dataframe
from ._types import ColType, infer_col_type

# String sentinels that represent null in object columns after .astype(str)
_NULL_SENTINELS = frozenset({"nan", "none", "<na>", "nat", "pd.na", ""})
_CAT_ENCODINGS = ("none", "label")
_MISSINGNESS_RESTORE_MODES = ("marginal", "joint")
DEFAULT_CAT_CONSTANT = "__ifcfill_missing__"
_STATE_VERSION = 2

# Multiplier to convert total_seconds() to each supported datetime unit
_SECONDS_PER_UNIT: dict[str, float] = {
    "D": 86400.0,
    "s": 1.0,
    "ms": 1e-3,
    "us": 1e-6,
    "ns": 1e-9,
}


def _datetime_to_numeric(
    series: pd.Series,
    anchor: pd.Timestamp,
    unit: str,
) -> np.ndarray:
    """Convert a datetime Series to a float64 NumPy array relative to *anchor*.

    NaT values become ``NaN``.
    """
    parsed = pd.to_datetime(series, errors="coerce")
    delta_seconds = (parsed - anchor).dt.total_seconds().to_numpy(dtype=float)
    return delta_seconds / _SECONDS_PER_UNIT[unit]


def _numeric_to_datetime(
    series: pd.Series,
    anchor: pd.Timestamp,
    unit: str,
) -> pd.Series:
    """Convert numeric datetime offsets back to timestamps."""
    numeric = pd.to_numeric(series, errors="coerce")
    return anchor + pd.to_timedelta(numeric, unit=unit)


class IFCTransformer:
    """Transform tabular data into Integer, Float and Categorical (IFC) columns,
    fill missing values, convert datetimes to integers, and drop constant columns.

    Accepts a CSV file path or a :class:`pandas.DataFrame` as input.
    All heavy computations are performed with NumPy for fast processing.

    Parameters
    ----------
    col_types:
        Optional per-column type overrides.  Keys are column names; values are
        ``"integer"``, ``"float"``, ``"categorical"``, or ``"datetime"``.
        Columns not listed are inferred automatically.
    int_fill:
        Strategy for filling missing values in integer columns.
        One of ``"mean"`` (→ ``int(round(mean))``), ``"median"``, ``"mode"``,
        ``"zero"``.
    float_fill:
        Strategy for filling missing values in float columns.
        One of ``"mean"``, ``"median"``, ``"mode"``, ``"zero"``.
    cat_fill:
        Strategy for filling missing values in categorical columns.
        ``"constant"`` uses *cat_constant* so categorical missingness can be
        learned by a synthetic-data generator as its own category. ``"mode"``
        uses the most frequent value.
    cat_constant:
        Fill string used when *cat_fill* is ``"constant"``.
        Defaults to ``"__ifcfill_missing__"`` to reduce collisions with real
        user categories.
    cat_encoding:
        Optional encoding for categorical columns. ``"none"`` keeps
        categorical columns as pandas categoricals. ``"label"`` fills
        categorical values first, then maps each completed category to an
        integer code through a separate encoder layer and stores mappings for
        :meth:`inverse_transform`.
    n_jobs:
        Number of worker threads to use for per-column ``fit`` and
        ``transform`` work. ``None`` or ``1`` runs sequentially. Negative values
        follow the joblib convention, so ``-1`` uses all available CPUs.
    datetime_anchor:
        Reference date for datetime-to-integer conversion.
        Defaults to the Unix epoch ``"1970-01-01"``.
    datetime_unit:
        Unit for the integer representation of datetimes.
        One of ``"D"`` (days), ``"s"`` (seconds), ``"ms"``, ``"us"``, ``"ns"``.

    Attributes (set after ``fit``)
    --------------------------------
    column_types_ : dict[str, str]
        Detected or user-specified type for every non-constant column.
    fill_values_ : dict[str, Any]
        Computed fill value for every non-constant column.
    dropped_constants_ : dict[str, tuple[Any, int]]
        ``{column_name: (constant_value, original_position_index)}`` for every
        column detected as constant and dropped.
    original_columns_ : list[str]
        Column names in their original order (including constant columns).
    missing_counts_ : dict[str, int]
        Number of missing values per column (all original columns).
    missing_fractions_ : dict[str, float]
        Fraction of missing values per column (all original columns).
    missing_pattern_distribution_ : dict[tuple[int, ...], float]
        Empirical joint distribution of row-wise missingness patterns. Pattern
        positions follow ``original_columns_``.
    categorical_distributions_ : dict[str, dict[str, float]]
        Observed real-data distributions used when categorical sentinel values
        cannot be replaced from synthetic values.
    category_mappings_ : dict[str, dict[str, int]]
        Forward mapping for label-encoded categorical columns.
    inverse_category_mappings_ : dict[str, dict[int, str]]
        Inverse mapping for label-encoded categorical columns.

    Examples
    --------
    >>> tf = IFCTransformer()
    >>> transformed = tf.fit_transform("data.csv")
    >>> restored = tf.inverse_transform(
    ...     transformed, missingness_restore="joint", random_state=42
    ... )
    """

    def __init__(
        self,
        col_types: dict[str, ColType] | None = None,
        int_fill: Literal["mean", "median", "mode", "zero"] = "median",
        float_fill: Literal["mean", "median", "mode", "zero"] = "mean",
        cat_fill: Literal["mode", "constant"] = "constant",
        cat_constant: str = DEFAULT_CAT_CONSTANT,
        cat_encoding: Literal["none", "label"] = "none",
        n_jobs: int | None = 1,
        datetime_anchor: str | pd.Timestamp = "1970-01-01",
        datetime_unit: Literal["D", "s", "ms", "us", "ns"] = "D",
    ) -> None:
        if datetime_unit not in _SECONDS_PER_UNIT:
            raise ValueError(
                f"Unknown datetime_unit {datetime_unit!r}. "
                f"Choose from: {tuple(_SECONDS_PER_UNIT)}."
            )
        if cat_encoding not in _CAT_ENCODINGS:
            raise ValueError(
                f"Unknown cat_encoding {cat_encoding!r}. "
                f"Choose from: {_CAT_ENCODINGS}."
            )
        self.col_types: dict[str, ColType] = col_types or {}
        self.int_fill = int_fill
        self.float_fill = float_fill
        self.cat_fill = cat_fill
        self.cat_constant = cat_constant
        self.cat_encoding = cat_encoding
        self.n_jobs = n_jobs
        self._effective_n_jobs()
        self.datetime_anchor = pd.Timestamp(datetime_anchor)
        self.datetime_unit = datetime_unit

        # populated by fit()
        self.column_types_: dict[str, ColType] = {}
        self.fill_values_: dict[str, Any] = {}
        self.dropped_constants_: dict[str, tuple[Any, int]] = {}
        self.original_columns_: list[str] = []
        self.original_column_types_: dict[str, ColType] = {}
        self.missing_counts_: dict[str, int] = {}
        self.missing_fractions_: dict[str, float] = {}
        self.missing_pattern_distribution_: dict[tuple[int, ...], float] = {}
        self.categorical_distributions_: dict[str, dict[str, float]] = {}
        self._category_encoder = LabelCategoryEncoder()
        self.category_mappings_ = self._category_encoder.category_mappings_
        self.inverse_category_mappings_ = self._category_encoder.inverse_category_mappings_
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def missing_report_(self) -> pd.DataFrame:
        """DataFrame summarising the missing-value distribution at ``fit`` time.

        Columns: ``column``, ``type``, ``missing_count``, ``missing_fraction``.
        Constant columns are listed with type ``"constant"``.
        """
        self._check_fitted()
        rows = [
            {
                "column": col,
                "type": self.column_types_.get(col, "constant"),
                "missing_count": self.missing_counts_.get(col, 0),
                "missing_fraction": round(self.missing_fractions_.get(col, 0.0), 6),
            }
            for col in self.original_columns_
        ]
        return pd.DataFrame(rows)

    def get_category_mappings(self, inverse: bool = False) -> dict[str, dict[Any, Any]]:
        """Return a copy of the learned categorical label mappings.

        Parameters
        ----------
        inverse:
            If ``False`` (default), return ``{column: {category: code}}``.
            If ``True``, return ``{column: {code: category}}``.

        Returns
        -------
        dict[str, dict[Any, Any]]
            A defensive copy of the requested mapping dictionary.
        """
        self._check_fitted()
        return self._category_encoder.get_mappings(inverse=inverse)

    def get_category_mapping(
        self,
        column: str,
        inverse: bool = False,
    ) -> dict[Any, Any]:
        """Return a copy of the learned label mapping for one categorical column."""
        self._check_fitted()
        return self._category_encoder.get_mapping(column, inverse=inverse)

    def save(self, path: str | Path) -> None:
        """Save the fitted transformation state to a JSON file.

        The saved state can be loaded on another machine with
        :meth:`load` and used for :meth:`transform` or
        :meth:`inverse_transform` without fitting again.
        """
        self._check_fitted()
        state = self._to_state()
        output_path = Path(path)
        output_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> IFCTransformer:
        """Load a fitted transformer state saved by :meth:`save`."""
        input_path = Path(path)
        state = json.loads(input_path.read_text(encoding="utf-8"))
        if state.get("state_version") != _STATE_VERSION:
            raise ValueError(
                f"Unsupported IFCTransformer state version "
                f"{state.get('state_version')!r}; expected {_STATE_VERSION}. "
                "Refit the transformer and save a new state file."
            )
        return cls._from_state(state)

    def fit(self, data: str | Path | pd.DataFrame) -> IFCTransformer:
        """Learn column types, fill values, and constant columns from *data*.

        Parameters
        ----------
        data:
            A CSV file path or a :class:`pandas.DataFrame`.

        Returns
        -------
        self
        """
        df = load_to_dataframe(data)
        self.original_columns_ = list(df.columns)

        self.dropped_constants_ = {}
        self.column_types_ = {}
        self.original_column_types_ = {}
        self.fill_values_ = {}
        self.missing_counts_ = {}
        self.missing_fractions_ = {}
        self.categorical_distributions_ = {}
        self._category_encoder.reset()
        self.category_mappings_ = self._category_encoder.category_mappings_
        self.inverse_category_mappings_ = self._category_encoder.inverse_category_mappings_

        n = len(df)
        self.missing_pattern_distribution_ = self._fit_missing_pattern_distribution(df)

        tasks = [(idx, col, df[col], n) for idx, col in enumerate(df.columns)]
        for result in self._map_columns(self._fit_column, tasks):
            col = result["column"]
            self.original_column_types_[col] = result["column_type"]
            self.missing_counts_[col] = result["missing_count"]
            self.missing_fractions_[col] = result["missing_fraction"]

            category_distribution = result.get("category_distribution")
            if category_distribution is not None:
                self.categorical_distributions_[col] = category_distribution

            if result["is_constant"]:
                self.dropped_constants_[col] = (
                    result["constant_value"],
                    result["position"],
                )
                continue

            col_type = result["column_type"]
            self.column_types_[col] = col_type
            self.fill_values_[col] = result["fill_value"]

            category_mapping = result.get("category_mapping")
            if category_mapping is not None:
                self._category_encoder.category_mappings_[col] = category_mapping
                self._category_encoder.inverse_category_mappings_[col] = {
                    code: value for value, code in category_mapping.items()
                }

        self._is_fitted = True
        return self

    def transform(self, data: str | Path | pd.DataFrame) -> pd.DataFrame:
        """Apply type casting, missing-value fill, datetime conversion, and
        constant-column removal to *data*.

        Parameters
        ----------
        data:
            A CSV file path or a :class:`pandas.DataFrame`.

        Returns
        -------
        pandas.DataFrame
            Transformed data without constant columns and without missing values.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        self._check_fitted()
        df = load_to_dataframe(data)
        if self._is_already_transformed(df):
            warnings.warn(
                "Input data appears to be already transformed by this "
                "IFCTransformer; returning it unchanged.",
                UserWarning,
                stacklevel=2,
            )
            return df.copy()

        result: dict[str, pd.Series | pd.Categorical] = {}

        tasks = [(col, df[col], df.index) for col in df.columns]
        for col, transformed in self._map_columns(self._transform_column, tasks):
            if transformed is not None:
                result[col] = transformed

        return pd.DataFrame(result, index=df.index)

    def fit_transform(self, data: str | Path | pd.DataFrame) -> pd.DataFrame:
        """Fit and transform *data* in one step."""
        return self.fit(data).transform(data)

    def inverse_transform(
        self,
        data: str | Path | pd.DataFrame,
        missingness_restore: Literal["marginal", "joint"] = "marginal",
        random_state: int | np.random.Generator | None = None,
    ) -> pd.DataFrame:
        """Restore semantic types, structure, and missingness.

        Missingness reconstruction always occurs. ``"marginal"`` preserves
        each fitted per-column distribution :math:`P(M_j)`, while ``"joint"``
        preserves the empirical row-pattern distribution
        :math:`P(M_1, \\ldots, M_p)`. Categorical synthesis sentinels are an
        internal representation only; the reconstructed mask is authoritative.

        Integer columns are rounded to the nearest integer and returned with
        pandas nullable ``Int64`` dtype. Datetime columns use ``NaT`` for
        reconstructed missing values.

        Parameters
        ----------
        data:
            A DataFrame produced by :meth:`transform` (or a CSV of one).
        missingness_restore:
            ``"marginal"`` (default) allocates exactly
            ``round(fitted_fraction * n_rows)`` missing values independently in
            every original column. ``"joint"`` uses largest-remainder allocation
            of the fitted empirical row-wise missingness patterns.
        random_state:
            Integer seed or :class:`numpy.random.Generator` for reproducible
            mask assignment and categorical sentinel replacement.

        Returns
        -------
        pandas.DataFrame
            Reconstructed data in the fitted column order.

        Raises
        ------
        ValueError
            If *missingness_restore* is not ``"marginal"`` or ``"joint"``.
        """
        self._check_fitted()
        if missingness_restore not in _MISSINGNESS_RESTORE_MODES:
            raise ValueError(
                f"Unknown missingness_restore {missingness_restore!r}. "
                f"Choose from: {_MISSINGNESS_RESTORE_MODES}."
            )

        result = load_to_dataframe(data).copy()
        rng = np.random.default_rng(random_state)

        # 1-2. Re-add columns removed before synthesis.
        for col, (value, _) in self.dropped_constants_.items():
            result[col] = value

        # 3. Decode label-encoded categorical columns.
        if self.cat_encoding == "label":
            for col in self.inverse_category_mappings_:
                if col in result.columns:
                    result[col] = self._category_encoder.inverse_transform_column(
                        col,
                        result[col],
                    )

        # 4. Convert synthesized datetime offsets back to timestamps. Dropped
        # datetime constants were reinserted in their original representation.
        for col, col_type in self.original_column_types_.items():
            if (
                col_type == "datetime"
                and col in result.columns
                and col not in self.dropped_constants_
            ):
                result[col] = _numeric_to_datetime(
                    result[col],
                    self.datetime_anchor,
                    self.datetime_unit,
                )
                if result[col].isna().any():
                    fill_datetime = self.datetime_anchor + pd.to_timedelta(
                        self.fill_values_[col], unit=self.datetime_unit
                    )
                    result[col] = result[col].fillna(fill_datetime)

        # 5. Restore semantic numeric types and neutralize any generator-created
        # numeric nulls so only IFCFill's final mask controls missingness.
        self._restore_semantic_types(result)

        # 6. Construct one authoritative mask over every original column.
        missing_mask = self._build_missingness_mask(
            len(result), missingness_restore, rng
        )

        # 7. Replace categorical sentinels/nulls wherever the final mask says
        # the value must be observed.
        self._resolve_categorical_sentinels(result, missing_mask, rng)

        # 8. Apply type-appropriate missing values.
        self._apply_missingness_mask(result, missing_mask)

        # 9. Restore fitted column order (and discard unexpected columns).
        available = [col for col in self.original_columns_ if col in result.columns]
        result = result[available]

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fit_missing_pattern_distribution(
        df: pd.DataFrame,
    ) -> dict[tuple[int, ...], float]:
        """Return the empirical distribution of raw row-wise missingness masks."""
        if len(df) == 0:
            return {}

        mask = df.isna().to_numpy(dtype=np.uint8)
        patterns, counts = np.unique(mask, axis=0, return_counts=True)
        return {
            tuple(int(value) for value in pattern): float(count / len(df))
            for pattern, count in zip(patterns, counts)
        }

    def _build_missingness_mask(
        self,
        n_rows: int,
        mode: Literal["marginal", "joint"],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Allocate a fitted missingness distribution to *n_rows* exactly."""
        n_columns = len(self.original_columns_)
        mask = np.zeros((n_rows, n_columns), dtype=bool)
        if n_rows == 0 or n_columns == 0:
            return mask

        if mode == "marginal":
            for column_index, column in enumerate(self.original_columns_):
                fraction = self.missing_fractions_.get(column, 0.0)
                n_missing = min(max(int(round(fraction * n_rows)), 0), n_rows)
                if n_missing:
                    row_indices = rng.choice(n_rows, size=n_missing, replace=False)
                    mask[row_indices, column_index] = True
            return mask

        if not self.missing_pattern_distribution_:
            return mask

        patterns = list(self.missing_pattern_distribution_)
        if any(len(pattern) != n_columns for pattern in patterns):
            raise RuntimeError(
                "Stored joint missingness patterns do not match the fitted "
                "original column order. Refit the IFCTransformer."
            )

        probabilities = np.asarray(
            [self.missing_pattern_distribution_[pattern] for pattern in patterns],
            dtype=float,
        )
        probability_sum = probabilities.sum()
        if not np.isfinite(probability_sum) or probability_sum <= 0:
            raise RuntimeError(
                "Stored joint missingness probabilities are invalid. "
                "Refit the IFCTransformer."
            )
        probabilities = probabilities / probability_sum

        expected_counts = probabilities * n_rows
        allocated_counts = np.floor(expected_counts).astype(int)
        remaining = n_rows - int(allocated_counts.sum())
        if remaining:
            remainders = expected_counts - allocated_counts
            order = np.argsort(-remainders, kind="stable")
            allocated_counts[order[:remaining]] += 1

        pattern_array = np.asarray(patterns, dtype=bool)
        allocated = np.repeat(pattern_array, allocated_counts, axis=0)
        return allocated[rng.permutation(n_rows)]

    def _restore_semantic_types(self, result: pd.DataFrame) -> None:
        """Restore numeric dtypes and remove generator-created numeric nulls."""
        for column, column_type in self.original_column_types_.items():
            if column not in result.columns:
                continue

            if column_type == "integer":
                numeric = pd.to_numeric(result[column], errors="coerce").round()
                fill_value = self._semantic_fill_value(column)
                if pd.notna(fill_value):
                    numeric = numeric.fillna(int(round(float(fill_value))))
                result[column] = numeric.astype("Int64")
            elif column_type == "float":
                numeric = pd.to_numeric(result[column], errors="coerce")
                fill_value = self._semantic_fill_value(column)
                if pd.notna(fill_value):
                    numeric = numeric.fillna(float(fill_value))
                result[column] = numeric.astype(np.float64)
            elif column_type == "datetime":
                datetimes = pd.to_datetime(result[column], errors="coerce")
                fill_value = self._semantic_fill_value(column)
                if pd.notna(fill_value):
                    datetimes = datetimes.fillna(pd.Timestamp(fill_value))
                result[column] = datetimes
            else:
                # Object dtype allows both replacement categories and np.nan,
                # including fallback values absent from a generated Categorical.
                result[column] = result[column].astype(object)

    def _semantic_fill_value(self, column: str) -> Any:
        """Return a semantic value suitable for generated null replacement."""
        if column in self.dropped_constants_:
            return self.dropped_constants_[column][0]

        fill_value = self.fill_values_.get(column, np.nan)
        if self.original_column_types_.get(column) == "datetime" and pd.notna(fill_value):
            return self.datetime_anchor + pd.to_timedelta(
                fill_value, unit=self.datetime_unit
            )
        return fill_value

    def _resolve_categorical_sentinels(
        self,
        result: pd.DataFrame,
        missing_mask: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        """Replace generated categorical null markers outside the final mask."""
        for column_index, column in enumerate(self.original_columns_):
            if (
                self.original_column_types_.get(column) != "categorical"
                or column not in result.columns
            ):
                continue

            series = result[column].astype(object)
            sentinel: str | None = None
            if self.cat_fill == "constant" and column in self.fill_values_:
                sentinel = str(self.fill_values_[column])

            is_sentinel = np.zeros(len(series), dtype=bool)
            if sentinel is not None:
                is_sentinel = series.astype(str).eq(sentinel).to_numpy()
            is_generated_missing = series.isna().to_numpy() | is_sentinel
            requires_observed = ~missing_mask[:, column_index]
            needs_replacement = requires_observed & is_generated_missing

            if needs_replacement.any():
                valid = series[~is_generated_missing].to_numpy(dtype=object)
                n_replacements = int(needs_replacement.sum())
                if valid.size:
                    replacements = rng.choice(valid, size=n_replacements, replace=True)
                else:
                    distribution = self.categorical_distributions_.get(column, {})
                    fallback_values = np.asarray(list(distribution), dtype=object)
                    fallback_probabilities = np.asarray(
                        list(distribution.values()), dtype=float
                    )
                    if sentinel is not None and fallback_values.size:
                        usable = fallback_values.astype(str) != sentinel
                        fallback_values = fallback_values[usable]
                        fallback_probabilities = fallback_probabilities[usable]
                    if fallback_values.size == 0:
                        raise RuntimeError(
                            f"Cannot replace generated missing categorical values "
                            f"in {column!r}: no observed fitted categories are available."
                        )
                    fallback_probabilities = (
                        fallback_probabilities / fallback_probabilities.sum()
                    )
                    warnings.warn(
                        f"Synthetic column {column!r} has no usable non-sentinel "
                        "values; sampling replacements from its fitted real-data "
                        "categorical distribution.",
                        UserWarning,
                        stacklevel=2,
                    )
                    replacements = rng.choice(
                        fallback_values,
                        size=n_replacements,
                        replace=True,
                        p=fallback_probabilities,
                    )
                series.iloc[np.flatnonzero(needs_replacement)] = replacements

            result[column] = series

    def _apply_missingness_mask(
        self,
        result: pd.DataFrame,
        missing_mask: np.ndarray,
    ) -> None:
        """Apply the authoritative mask with each semantic type's null value."""
        for column_index, column in enumerate(self.original_columns_):
            if column not in result.columns:
                continue

            column_mask = missing_mask[:, column_index]
            column_type = self.original_column_types_[column]
            if column_type == "integer":
                result[column] = result[column].mask(column_mask, pd.NA).astype("Int64")
            elif column_type == "float":
                result[column] = result[column].mask(column_mask, np.nan).astype(
                    np.float64
                )
            elif column_type == "datetime":
                result[column] = result[column].mask(column_mask, pd.NaT)
            else:
                result[column] = result[column].mask(column_mask, np.nan)

    @staticmethod
    def _filled_categorical(series: pd.Series, fill_val: Any) -> pd.Series:
        s = series.astype(str)
        s = s.where(~s.str.lower().isin(_NULL_SENTINELS), other=fill_val)
        return s.fillna(fill_val)

    def _fit_column(
        self,
        task: tuple[int, str, pd.Series, int],
    ) -> dict[str, Any]:
        idx, col, series, n = task

        n_missing = int(series.isna().sum())
        missing_fraction = n_missing / n if n > 0 else 0.0

        col_type: ColType = self.col_types.get(col) or infer_col_type(series)  # type: ignore[assignment]

        unique_vals = series.dropna().unique()
        category_distribution: dict[str, float] | None = None
        if col_type == "categorical":
            observed = series[series.notna()].astype(str)
            counts = observed.value_counts(sort=False)
            category_distribution = {
                str(value): float(count / counts.sum())
                for value, count in counts.items()
            }
        keep_missing_category = (
            col_type == "categorical"
            and self.cat_fill == "constant"
            and len(unique_vals) == 1
            and n_missing > 0
        )
        if len(unique_vals) <= 1 and not keep_missing_category:
            const_val = unique_vals[0] if len(unique_vals) == 1 else np.nan
            return {
                "column": col,
                "position": idx,
                "missing_count": n_missing,
                "missing_fraction": missing_fraction,
                "is_constant": True,
                "constant_value": const_val,
                "column_type": col_type,
                "category_distribution": category_distribution,
            }

        if col_type == "datetime":
            arr = _datetime_to_numeric(series, self.datetime_anchor, self.datetime_unit)
            valid = arr[~np.isnan(arr)]
            raw = float(np.median(valid)) if valid.size > 0 else 0.0
            fill_value = int(np.round(raw))
        elif col_type == "integer":
            arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
            fill_value = compute_fill_integer(arr, self.int_fill)
        elif col_type == "float":
            arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
            fill_value = compute_fill_float(arr, self.float_fill)
        else:
            fill_value = compute_fill_categorical(
                series.to_numpy(), self.cat_fill, self.cat_constant
            )

        result: dict[str, Any] = {
            "column": col,
            "position": idx,
            "missing_count": n_missing,
            "missing_fraction": missing_fraction,
            "is_constant": False,
            "column_type": col_type,
            "fill_value": fill_value,
            "category_distribution": category_distribution,
        }

        if col_type == "categorical" and self.cat_encoding == "label":
            filled = self._filled_categorical(series, fill_value)
            categories = LabelCategoryEncoder._categories(filled, fill_value)
            result["category_mapping"] = {
                value: code for code, value in enumerate(categories)
            }

        return result

    def _transform_column(
        self,
        task: tuple[str, pd.Series, pd.Index],
    ) -> tuple[str, pd.Series | pd.Categorical | None]:
        col, series, index = task

        if col in self.dropped_constants_:
            return col, None

        if col not in self.column_types_:
            return col, series

        col_type = self.column_types_[col]
        fill_val = self.fill_values_[col]

        if col_type == "datetime":
            arr = _datetime_to_numeric(series, self.datetime_anchor, self.datetime_unit)
            arr = np.where(np.isnan(arr), fill_val, arr)
            return col, pd.Series(arr.astype(np.int64), index=index, name=col)

        if col_type == "integer":
            arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
            arr = np.where(np.isnan(arr), fill_val, arr)
            return col, pd.Series(arr.astype(np.int64), index=index, name=col)

        if col_type == "float":
            arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
            arr = np.where(np.isnan(arr), fill_val, arr)
            return col, pd.Series(arr.astype(np.float64), index=index, name=col)

        s = self._filled_categorical(series, fill_val)
        if self.cat_encoding == "label":
            return col, self._category_encoder.transform_column(
                col,
                s,
                fallback_value=fill_val,
            )
        return col, pd.Categorical(s)

    def _is_already_transformed(self, df: pd.DataFrame) -> bool:
        if not self.column_types_:
            return False
        if any(col in df.columns for col in self.dropped_constants_):
            return False
        if not all(col in df.columns for col in self.column_types_):
            return False

        for col, col_type in self.column_types_.items():
            series = df[col]
            if series.isna().any():
                return False

            if col_type in {"datetime", "integer"}:
                if not pd.api.types.is_integer_dtype(series):
                    return False
            elif col_type == "float":
                if not pd.api.types.is_float_dtype(series):
                    return False
            elif self.cat_encoding == "label":
                if not pd.api.types.is_integer_dtype(series):
                    return False
                codes = set(pd.to_numeric(series, errors="coerce").astype(int))
                valid_codes = set(self.inverse_category_mappings_.get(col, {}))
                if not codes.issubset(valid_codes):
                    return False
            elif not isinstance(series.dtype, pd.CategoricalDtype):
                return False

        return True

    def _map_columns(self, func: Any, tasks: list[Any]) -> list[Any]:
        n_jobs = self._effective_n_jobs()
        if n_jobs == 1 or len(tasks) <= 1:
            return [func(task) for task in tasks]

        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            return list(executor.map(func, tasks))

    def _effective_n_jobs(self) -> int:
        if self.n_jobs is None:
            return 1
        if isinstance(self.n_jobs, bool) or not isinstance(self.n_jobs, int):
            raise TypeError("n_jobs must be an integer, None, or omitted.")
        if self.n_jobs == 0:
            raise ValueError("n_jobs must not be 0.")
        if self.n_jobs < 0:
            return max((os.cpu_count() or 1) + 1 + self.n_jobs, 1)
        return self.n_jobs

    def _to_state(self) -> dict[str, Any]:
        return {
            "state_version": _STATE_VERSION,
            "params": {
                "col_types": self.col_types,
                "int_fill": self.int_fill,
                "float_fill": self.float_fill,
                "cat_fill": self.cat_fill,
                "cat_constant": self.cat_constant,
                "cat_encoding": self.cat_encoding,
                "n_jobs": self.n_jobs,
                "datetime_anchor": self.datetime_anchor.isoformat(),
                "datetime_unit": self.datetime_unit,
            },
            "fitted": {
                "column_types": self.column_types_,
                "fill_values": {
                    col: self._serialize_value(value)
                    for col, value in self.fill_values_.items()
                },
                "dropped_constants": {
                    col: {
                        "value": self._serialize_value(value),
                        "position": position,
                    }
                    for col, (value, position) in self.dropped_constants_.items()
                },
                "original_columns": self.original_columns_,
                "original_column_types": self.original_column_types_,
                "missing_counts": self.missing_counts_,
                "missing_fractions": self.missing_fractions_,
                "missing_pattern_distribution": [
                    {
                        "pattern": list(pattern),
                        "probability": probability,
                    }
                    for pattern, probability in self.missing_pattern_distribution_.items()
                ],
                "categorical_distributions": self.categorical_distributions_,
                "category_mappings": self.category_mappings_,
            },
        }

    @classmethod
    def _from_state(cls, state: dict[str, Any]) -> IFCTransformer:
        params = state["params"]
        transformer = cls(
            col_types=params["col_types"],
            int_fill=params["int_fill"],
            float_fill=params["float_fill"],
            cat_fill=params["cat_fill"],
            cat_constant=params["cat_constant"],
            cat_encoding=params["cat_encoding"],
            n_jobs=params.get("n_jobs", 1),
            datetime_anchor=params["datetime_anchor"],
            datetime_unit=params["datetime_unit"],
        )

        fitted = state["fitted"]
        transformer.column_types_ = dict(fitted["column_types"])
        transformer.fill_values_ = {
            col: cls._deserialize_value(value)
            for col, value in fitted["fill_values"].items()
        }
        transformer.dropped_constants_ = {
            col: (
                cls._deserialize_value(payload["value"]),
                int(payload["position"]),
            )
            for col, payload in fitted["dropped_constants"].items()
        }
        transformer.original_columns_ = list(fitted["original_columns"])
        transformer.original_column_types_ = dict(fitted["original_column_types"])
        transformer.missing_counts_ = {
            col: int(count) for col, count in fitted["missing_counts"].items()
        }
        transformer.missing_fractions_ = {
            col: float(fraction)
            for col, fraction in fitted["missing_fractions"].items()
        }
        transformer.missing_pattern_distribution_ = {
            tuple(int(value) for value in payload["pattern"]): float(
                payload["probability"]
            )
            for payload in fitted["missing_pattern_distribution"]
        }
        transformer.categorical_distributions_ = {
            col: {
                str(category): float(probability)
                for category, probability in distribution.items()
            }
            for col, distribution in fitted["categorical_distributions"].items()
        }

        transformer._category_encoder.reset()
        transformer._category_encoder.category_mappings_ = {
            col: {str(category): int(code) for category, code in mapping.items()}
            for col, mapping in fitted["category_mappings"].items()
        }
        transformer._category_encoder.inverse_category_mappings_ = {
            col: {code: category for category, code in mapping.items()}
            for col, mapping in transformer._category_encoder.category_mappings_.items()
        }
        transformer.category_mappings_ = transformer._category_encoder.category_mappings_
        transformer.inverse_category_mappings_ = (
            transformer._category_encoder.inverse_category_mappings_
        )
        transformer._is_fitted = True
        return transformer

    @staticmethod
    def _serialize_value(value: Any) -> dict[str, Any]:
        if pd.isna(value):
            return {"type": "missing", "value": None}
        if isinstance(value, pd.Timestamp):
            return {"type": "timestamp", "value": value.isoformat()}
        if isinstance(value, np.integer):
            return {"type": "int", "value": int(value)}
        if isinstance(value, np.floating):
            return {"type": "float", "value": float(value)}
        if isinstance(value, np.bool_):
            return {"type": "bool", "value": bool(value)}
        return {"type": "python", "value": value}

    @staticmethod
    def _deserialize_value(payload: dict[str, Any]) -> Any:
        value_type = payload["type"]
        value = payload["value"]
        if value_type == "missing":
            return np.nan
        if value_type == "timestamp":
            return pd.Timestamp(value)
        return value

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "This IFCTransformer instance is not fitted yet. "
                "Call fit() before using this method."
            )
