from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from ._impute import compute_fill_categorical, compute_fill_float, compute_fill_integer
from ._io import load_to_dataframe
from ._types import ColType, infer_col_type

# String sentinels that represent null in object columns after .astype(str)
_NULL_SENTINELS = frozenset({"nan", "none", "<na>", "nat", "pd.na", ""})
_CAT_ENCODINGS = ("none", "label")

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
        ``"mode"`` uses the most frequent value; ``"constant"`` uses
        *cat_constant*.
    cat_constant:
        Fill string used when *cat_fill* is ``"constant"``.
    cat_encoding:
        Optional encoding for categorical columns. ``"none"`` keeps
        categorical columns as pandas categoricals. ``"label"`` maps each
        category to an integer code and stores the inverse mapping for
        :meth:`inverse_transform`.
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
    category_mappings_ : dict[str, dict[str, int]]
        Forward mapping for label-encoded categorical columns.
    inverse_category_mappings_ : dict[str, dict[int, str]]
        Inverse mapping for label-encoded categorical columns.

    Examples
    --------
    >>> tf = IFCTransformer()
    >>> transformed = tf.fit_transform("data.csv")
    >>> restored = tf.inverse_transform(transformed, restore_missing=True)
    """

    def __init__(
        self,
        col_types: dict[str, ColType] | None = None,
        int_fill: Literal["mean", "median", "mode", "zero"] = "median",
        float_fill: Literal["mean", "median", "mode", "zero"] = "mean",
        cat_fill: Literal["mode", "constant"] = "mode",
        cat_constant: str = "missing",
        cat_encoding: Literal["none", "label"] = "none",
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
        self.datetime_anchor = pd.Timestamp(datetime_anchor)
        self.datetime_unit = datetime_unit

        # populated by fit()
        self.column_types_: dict[str, ColType] = {}
        self.fill_values_: dict[str, Any] = {}
        self.dropped_constants_: dict[str, tuple[Any, int]] = {}
        self.original_columns_: list[str] = []
        self.missing_counts_: dict[str, int] = {}
        self.missing_fractions_: dict[str, float] = {}
        self.category_mappings_: dict[str, dict[str, int]] = {}
        self.inverse_category_mappings_: dict[str, dict[int, str]] = {}
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
        self.fill_values_ = {}
        self.missing_counts_ = {}
        self.missing_fractions_ = {}
        self.category_mappings_ = {}
        self.inverse_category_mappings_ = {}

        n = len(df)

        for idx, col in enumerate(df.columns):
            series = df[col]

            # Track missing values for all columns
            n_missing = int(series.isna().sum())
            self.missing_counts_[col] = n_missing
            self.missing_fractions_[col] = n_missing / n if n > 0 else 0.0

            # Detect constant columns (≤1 unique non-null value)
            unique_vals = series.dropna().unique()
            if len(unique_vals) <= 1:
                const_val = unique_vals[0] if len(unique_vals) == 1 else np.nan
                self.dropped_constants_[col] = (const_val, idx)
                continue

            # Determine type: user override takes priority over inference
            col_type: ColType = self.col_types.get(col) or infer_col_type(series)  # type: ignore[assignment]
            self.column_types_[col] = col_type

            # Compute fill value
            if col_type == "datetime":
                arr = _datetime_to_numeric(series, self.datetime_anchor, self.datetime_unit)
                valid = arr[~np.isnan(arr)]
                raw = float(np.median(valid)) if valid.size > 0 else 0.0
                self.fill_values_[col] = int(np.round(raw))

            elif col_type == "integer":
                arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
                self.fill_values_[col] = compute_fill_integer(arr, self.int_fill)

            elif col_type == "float":
                arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
                self.fill_values_[col] = compute_fill_float(arr, self.float_fill)

            else:  # categorical
                self.fill_values_[col] = compute_fill_categorical(
                    series.to_numpy(), self.cat_fill, self.cat_constant
                )
                if self.cat_encoding == "label":
                    categories = self._filled_categorical(series, self.fill_values_[col])
                    unique_categories = sorted(pd.unique(categories.astype(str)))
                    mapping = {value: code for code, value in enumerate(unique_categories)}
                    self.category_mappings_[col] = mapping
                    self.inverse_category_mappings_[col] = {
                        code: value for value, code in mapping.items()
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
        result: dict[str, pd.Series] = {}

        for col in df.columns:
            # Drop constant columns
            if col in self.dropped_constants_:
                continue

            # Pass through columns unseen during fit
            if col not in self.column_types_:
                result[col] = df[col]
                continue

            col_type = self.column_types_[col]
            fill_val = self.fill_values_[col]
            series = df[col]

            if col_type == "datetime":
                arr = _datetime_to_numeric(series, self.datetime_anchor, self.datetime_unit)
                arr = np.where(np.isnan(arr), fill_val, arr)
                result[col] = pd.Series(arr.astype(np.int64), index=df.index, name=col)

            elif col_type == "integer":
                arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
                arr = np.where(np.isnan(arr), fill_val, arr)
                result[col] = pd.Series(arr.astype(np.int64), index=df.index, name=col)

            elif col_type == "float":
                arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
                arr = np.where(np.isnan(arr), fill_val, arr)
                result[col] = pd.Series(arr.astype(np.float64), index=df.index, name=col)

            else:  # categorical
                s = self._filled_categorical(series, fill_val)
                if self.cat_encoding == "label":
                    mapping = self.category_mappings_[col]
                    fallback = mapping[str(fill_val)]
                    codes = s.astype(str).map(mapping).fillna(fallback).to_numpy(dtype=np.int64)
                    result[col] = pd.Series(codes, index=df.index, name=col)
                else:
                    result[col] = pd.Categorical(s)

        return pd.DataFrame(result, index=df.index)

    def fit_transform(self, data: str | Path | pd.DataFrame) -> pd.DataFrame:
        """Fit and transform *data* in one step."""
        return self.fit(data).transform(data)

    def inverse_transform(
        self,
        data: str | Path | pd.DataFrame,
        restore_missing: bool = False,
        random_state: int | np.random.Generator | None = None,
    ) -> pd.DataFrame:
        """Restore the structure of the original table from a transformed one.

        Specifically:

        1. Re-inserts dropped constant columns at their original positions.
        2. Reorders columns to match the original input order.
        3. (Optional) Randomly replaces values with ``NaN`` in each
           non-constant column at the same rate as the original missing
           fraction, recreating the original missing-data distribution.

        Parameters
        ----------
        data:
            A DataFrame produced by :meth:`transform` (or a CSV of one).
        restore_missing:
            If ``True``, randomly introduce ``NaN`` values proportional to
            the missing fractions recorded during :meth:`fit`.
        random_state:
            Integer seed or :class:`numpy.random.Generator` for reproducible
            missing-value restoration.

        Returns
        -------
        pandas.DataFrame
        """
        self._check_fitted()
        result = load_to_dataframe(data).copy()
        rng = np.random.default_rng(random_state)

        # Re-add constant columns
        for col, (value, _) in self.dropped_constants_.items():
            result[col] = value

        # Decode label-encoded categorical columns before restoring structure.
        if self.cat_encoding == "label":
            for col, inverse_mapping in self.inverse_category_mappings_.items():
                if col in result.columns:
                    result[col] = self._decode_label_encoded(result[col], inverse_mapping)

        # Reorder to original column order (only columns present)
        available = [c for c in self.original_columns_ if c in result.columns]
        result = result[available]

        # Optionally restore missing-value distribution (non-constant cols only)
        if restore_missing:
            n = len(result)
            for col in available:
                if col in self.dropped_constants_:
                    continue
                frac = self.missing_fractions_.get(col, 0.0)
                if frac > 0.0 and n > 0:
                    n_missing = max(1, int(np.round(frac * n)))
                    n_missing = min(n_missing, n)
                    idx = rng.choice(n, size=n_missing, replace=False)
                    result[col] = result[col].astype(object)
                    result.iloc[idx, result.columns.get_loc(col)] = np.nan

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _filled_categorical(series: pd.Series, fill_val: Any) -> pd.Series:
        s = series.astype(str)
        s = s.where(~s.str.lower().isin(_NULL_SENTINELS), other=fill_val)
        return s.fillna(fill_val)

    @staticmethod
    def _decode_label_encoded(
        series: pd.Series,
        inverse_mapping: dict[int, str],
    ) -> pd.Series:
        codes = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        max_code = max(inverse_mapping)
        decoded: list[str] = []

        for code in codes:
            if np.isnan(code):
                clipped_code = 0
            else:
                clipped_code = int(np.clip(np.round(code), 0, max_code))
            decoded.append(inverse_mapping[clipped_code])

        return pd.Series(decoded, index=series.index, name=series.name)

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "This IFCTransformer instance is not fitted yet. "
                "Call fit() before using this method."
            )
