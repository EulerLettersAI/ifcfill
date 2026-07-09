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
DEFAULT_CAT_CONSTANT = "__ifcfill_missing__"
_STATE_VERSION = 1

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
        self.missing_counts_: dict[str, int] = {}
        self.missing_fractions_: dict[str, float] = {}
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
                f"Unsupported IFCTransformer state version {state.get('state_version')!r}."
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
        self.fill_values_ = {}
        self.missing_counts_ = {}
        self.missing_fractions_ = {}
        self._category_encoder.reset()
        self.category_mappings_ = self._category_encoder.category_mappings_
        self.inverse_category_mappings_ = self._category_encoder.inverse_category_mappings_

        n = len(df)

        tasks = [(idx, col, df[col], n) for idx, col in enumerate(df.columns)]
        for result in self._map_columns(self._fit_column, tasks):
            col = result["column"]
            self.missing_counts_[col] = result["missing_count"]
            self.missing_fractions_[col] = result["missing_fraction"]

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
        restore_missing: bool = False,
        random_state: int | np.random.Generator | None = None,
    ) -> pd.DataFrame:
        """Restore the structure of the original table from a transformed one.

        Specifically:

        1. Re-inserts dropped constant columns at their original positions.
        2. Decodes label-encoded categorical columns when enabled.
        3. Converts the learned categorical missing category back to ``NaN``
           when ``cat_fill="constant"``.
        4. Reorders columns to match the original input order.
        5. (Optional) Randomly replaces values with ``NaN`` in each
           non-categorical, non-constant column at the same rate as the
           original missing fraction.

        Parameters
        ----------
        data:
            A DataFrame produced by :meth:`transform` (or a CSV of one).
        restore_missing:
            If ``True``, randomly introduce ``NaN`` values in non-categorical
            columns proportional to the missing fractions recorded during
            :meth:`fit`. Categorical missing values represented by the learned
            missing category are restored deterministically regardless of this
            setting.
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
            for col in self.inverse_category_mappings_:
                if col in result.columns:
                    result[col] = self._category_encoder.inverse_transform_column(
                        col,
                        result[col],
                    )

        # Convert the learned categorical missing category back to missing values.
        if self.cat_fill == "constant":
            for col, col_type in self.column_types_.items():
                if col_type == "categorical" and col in result.columns:
                    result[col] = result[col].replace(str(self.fill_values_[col]), np.nan)

        # Convert integer datetime offsets back to pandas timestamps.
        for col, col_type in self.column_types_.items():
            if col_type == "datetime" and col in result.columns:
                result[col] = _numeric_to_datetime(
                    result[col],
                    self.datetime_anchor,
                    self.datetime_unit,
                )

        # Reorder to original column order (only columns present)
        available = [c for c in self.original_columns_ if c in result.columns]
        result = result[available]

        # Optionally restore missing-value distribution (non-constant cols only)
        if restore_missing:
            n = len(result)
            for col in available:
                if col in self.dropped_constants_:
                    continue
                if self.column_types_.get(col) == "categorical":
                    continue
                frac = self.missing_fractions_.get(col, 0.0)
                if frac > 0.0 and n > 0:
                    n_missing = max(1, int(np.round(frac * n)))
                    n_missing = min(n_missing, n)
                    idx = rng.choice(n, size=n_missing, replace=False)
                    if self.column_types_.get(col) == "datetime":
                        result.iloc[idx, result.columns.get_loc(col)] = pd.NaT
                    else:
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

    def _fit_column(
        self,
        task: tuple[int, str, pd.Series, int],
    ) -> dict[str, Any]:
        idx, col, series, n = task

        n_missing = int(series.isna().sum())
        missing_fraction = n_missing / n if n > 0 else 0.0

        col_type: ColType = self.col_types.get(col) or infer_col_type(series)  # type: ignore[assignment]

        unique_vals = series.dropna().unique()
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
                "missing_counts": self.missing_counts_,
                "missing_fractions": self.missing_fractions_,
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
        transformer.missing_counts_ = {
            col: int(count) for col, count in fitted["missing_counts"].items()
        }
        transformer.missing_fractions_ = {
            col: float(fraction)
            for col, fraction in fitted["missing_fractions"].items()
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
