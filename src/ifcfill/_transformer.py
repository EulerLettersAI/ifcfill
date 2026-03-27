from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from ._io import load_to_dataframe

_IntStrategy = Literal["median", "mode", "zero"]
_FloatStrategy = Literal["mean", "median", "zero"]
_CatStrategy = Literal["mode", "constant"]


def _infer_type(series: pd.Series) -> Literal["integer", "float", "categorical"]:
    """Infer the IFC type of a pandas Series using NumPy."""
    if pd.api.types.is_bool_dtype(series):
        return "categorical"

    if pd.api.types.is_integer_dtype(series):
        return "integer"

    if pd.api.types.is_float_dtype(series):
        arr = series.dropna().to_numpy(dtype=float)
        if arr.size > 0 and np.all(arr == np.floor(arr)):
            return "integer"
        return "float"

    # object / string dtype — attempt numeric coercion
    numeric = pd.to_numeric(series, errors="coerce")
    total_valid = series.notna().sum()
    coercible = numeric.notna().sum()

    if total_valid > 0 and coercible / total_valid >= 0.9:
        arr = numeric.dropna().to_numpy(dtype=float)
        if arr.size > 0 and np.all(arr == np.floor(arr)):
            return "integer"
        return "float"

    return "categorical"


def _numeric_fill(arr: np.ndarray, strategy: str) -> float:
    """Compute a fill value for a numeric column using NumPy."""
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return 0.0
    if strategy == "mean":
        return float(np.mean(valid))
    if strategy == "median":
        return float(np.median(valid))
    if strategy == "mode":
        values, counts = np.unique(valid, return_counts=True)
        return float(values[np.argmax(counts)])
    if strategy == "zero":
        return 0.0
    raise ValueError(f"Unknown strategy {strategy!r}")


def _categorical_fill(arr: np.ndarray, strategy: str, constant: str) -> str:
    """Compute a fill value for a categorical column."""
    if strategy == "constant":
        return constant
    valid = arr[pd.notna(arr)]
    if valid.size == 0:
        return constant
    values, counts = np.unique(valid.astype(str), return_counts=True)
    return str(values[np.argmax(counts)])


class IFCTransformer:
    """Transform tabular data into Integer, Float and Categorical (IFC) columns
    and fill missing values using a NumPy-based engine.

    Accepts a path to a CSV file or a :class:`pandas.DataFrame` as input.

    Parameters
    ----------
    int_fill : {"median", "mode", "zero"}
        Strategy for filling missing values in integer columns.
    float_fill : {"mean", "median", "zero"}
        Strategy for filling missing values in float columns.
    cat_fill : {"mode", "constant"}
        Strategy for filling missing values in categorical columns.
    cat_constant : str
        Constant string used when *cat_fill* is ``"constant"``.

    Examples
    --------
    >>> from ifcfill import IFCTransformer
    >>> tf = IFCTransformer()
    >>> result = tf.fit_transform("data.csv")      # from CSV
    >>> result = tf.fit_transform(df)              # from DataFrame
    """

    def __init__(
        self,
        int_fill: _IntStrategy = "median",
        float_fill: _FloatStrategy = "mean",
        cat_fill: _CatStrategy = "mode",
        cat_constant: str = "missing",
    ) -> None:
        self.int_fill = int_fill
        self.float_fill = float_fill
        self.cat_fill = cat_fill
        self.cat_constant = cat_constant

        # set after fit()
        self.column_types_: dict[str, str] = {}
        self.fill_values_: dict[str, object] = {}
        self._is_fitted: bool = False

    def fit(self, data: str | Path | pd.DataFrame) -> IFCTransformer:
        """Detect column types and compute fill values from *data*.

        Parameters
        ----------
        data:
            A path to a CSV file or a :class:`pandas.DataFrame`.

        Returns
        -------
        self
        """
        df = load_to_dataframe(data)
        self.column_types_ = {}
        self.fill_values_ = {}

        for col in df.columns:
            col_type = _infer_type(df[col])
            self.column_types_[col] = col_type

            if col_type in ("integer", "float"):
                numeric = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
                strategy = self.int_fill if col_type == "integer" else self.float_fill
                self.fill_values_[col] = _numeric_fill(numeric, strategy)
            else:
                self.fill_values_[col] = _categorical_fill(
                    df[col].to_numpy(), self.cat_fill, self.cat_constant
                )

        self._is_fitted = True
        return self

    def transform(self, data: str | Path | pd.DataFrame) -> pd.DataFrame:
        """Apply type casting and missing value fill to *data*.

        Parameters
        ----------
        data:
            A path to a CSV file or a :class:`pandas.DataFrame`.

        Returns
        -------
        pandas.DataFrame
            DataFrame with IFC-typed columns and no missing values.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called first.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform().")

        df = load_to_dataframe(data)
        result: dict[str, pd.Series] = {}

        for col in df.columns:
            if col not in self.column_types_:
                result[col] = df[col]
                continue

            col_type = self.column_types_[col]
            fill_val = self.fill_values_[col]
            series = df[col].copy()

            if col_type == "integer":
                numeric = pd.to_numeric(series, errors="coerce").fillna(fill_val)
                result[col] = numeric.astype(np.int64)

            elif col_type == "float":
                numeric = pd.to_numeric(series, errors="coerce").fillna(fill_val)
                result[col] = numeric.astype(np.float64)

            else:
                series = series.astype(str).replace(
                    {"nan": fill_val, "None": fill_val, "<NA>": fill_val, "NaT": fill_val}
                )
                series = series.fillna(fill_val)
                result[col] = pd.Categorical(series)

        return pd.DataFrame(result)

    def fit_transform(self, data: str | Path | pd.DataFrame) -> pd.DataFrame:
        """Fit and transform *data* in one step.

        Parameters
        ----------
        data:
            A path to a CSV file or a :class:`pandas.DataFrame`.

        Returns
        -------
        pandas.DataFrame
        """
        return self.fit(data).transform(data)
