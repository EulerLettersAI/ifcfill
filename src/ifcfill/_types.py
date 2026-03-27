from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

ColType = Literal["integer", "float", "categorical", "datetime"]

_DATETIME_SAMPLE_SIZE = 20
_DATETIME_COERCE_THRESHOLD = 0.8
_NUMERIC_COERCE_THRESHOLD = 0.9


def infer_col_type(series: pd.Series) -> ColType:
    """Infer the IFC column type for *series* using NumPy-backed heuristics.

    Detection order
    ---------------
    1. Native datetime64 dtype → ``"datetime"``
    2. Object dtype that parses as dates → ``"datetime"``
    3. Boolean dtype → ``"categorical"``
    4. Integer dtype → ``"integer"``
    5. Float dtype whose non-null values are all whole numbers → ``"integer"``
    6. Float dtype → ``"float"``
    7. Object dtype that coerces numerically (≥90 % success):
       - whole numbers → ``"integer"``
       - otherwise → ``"float"``
    8. Fallback → ``"categorical"``
    """
    # 1. Native datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    # 2. Object dtype that looks like dates
    if pd.api.types.is_object_dtype(series):
        sample = series.dropna().head(_DATETIME_SAMPLE_SIZE)
        if len(sample) > 0:
            try:
                pd.to_datetime(sample)
                full_parsed = pd.to_datetime(series, errors="coerce")
                n_valid = series.notna().sum()
                if n_valid > 0 and full_parsed.notna().sum() / n_valid >= _DATETIME_COERCE_THRESHOLD:
                    return "datetime"
            except Exception:
                pass

    # 3. Boolean
    if pd.api.types.is_bool_dtype(series):
        return "categorical"

    # 4. Integer
    if pd.api.types.is_integer_dtype(series):
        return "integer"

    # 5 & 6. Float
    if pd.api.types.is_float_dtype(series):
        arr = series.dropna().to_numpy(dtype=float)
        if arr.size > 0 and np.all(arr == np.floor(arr)):
            return "integer"
        return "float"

    # 7. Object → try numeric coercion
    numeric = pd.to_numeric(series, errors="coerce")
    n_valid = series.notna().sum()
    n_coerced = numeric.notna().sum()
    if n_valid > 0 and n_coerced / n_valid >= _NUMERIC_COERCE_THRESHOLD:
        arr = numeric.dropna().to_numpy(dtype=float)
        if arr.size > 0 and np.all(arr == np.floor(arr)):
            return "integer"
        return "float"

    # 8. Fallback
    return "categorical"
