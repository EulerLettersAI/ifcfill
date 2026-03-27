from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

IntStrategy = Literal["mean", "median", "mode", "zero"]
FloatStrategy = Literal["mean", "median", "mode", "zero"]
CatStrategy = Literal["mode", "constant"]

_INT_STRATEGIES = ("mean", "median", "mode", "zero")
_FLOAT_STRATEGIES = ("mean", "median", "mode", "zero")
_CAT_STRATEGIES = ("mode", "constant")


def compute_fill_integer(arr: np.ndarray, strategy: str) -> int:
    """Return an integer fill value computed from *arr* using *strategy*.

    Parameters
    ----------
    arr:
        1-D float array (may contain ``NaN``).
    strategy:
        ``"mean"``   → ``int(round(mean))``
        ``"median"`` → ``int(round(median))``
        ``"mode"``   → most frequent value cast to int
        ``"zero"``   → ``0``
    """
    if strategy not in _INT_STRATEGIES:
        raise ValueError(
            f"Unknown integer fill strategy {strategy!r}. "
            f"Choose from: {_INT_STRATEGIES}."
        )
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return 0
    if strategy == "mean":
        return int(np.round(np.mean(valid)))
    if strategy == "median":
        return int(np.round(np.median(valid)))
    if strategy == "mode":
        values, counts = np.unique(valid, return_counts=True)
        return int(values[np.argmax(counts)])
    return 0  # "zero"


def compute_fill_float(arr: np.ndarray, strategy: str) -> float:
    """Return a float fill value computed from *arr* using *strategy*.

    Parameters
    ----------
    arr:
        1-D float array (may contain ``NaN``).
    strategy:
        ``"mean"``   → arithmetic mean
        ``"median"`` → median
        ``"mode"``   → most frequent value
        ``"zero"``   → ``0.0``
    """
    if strategy not in _FLOAT_STRATEGIES:
        raise ValueError(
            f"Unknown float fill strategy {strategy!r}. "
            f"Choose from: {_FLOAT_STRATEGIES}."
        )
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
    return 0.0  # "zero"


def compute_fill_categorical(arr: np.ndarray, strategy: str, constant: str) -> str:
    """Return a string fill value for a categorical column.

    Parameters
    ----------
    arr:
        Object/string NumPy array (may contain ``None`` / ``NaN``).
    strategy:
        ``"mode"``     → most frequent non-null value
        ``"constant"`` → return *constant* unchanged
    constant:
        Fallback string used when *strategy* is ``"constant"`` or the column
        has no non-null values.
    """
    if strategy not in _CAT_STRATEGIES:
        raise ValueError(
            f"Unknown categorical fill strategy {strategy!r}. "
            f"Choose from: {_CAT_STRATEGIES}."
        )
    if strategy == "constant":
        return constant
    valid = arr[pd.notna(arr)]
    if valid.size == 0:
        return constant
    values, counts = np.unique(valid.astype(str), return_counts=True)
    return str(values[np.argmax(counts)])
