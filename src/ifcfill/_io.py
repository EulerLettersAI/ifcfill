from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_to_dataframe(data: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Return a pandas DataFrame from a CSV path or an existing DataFrame.

    Parameters
    ----------
    data:
        Path to a CSV file (``str`` or :class:`pathlib.Path`) or a
        :class:`pandas.DataFrame`.

    Returns
    -------
    pandas.DataFrame
        A copy of the input data as a DataFrame.

    Raises
    ------
    TypeError
        If *data* is not a supported type.
    FileNotFoundError
        If *data* is a path that does not exist.
    """
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, (str, Path)):
        path = Path(data)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")
        return pd.read_csv(path)
    raise TypeError(
        f"Expected a path to a CSV file or a pandas DataFrame, "
        f"got {type(data).__name__!r}"
    )
