from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class LabelCategoryEncoder:
    """Fit, apply, and invert integer label encodings for categorical columns."""

    def __init__(self) -> None:
        self.category_mappings_: dict[str, dict[str, int]] = {}
        self.inverse_category_mappings_: dict[str, dict[int, str]] = {}

    def reset(self) -> None:
        self.category_mappings_ = {}
        self.inverse_category_mappings_ = {}

    def fit_column(
        self,
        column: str,
        values: pd.Series,
        fill_value: Any | None = None,
    ) -> None:
        categories = self._categories(values, fill_value)
        mapping = {value: code for code, value in enumerate(categories)}
        self.category_mappings_[column] = mapping
        self.inverse_category_mappings_[column] = {
            code: value for value, code in mapping.items()
        }

    def transform_column(
        self,
        column: str,
        values: pd.Series,
        fallback_value: Any,
    ) -> pd.Series:
        mapping = self.category_mappings_[column]
        fallback = mapping[str(fallback_value)]
        codes = values.astype(str).map(mapping).fillna(fallback).to_numpy(dtype=np.int64)
        return pd.Series(codes, index=values.index, name=values.name)

    def inverse_transform_column(self, column: str, values: pd.Series) -> pd.Series:
        inverse_mapping = self.inverse_category_mappings_[column]
        decoded = self._decode(values, inverse_mapping)
        return pd.Series(decoded, index=values.index, name=values.name)

    def get_mappings(self, inverse: bool = False) -> dict[str, dict[Any, Any]]:
        mappings = self.inverse_category_mappings_ if inverse else self.category_mappings_
        return {col: mapping.copy() for col, mapping in mappings.items()}

    def get_mapping(self, column: str, inverse: bool = False) -> dict[Any, Any]:
        mappings = self.inverse_category_mappings_ if inverse else self.category_mappings_
        if column not in mappings:
            raise KeyError(f"No category mapping was learned for column {column!r}.")
        return mappings[column].copy()

    @staticmethod
    def _categories(values: pd.Series, fill_value: Any | None) -> list[str]:
        category_values = list(pd.unique(values.astype(str)))
        if fill_value is not None:
            fill_category = str(fill_value)
            if fill_category not in category_values:
                category_values.append(fill_category)
        return sorted(category_values)

    @staticmethod
    def _decode(values: pd.Series, inverse_mapping: dict[int, str]) -> list[str]:
        codes = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        max_code = max(inverse_mapping)
        decoded: list[str] = []

        for code in codes:
            if np.isnan(code):
                clipped_code = 0
            else:
                clipped_code = int(np.clip(np.round(code), 0, max_code))
            decoded.append(inverse_mapping[clipped_code])

        return decoded
