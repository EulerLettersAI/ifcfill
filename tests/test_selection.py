import sys

import numpy as np
import pandas as pd
import pytest

from ifcfill import ModeSelection, missingness_dependence_gap, select_mode

pytest.importorskip("sklearn")


def _value_dependent(n: int = 900, seed: int = 0) -> pd.DataFrame:
    """``income`` is missing mostly for older people (missing at random)."""
    rng = np.random.default_rng(seed)
    age = rng.integers(20, 80, n)
    frame = pd.DataFrame(
        {
            "age": age,
            "income": rng.normal(50_000, 10_000, n).round(2),
            "region": rng.choice(["north", "south", "east"], n),
        }
    )
    probability = 1 / (1 + np.exp(-(age - 60) / 4))
    frame.loc[rng.random(n) < probability, "income"] = np.nan
    return frame


def _block(n: int = 900, seed: int = 0) -> pd.DataFrame:
    """``tsh`` and ``t4`` are missing together, at random rows."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "age": rng.integers(20, 80, n),
            "tsh": rng.normal(2.0, 0.5, n),
            "t4": rng.normal(10.0, 2.0, n),
            "sex": rng.choice(["F", "M"], n),
        }
    )
    panel_missing = rng.random(n) < 0.3
    frame.loc[panel_missing, ["tsh", "t4"]] = np.nan
    return frame


class TestSelectMode:
    def test_value_dependent_missingness_selects_hybrid(self):
        choice = select_mode(_value_dependent(), random_state=0)
        assert isinstance(choice, ModeSelection)
        assert choice.mode == "hybrid"
        assert set(choice.gaps) == {"hybrid", "joint", "marginal"}

    def test_block_missingness_selects_joint(self):
        assert select_mode(_block(), random_state=0).mode == "joint"

    def test_is_reproducible(self):
        data = _value_dependent()
        assert select_mode(data, random_state=3) == select_mode(data, random_state=3)

    def test_candidate_subset(self):
        choice = select_mode(_value_dependent(), modes=("joint", "marginal"), random_state=0)
        assert set(choice.gaps) == {"joint", "marginal"}

    def test_generated_mode_is_rejected(self):
        with pytest.raises(ValueError, match="generated"):
            select_mode(_value_dependent(), modes=("generated", "hybrid"))

    def test_too_few_folds_is_rejected(self):
        with pytest.raises(ValueError, match="n_folds"):
            select_mode(_value_dependent(), n_folds=2)

    def test_complete_data_cannot_be_scored(self):
        with pytest.raises(ValueError, match="enough missing"):
            select_mode(_value_dependent().dropna(), random_state=0)

    def test_transformer_params_are_forwarded(self):
        with pytest.raises(ValueError, match="cat_encoding"):
            select_mode(_value_dependent(), cat_encoding="unknown")

    def test_missing_sklearn_raises_helpful_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "sklearn", None)
        with pytest.raises(ImportError, match=r"ifcfill\[select\]"):
            select_mode(_value_dependent())


class TestMissingnessDependenceGap:
    def test_identical_data_has_zero_gap(self):
        data = _value_dependent(seed=1)
        reference, evaluation = data.iloc[:450], data.iloc[450:].reset_index(drop=True)
        gap = missingness_dependence_gap(
            reference, reference, evaluation, categorical_columns=["region"], random_state=0
        )
        assert gap == 0.0

    def test_complete_comparison_loses_the_dependence(self):
        data = _value_dependent(seed=1)
        reference, evaluation = data.iloc[:450], data.iloc[450:].reset_index(drop=True)
        imputed = reference.fillna({"income": reference["income"].mean()})
        gap = missingness_dependence_gap(
            reference, imputed, evaluation, categorical_columns=["region"], random_state=0
        )
        assert gap > 0.2

    def test_mismatched_columns_raise(self):
        data = _value_dependent()
        with pytest.raises(ValueError, match="same column names"):
            missingness_dependence_gap(data, data.drop(columns="age"), data)

    def test_no_scorable_column_returns_nan(self):
        data = _value_dependent().dropna()
        assert np.isnan(missingness_dependence_gap(data, data, data))

    def test_datetime_columns_are_supported(self):
        data = _value_dependent(seed=2)
        data["visit"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(data["age"], unit="D")
        reference, evaluation = data.iloc[:450], data.iloc[450:].reset_index(drop=True)
        gap = missingness_dependence_gap(
            reference,
            reference,
            evaluation,
            categorical_columns=["region"],
            datetime_columns=["visit"],
            random_state=0,
        )
        assert gap == 0.0
