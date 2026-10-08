import json

import numpy as np
import pandas as pd
import pytest

from ifcfill import IFCTransformer


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Mixed-type DataFrame with missing values in every column."""
    return pd.DataFrame(
        {
            "age":    [25, 30, np.nan, 40],
            "salary": [50_000.5, np.nan, 75_000.0, 90_000.25],
            "city":   ["London", None, "Paris", "London"],
            "score":  [1.0, 2.0, np.nan, 4.0],       # whole-number floats → integer
            "const":  ["x", "x", "x", "x"],           # constant → should be dropped
            "dates":  pd.to_datetime(["2020-01-01", "2021-06-15", None, "2023-03-10"]),
        }
    )


@pytest.fixture()
def fitted(sample_df) -> IFCTransformer:
    tf = IFCTransformer()
    tf.fit(sample_df)
    return tf


def resized_rows(df: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    """Repeat transformed rows to a requested synthetic sample size."""
    if n_rows == 0:
        return df.iloc[:0].copy()
    return df.iloc[np.arange(n_rows) % len(df)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature 1 – Input types: CSV and DataFrame
# ---------------------------------------------------------------------------

def test_fit_transform_from_dataframe(sample_df):
    result = IFCTransformer().fit_transform(sample_df)
    assert isinstance(result, pd.DataFrame)


def test_fit_transform_from_csv(tmp_path, sample_df):
    csv_file = tmp_path / "data.csv"
    # Write a simple CSV without datetime to keep CSV parsing simple
    df = pd.DataFrame({"a": [1, 2, None], "b": ["x", None, "z"]})
    df.to_csv(csv_file, index=False)
    result = IFCTransformer().fit_transform(str(csv_file))
    assert isinstance(result, pd.DataFrame)
    assert result.isnull().sum().sum() == 0


def test_missing_csv_raises():
    with pytest.raises(FileNotFoundError):
        IFCTransformer().fit("/nonexistent/path/data.csv")


def test_invalid_input_type_raises():
    with pytest.raises(TypeError):
        IFCTransformer().fit(12345)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Feature 2 – User-specified column types override
# ---------------------------------------------------------------------------

def test_user_type_override_forces_categorical(sample_df):
    tf = IFCTransformer(col_types={"age": "categorical"})
    result = tf.fit_transform(sample_df)
    assert hasattr(result["age"], "cat"), "age should be Categorical when overridden"


def test_user_type_override_forces_float(sample_df):
    tf = IFCTransformer(col_types={"score": "float"})
    result = tf.fit_transform(sample_df)
    assert result["score"].dtype == np.float64


def test_inferred_types_match_expectations(fitted, sample_df):
    assert fitted.column_types_["age"] == "integer"
    assert fitted.column_types_["salary"] == "float"
    assert fitted.column_types_["city"] == "categorical"
    assert fitted.column_types_["score"] == "integer"
    assert fitted.column_types_["dates"] == "datetime"


# ---------------------------------------------------------------------------
# Feature 3 – Imputation strategies
# ---------------------------------------------------------------------------

def test_integer_fill_median(sample_df):
    tf = IFCTransformer(int_fill="median")
    result = tf.fit_transform(sample_df)
    expected = int(np.round(np.nanmedian([25, 30, 40])))
    assert result["age"].iloc[2] == expected


def test_integer_fill_mean(sample_df):
    tf = IFCTransformer(int_fill="mean")
    result = tf.fit_transform(sample_df)
    expected = int(np.round(np.nanmean([25, 30, 40])))
    assert result["age"].iloc[2] == expected


def test_integer_fill_mode(sample_df):
    df = pd.DataFrame({"v": [1, 2, 2, None]})
    tf = IFCTransformer(int_fill="mode")
    result = tf.fit_transform(df)
    assert result["v"].iloc[3] == 2


def test_integer_fill_zero(sample_df):
    tf = IFCTransformer(int_fill="zero")
    result = tf.fit_transform(sample_df)
    assert result["age"].iloc[2] == 0


def test_float_fill_mean(sample_df):
    tf = IFCTransformer(float_fill="mean")
    result = tf.fit_transform(sample_df)
    expected = np.nanmean([50_000.5, 75_000.0, 90_000.25])
    assert result["salary"].iloc[1] == pytest.approx(expected)


def test_float_fill_median(sample_df):
    tf = IFCTransformer(float_fill="median")
    result = tf.fit_transform(sample_df)
    expected = float(np.nanmedian([50_000.5, 75_000.0, 90_000.25]))
    assert result["salary"].iloc[1] == pytest.approx(expected)


def test_cat_fill_mode(sample_df):
    tf = IFCTransformer(cat_fill="mode")
    result = tf.fit_transform(sample_df)
    assert result["city"].iloc[1] == "London"   # London appears twice


def test_cat_fill_constant(sample_df):
    tf = IFCTransformer(cat_fill="constant", cat_constant="unknown")
    result = tf.fit_transform(sample_df)
    assert result["city"].iloc[1] == "unknown"


# ---------------------------------------------------------------------------
# Feature 4 – NumPy-based output (no missing values, correct dtypes)
# ---------------------------------------------------------------------------

def test_no_missing_values_after_transform(sample_df):
    result = IFCTransformer().fit_transform(sample_df)
    assert result.isnull().sum().sum() == 0


def test_integer_columns_are_int64(sample_df):
    result = IFCTransformer().fit_transform(sample_df)
    assert result["age"].dtype == np.int64
    assert result["score"].dtype == np.int64


def test_float_columns_are_float64(sample_df):
    result = IFCTransformer().fit_transform(sample_df)
    assert result["salary"].dtype == np.float64


def test_categorical_columns_have_cat_accessor(sample_df):
    result = IFCTransformer().fit_transform(sample_df)
    assert hasattr(result["city"], "cat")


def test_label_encoded_categorical_columns_are_int64(sample_df):
    result = IFCTransformer(cat_encoding="label").fit_transform(sample_df)
    assert result["city"].dtype == np.int64


def test_parallel_fit_transform_matches_sequential(sample_df):
    sequential = IFCTransformer(cat_encoding="label")
    parallel = IFCTransformer(cat_encoding="label", n_jobs=2)

    sequential_result = sequential.fit_transform(sample_df)
    parallel_result = parallel.fit_transform(sample_df)

    pd.testing.assert_frame_equal(parallel_result, sequential_result)
    assert parallel.column_types_ == sequential.column_types_
    assert parallel.fill_values_ == sequential.fill_values_
    assert parallel.dropped_constants_ == sequential.dropped_constants_
    assert parallel.category_mappings_ == sequential.category_mappings_


def test_n_jobs_minus_one_matches_sequential(sample_df):
    sequential = IFCTransformer().fit_transform(sample_df)
    parallel = IFCTransformer(n_jobs=-1).fit_transform(sample_df)

    pd.testing.assert_frame_equal(parallel, sequential)


def test_invalid_n_jobs_zero_raises():
    with pytest.raises(ValueError, match="n_jobs"):
        IFCTransformer(n_jobs=0)


def test_transform_already_transformed_data_returns_unchanged(sample_df):
    tf = IFCTransformer()
    transformed = tf.fit_transform(sample_df)

    with pytest.warns(UserWarning, match="already transformed"):
        transformed_again = tf.transform(transformed)

    pd.testing.assert_frame_equal(transformed_again, transformed)


def test_transform_already_label_encoded_data_returns_unchanged(sample_df):
    tf = IFCTransformer(cat_encoding="label")
    transformed = tf.fit_transform(sample_df)

    with pytest.warns(UserWarning, match="already transformed"):
        transformed_again = tf.transform(transformed)

    pd.testing.assert_frame_equal(transformed_again, transformed)


def test_label_encoding_mapping_is_tracked(sample_df):
    tf = IFCTransformer(cat_encoding="label")
    tf.fit(sample_df)
    assert tf.category_mappings_["city"] == {
        "London": 0,
        "Paris": 1,
        "__ifcfill_missing__": 2,
    }
    assert tf.inverse_category_mappings_["city"] == {
        0: "London",
        1: "Paris",
        2: "__ifcfill_missing__",
    }


def test_label_encoding_includes_constant_fill_category_without_fit_missing():
    fit_df = pd.DataFrame({"city": ["London", "Paris", "London"]})
    transform_df = pd.DataFrame({"city": ["London", None, "Paris"]})
    tf = IFCTransformer(cat_encoding="label", cat_fill="constant", cat_constant="missing")

    tf.fit(fit_df)
    transformed = tf.transform(transform_df)

    assert tf.category_mappings_["city"] == {
        "London": 0,
        "Paris": 1,
        "missing": 2,
    }
    assert transformed["city"].iloc[1] == tf.category_mappings_["city"]["missing"]


def test_label_encoding_keeps_single_category_plus_missing():
    df = pd.DataFrame({"city": ["London", None, "London"]})
    tf = IFCTransformer(cat_encoding="label", cat_fill="constant", cat_constant="missing")

    transformed = tf.fit_transform(df)

    assert "city" not in tf.dropped_constants_
    assert tf.category_mappings_["city"] == {"London": 0, "missing": 1}
    assert list(transformed["city"]) == [0, 1, 0]


def test_category_mapping_accessors_return_copies(sample_df):
    tf = IFCTransformer(cat_encoding="label")
    tf.fit(sample_df)

    mappings = tf.get_category_mappings()
    inverse_mapping = tf.get_category_mapping("city", inverse=True)
    mappings["city"]["Berlin"] = 99
    inverse_mapping[99] = "Berlin"

    assert "Berlin" not in tf.category_mappings_["city"]
    assert 99 not in tf.inverse_category_mappings_["city"]


def test_label_encoding_inverse_restores_categories(sample_df):
    tf = IFCTransformer(cat_encoding="label")
    transformed = tf.fit_transform(sample_df)
    restored = tf.inverse_transform(transformed, random_state=10)
    assert restored["city"].isna().sum() == 1
    assert set(restored["city"].dropna()).issubset({"London", "Paris"})


def test_label_encoding_inverse_rounds_and_clips_generated_codes(sample_df):
    fit_df = pd.DataFrame({"city": ["London", "Paris", "London", "Paris"]})
    tf = IFCTransformer(cat_encoding="label")
    transformed = tf.fit_transform(fit_df)
    generated = transformed.copy()
    generated["city"] = [-1.4, 0.2, 0.8, 99.0]
    restored = tf.inverse_transform(generated, random_state=0)
    assert list(restored["city"].iloc[:3]) == ["London", "London", "Paris"]
    assert restored["city"].iloc[3] in {"London", "Paris"}
    assert not restored["city"].isna().any()


def test_label_encoding_transform_unseen_category_uses_fill_value(sample_df):
    tf = IFCTransformer(cat_encoding="label")
    tf.fit(sample_df)
    new_df = sample_df.copy()
    new_df.loc[0, "city"] = "Berlin"
    transformed = tf.transform(new_df)
    assert transformed["city"].iloc[0] == tf.category_mappings_["city"]["__ifcfill_missing__"]


def test_categorical_missing_category_inverts_to_nan(sample_df):
    tf = IFCTransformer()
    transformed = tf.fit_transform(sample_df)
    assert transformed["city"].iloc[1] == "__ifcfill_missing__"
    restored = tf.inverse_transform(transformed, random_state=2)
    assert restored["city"].isna().sum() == 1
    assert "__ifcfill_missing__" not in restored["city"].astype(str).tolist()


def test_custom_categorical_missing_category_inverts_to_nan(sample_df):
    tf = IFCTransformer(cat_fill="constant", cat_constant="UNKNOWN")
    transformed = tf.fit_transform(sample_df)
    assert transformed["city"].iloc[1] == "UNKNOWN"
    restored = tf.inverse_transform(transformed, random_state=2)
    assert restored["city"].isna().sum() == 1
    assert "UNKNOWN" not in restored["city"].astype(str).tolist()


# ---------------------------------------------------------------------------
# Feature 5 – Datetime → integer conversion
# ---------------------------------------------------------------------------

def test_datetime_column_becomes_int64(sample_df):
    result = IFCTransformer().fit_transform(sample_df)
    assert result["dates"].dtype == np.int64


def test_datetime_anchor_epoch(sample_df):
    tf = IFCTransformer(datetime_anchor="1970-01-01", datetime_unit="D")
    result = tf.fit_transform(sample_df)
    # 2020-01-01 is 18262 days after 1970-01-01
    expected_days = (pd.Timestamp("2020-01-01") - pd.Timestamp("1970-01-01")).days
    assert result["dates"].iloc[0] == expected_days


def test_datetime_custom_anchor(sample_df):
    tf = IFCTransformer(datetime_anchor="2020-01-01", datetime_unit="D")
    result = tf.fit_transform(sample_df)
    assert result["dates"].iloc[0] == 0   # 2020-01-01 → 0 days from anchor


def test_datetime_missing_filled(sample_df):
    result = IFCTransformer().fit_transform(sample_df)
    # Index 2 was NaT — should be filled with median of the other three
    assert pd.notna(result["dates"].iloc[2])


def test_inverse_datetime_restores_timestamp_dtype_and_values(sample_df):
    complete = sample_df.drop(index=2).reset_index(drop=True)
    tf = IFCTransformer()
    transformed = tf.fit_transform(complete)

    restored = tf.inverse_transform(transformed)

    assert pd.api.types.is_datetime64_any_dtype(restored["dates"])
    pd.testing.assert_series_equal(restored["dates"], complete["dates"])


def test_inverse_datetime_restores_seconds_unit():
    df = pd.DataFrame(
        {
            "event_at": pd.to_datetime(
                ["2024-01-01 00:00:05", "2024-01-01 00:01:30"]
            ),
            "value": [1, 2],
        }
    )
    tf = IFCTransformer(datetime_anchor="2024-01-01", datetime_unit="s")
    transformed = tf.fit_transform(df)

    restored = tf.inverse_transform(transformed)

    pd.testing.assert_series_equal(restored["event_at"], df["event_at"])


def test_inverse_datetime_restore_missing_uses_nat(sample_df):
    tf = IFCTransformer()
    transformed = tf.fit_transform(sample_df)

    restored = tf.inverse_transform(transformed, random_state=1)

    assert pd.api.types.is_datetime64_any_dtype(restored["dates"])
    assert restored["dates"].isna().sum() == 1


# ---------------------------------------------------------------------------
# Feature 6 – Constant columns are dropped
# ---------------------------------------------------------------------------

def test_constant_column_not_in_output(sample_df):
    result = IFCTransformer().fit_transform(sample_df)
    assert "const" not in result.columns


def test_constant_column_tracked(fitted):
    assert "const" in fitted.dropped_constants_
    value, position = fitted.dropped_constants_["const"]
    assert value == "x"


def test_all_null_column_dropped():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [np.nan, np.nan, np.nan]})
    tf = IFCTransformer().fit(df)
    assert "b" in tf.dropped_constants_


# ---------------------------------------------------------------------------
# Feature 7 – Missing value distribution tracking
# ---------------------------------------------------------------------------

def test_missing_counts_tracked(fitted, sample_df):
    assert fitted.missing_counts_["age"] == 1
    assert fitted.missing_counts_["salary"] == 1
    assert fitted.missing_counts_["city"] == 1
    assert fitted.missing_counts_["const"] == 0


def test_missing_fractions_tracked(fitted, sample_df):
    assert fitted.missing_fractions_["age"] == pytest.approx(0.25)


def test_missing_report_is_dataframe(fitted, sample_df):
    report = fitted.missing_report_
    assert isinstance(report, pd.DataFrame)
    assert set(report.columns) == {"column", "type", "missing_count", "missing_fraction"}
    # All original columns including "const" should appear
    assert set(report["column"]) == set(sample_df.columns)


def test_missing_report_constant_type(fitted):
    report = fitted.missing_report_
    const_row = report[report["column"] == "const"].iloc[0]
    assert const_row["type"] == "constant"


def test_missing_report_before_fit_raises():
    with pytest.raises(RuntimeError):
        IFCTransformer().missing_report_


# ---------------------------------------------------------------------------
# Feature 8 – inverse_transform
# ---------------------------------------------------------------------------

def test_inverse_restores_constant_columns(fitted, sample_df):
    transformed = fitted.transform(sample_df)
    restored = fitted.inverse_transform(transformed)
    assert "const" in restored.columns


def test_inverse_restores_original_column_order(fitted, sample_df):
    transformed = fitted.transform(sample_df)
    restored = fitted.inverse_transform(transformed)
    assert list(restored.columns) == list(sample_df.columns)


def test_inverse_constant_value_is_correct(fitted, sample_df):
    transformed = fitted.transform(sample_df)
    restored = fitted.inverse_transform(transformed)
    assert (restored["const"] == "x").all()


def test_inverse_restore_missing_introduces_nans(fitted, sample_df):
    transformed = fitted.transform(sample_df)
    restored = fitted.inverse_transform(transformed, random_state=42)
    for col in ["age", "salary", "city", "score", "dates"]:
        assert restored[col].isna().sum() == 1


def test_inverse_restore_missing_reproducible(fitted, sample_df):
    transformed = fitted.transform(sample_df)
    r1 = fitted.inverse_transform(transformed, random_state=0)
    r2 = fitted.inverse_transform(transformed, random_state=0)
    pd.testing.assert_frame_equal(r1, r2)


def test_inverse_restore_missing_argument_is_removed(fitted, sample_df):
    transformed = fitted.transform(sample_df)
    with pytest.raises(TypeError, match="restore_missing"):
        fitted.inverse_transform(transformed, restore_missing=False)  # type: ignore[call-arg]


def test_save_and_load_preserves_inverse_transform_state(tmp_path, sample_df):
    tf = IFCTransformer(cat_encoding="label", cat_fill="constant", cat_constant="missing")
    transformed = tf.fit_transform(sample_df)
    state_file = tmp_path / "ifcfill-state.json"

    tf.save(state_file)
    loaded = IFCTransformer.load(state_file)
    restored = loaded.inverse_transform(transformed, random_state=17)
    expected = tf.inverse_transform(transformed, random_state=17)

    assert loaded.category_mappings_ == tf.category_mappings_
    assert loaded.inverse_category_mappings_ == tf.inverse_category_mappings_
    assert loaded.categorical_distributions_ == tf.categorical_distributions_
    assert loaded.missing_pattern_distribution_ == tf.missing_pattern_distribution_
    assert list(restored.columns) == list(sample_df.columns)
    pd.testing.assert_frame_equal(restored, expected)
    assert pd.api.types.is_datetime64_any_dtype(restored["dates"])
    assert (restored["const"] == "x").all()


def test_save_and_load_preserves_transform_state(tmp_path, sample_df):
    tf = IFCTransformer(cat_encoding="label")
    tf.fit(sample_df)
    state_file = tmp_path / "ifcfill-state.json"
    new_df = sample_df.copy()
    new_df.loc[0, "city"] = "Berlin"

    tf.save(state_file)
    loaded = IFCTransformer.load(state_file)

    pd.testing.assert_frame_equal(loaded.transform(new_df), tf.transform(new_df))


def test_save_before_fit_raises(tmp_path):
    with pytest.raises(RuntimeError, match="not fitted"):
        IFCTransformer().save(tmp_path / "ifcfill-state.json")


# ---------------------------------------------------------------------------
# Feature 9 – authoritative marginal and joint missingness reconstruction
# ---------------------------------------------------------------------------

def test_invalid_missingness_restore_raises(fitted, sample_df):
    transformed = fitted.transform(sample_df)
    with pytest.raises(ValueError, match="missingness_restore"):
        fitted.inverse_transform(  # type: ignore[arg-type]
            transformed, missingness_restore="conditional"
        )


def test_default_missingness_restore_is_marginal(fitted, sample_df):
    transformed = fitted.transform(sample_df)
    default = fitted.inverse_transform(transformed, random_state=4)
    explicit = fitted.inverse_transform(
        transformed, missingness_restore="marginal", random_state=4
    )
    pd.testing.assert_frame_equal(default, explicit)


@pytest.mark.parametrize("n_rows", [3, 8, 13])
def test_marginal_restoration_all_types_and_sample_sizes(n_rows):
    df = pd.DataFrame(
        {
            "integer": pd.Series([1, 2, pd.NA, 4, 5, pd.NA, 7, 8], dtype="Int64"),
            "float": [1.5, np.nan, 3.5, 4.5, 5.5, np.nan, 7.5, 8.5],
            "datetime": pd.to_datetime(
                [
                    "2024-01-01",
                    None,
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                    None,
                    "2024-01-07",
                    "2024-01-08",
                ]
            ),
            "category": ["a", None, "b", "a", "b", None, "a", "b"],
        }
    )
    tf = IFCTransformer()
    transformed = tf.fit_transform(df)
    synthetic = resized_rows(transformed, n_rows)

    restored = tf.inverse_transform(synthetic, random_state=8)

    expected_missing = round(0.25 * n_rows)
    assert restored.isna().sum().to_dict() == {
        column: expected_missing for column in df.columns
    }
    assert str(restored["integer"].dtype) == "Int64"
    assert pd.api.types.is_float_dtype(restored["float"])
    assert pd.api.types.is_datetime64_any_dtype(restored["datetime"])
    assert "__ifcfill_missing__" not in restored["category"].astype(str).tolist()


def _dependent_missingness_data() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "A": np.arange(10, dtype=float),
            "B": np.arange(10, dtype=float) + 20,
            "C": np.arange(10, dtype=float) + 40,
        }
    )
    df.loc[5:7, ["A", "B"]] = np.nan
    df.loc[8:9, "A"] = np.nan
    return df


@pytest.mark.parametrize(
    ("n_rows", "expected"),
    [
        (7, {(0, 0, 0): 4, (1, 1, 0): 2, (1, 0, 0): 1}),
        (10, {(0, 0, 0): 5, (1, 1, 0): 3, (1, 0, 0): 2}),
        (13, {(0, 0, 0): 6, (1, 1, 0): 4, (1, 0, 0): 3}),
    ],
)
def test_joint_restoration_uses_largest_remainder_for_sample_sizes(n_rows, expected):
    df = _dependent_missingness_data()
    tf = IFCTransformer()
    transformed = tf.fit_transform(df)

    restored = tf.inverse_transform(
        resized_rows(transformed, n_rows),
        missingness_restore="joint",
        random_state=11,
    )

    patterns = restored.isna().astype(int).value_counts(sort=False).to_dict()
    assert patterns == expected


def test_joint_distribution_is_learned_before_transformation():
    tf = IFCTransformer().fit(_dependent_missingness_data())
    assert tf.missing_pattern_distribution_ == {
        (0, 0, 0): pytest.approx(0.5),
        (1, 0, 0): pytest.approx(0.2),
        (1, 1, 0): pytest.approx(0.3),
    }


def test_joint_restoration_includes_categorical_columns():
    df = _dependent_missingness_data()
    df["B"] = ["x", "y", "x", "y", "x", None, None, None, "x", "y"]
    tf = IFCTransformer()
    transformed = tf.fit_transform(df)

    restored = tf.inverse_transform(
        transformed, missingness_restore="joint", random_state=19
    )

    patterns = restored.isna().astype(int).value_counts(sort=False).to_dict()
    assert patterns == {(0, 0, 0): 5, (1, 1, 0): 3, (1, 0, 0): 2}
    assert "__ifcfill_missing__" not in restored["B"].astype(str).tolist()


def test_marginal_restoration_does_not_force_joint_dependency():
    df = _dependent_missingness_data()
    tf = IFCTransformer()
    transformed = tf.fit_transform(df)
    restored = tf.inverse_transform(
        resized_rows(transformed, 100),
        missingness_restore="marginal",
        random_state=3,
    )

    assert restored["A"].isna().sum() == 50
    assert restored["B"].isna().sum() == 30
    assert ((restored["A"].notna()) & (restored["B"].isna())).any()


@pytest.mark.parametrize("mode", ["marginal", "joint"])
def test_missingness_reconstruction_is_reproducible(mode):
    df = _dependent_missingness_data()
    tf = IFCTransformer()
    transformed = resized_rows(tf.fit_transform(df), 23)

    first = tf.inverse_transform(
        transformed, missingness_restore=mode, random_state=123
    )
    second = tf.inverse_transform(
        transformed, missingness_restore=mode, random_state=123
    )

    pd.testing.assert_frame_equal(first, second)


def test_constant_with_missing_restores_missingness_and_integer_dtype():
    df = pd.DataFrame(
        {"constant_integer": pd.Series([5, pd.NA, 5, pd.NA, 5], dtype="Int64")}
    )
    tf = IFCTransformer()
    transformed = tf.fit_transform(df)
    assert "constant_integer" in tf.dropped_constants_

    restored = tf.inverse_transform(transformed, random_state=0)

    assert restored["constant_integer"].isna().sum() == 2
    assert str(restored["constant_integer"].dtype) == "Int64"
    assert (restored["constant_integer"].dropna() == 5).all()


# ---------------------------------------------------------------------------
# Feature 10 – categorical sentinel authority and fallback
# ---------------------------------------------------------------------------

def _categorical_transformer() -> tuple[IFCTransformer, pd.DataFrame]:
    real = pd.DataFrame(
        {"category": ["A", "A", "A", "A", "A", "B", "B", "B", None, None]}
    )
    tf = IFCTransformer()
    return tf, tf.fit_transform(real)


def test_excess_generated_sentinels_are_replaced_from_synthetic_values():
    tf, transformed = _categorical_transformer()
    generated = transformed.copy()
    generated["category"] = ["__ifcfill_missing__"] * 6 + ["SYNTHETIC"] * 4

    restored = tf.inverse_transform(generated, random_state=6)

    assert restored["category"].isna().sum() == 2
    assert set(restored["category"].dropna()) == {"SYNTHETIC"}
    assert "__ifcfill_missing__" not in restored["category"].astype(str).tolist()


def test_fewer_generated_sentinels_does_not_reduce_final_missingness():
    tf, transformed = _categorical_transformer()
    generated = transformed.copy()
    generated["category"] = "SYNTHETIC"

    restored = tf.inverse_transform(generated, random_state=6)

    assert restored["category"].isna().sum() == 2
    assert set(restored["category"].dropna()) == {"SYNTHETIC"}


def test_all_generated_sentinels_fall_back_to_fitted_distribution():
    tf, transformed = _categorical_transformer()
    generated = transformed.copy()
    generated["category"] = "__ifcfill_missing__"

    with pytest.warns(UserWarning, match="fitted real-data categorical distribution"):
        first = tf.inverse_transform(generated, random_state=91)
    with pytest.warns(UserWarning):
        second = tf.inverse_transform(generated, random_state=91)

    pd.testing.assert_frame_equal(first, second)
    assert first["category"].isna().sum() == 2
    assert set(first["category"].dropna()).issubset({"A", "B"})
    assert "B" in set(first["category"].dropna())
    assert "__ifcfill_missing__" not in first["category"].astype(str).tolist()


# ---------------------------------------------------------------------------
# Feature 11 – semantic numeric restoration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("col_types", [None, {"value": "integer"}])
def test_integer_values_are_rounded_to_nullable_int64_without_clipping(col_types):
    tf = IFCTransformer(col_types=col_types)
    tf.fit(pd.DataFrame({"value": [1, 2, 3]}))
    generated = pd.DataFrame({"value": [24.7, 31.2, 100.9]})

    restored = tf.inverse_transform(generated, random_state=0)

    assert restored["value"].tolist() == [25, 31, 101]
    assert str(restored["value"].dtype) == "Int64"


def test_integer_dtype_remains_nullable_with_reconstructed_missingness():
    real = pd.DataFrame(
        {"value": pd.Series([1, pd.NA, 3, 4], dtype="Int64")}
    )
    tf = IFCTransformer()
    tf.fit(real)
    generated = pd.DataFrame({"value": [24.7, 31.2, 42.9, 50.1]})

    restored = tf.inverse_transform(generated, random_state=2)

    assert str(restored["value"].dtype) == "Int64"
    assert restored["value"].isna().sum() == 1
    assert set(restored["value"].dropna()).issubset({25, 31, 43, 50})


def test_float_dtype_remains_float_after_missingness_reconstruction():
    real = pd.DataFrame({"value": [1.5, np.nan, 3.5, 4.5]})
    tf = IFCTransformer()
    transformed = tf.fit_transform(real)

    restored = tf.inverse_transform(transformed, random_state=2)

    assert pd.api.types.is_float_dtype(restored["value"])
    assert restored["value"].isna().sum() == 1


def test_save_load_preserves_marginal_and_joint_reconstruction(tmp_path):
    df = _dependent_missingness_data()
    tf = IFCTransformer()
    transformed = resized_rows(tf.fit_transform(df), 13)
    state_file = tmp_path / "ifcfill-state-v2.json"
    tf.save(state_file)
    loaded = IFCTransformer.load(state_file)

    assert loaded.missing_pattern_distribution_ == tf.missing_pattern_distribution_
    assert loaded.categorical_distributions_ == tf.categorical_distributions_
    for mode in ["marginal", "joint"]:
        expected = tf.inverse_transform(
            transformed, missingness_restore=mode, random_state=77
        )
        actual = loaded.inverse_transform(
            transformed, missingness_restore=mode, random_state=77
        )
        pd.testing.assert_frame_equal(actual, expected)


def test_old_saved_state_fails_with_clear_version_error(tmp_path):
    state_file = tmp_path / "ifcfill-state-v1.json"
    state_file.write_text(json.dumps({"state_version": 1}), encoding="utf-8")

    with pytest.raises(ValueError, match="expected 2.*Refit"):
        IFCTransformer.load(state_file)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        IFCTransformer().transform(pd.DataFrame({"a": [1, 2]}))


def test_inverse_transform_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        IFCTransformer().inverse_transform(pd.DataFrame({"a": [1, 2]}))


def test_invalid_datetime_unit():
    with pytest.raises(ValueError, match="datetime_unit"):
        IFCTransformer(datetime_unit="years")  # type: ignore[arg-type]


def test_invalid_cat_encoding():
    with pytest.raises(ValueError, match="cat_encoding"):
        IFCTransformer(cat_encoding="target")  # type: ignore[arg-type]
