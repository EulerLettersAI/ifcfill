import io
import textwrap

import numpy as np
import pandas as pd
import pytest

from ifcfill import IFCTransformer


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [25, 30, np.nan, 40],
            "salary": [50000.5, np.nan, 75000.0, 90000.25],
            "city": ["London", None, "Paris", "London"],
            "score": [1, 2, np.nan, 4],  # float column that is whole numbers → integer
        }
    )


def test_fit_transform_returns_dataframe(sample_df):
    result = IFCTransformer().fit_transform(sample_df)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == sample_df.shape


def test_no_missing_values_after_transform(sample_df):
    result = IFCTransformer().fit_transform(sample_df)
    assert result.isnull().sum().sum() == 0


def test_integer_column_dtype(sample_df):
    result = IFCTransformer().fit_transform(sample_df)
    assert result["age"].dtype == np.int64
    assert result["score"].dtype == np.int64


def test_float_column_dtype(sample_df):
    result = IFCTransformer().fit_transform(sample_df)
    assert result["salary"].dtype == np.float64


def test_categorical_column_type(sample_df):
    result = IFCTransformer().fit_transform(sample_df)
    assert hasattr(result["city"], "cat")


def test_cat_fill_constant(sample_df):
    tf = IFCTransformer(cat_fill="constant", cat_constant="unknown")
    result = tf.fit_transform(sample_df)
    assert "unknown" in result["city"].values


def test_float_fill_median(sample_df):
    tf = IFCTransformer(float_fill="median")
    result = tf.fit_transform(sample_df)
    expected_median = np.nanmedian(sample_df["salary"].to_numpy())
    assert result["salary"].notna().all()
    # The originally-missing entry should equal the median
    assert result["salary"].iloc[1] == pytest.approx(expected_median)


def test_transform_before_fit_raises():
    tf = IFCTransformer()
    with pytest.raises(RuntimeError, match="fit()"):
        tf.transform(pd.DataFrame({"a": [1, 2]}))


def test_invalid_input_type_raises():
    tf = IFCTransformer()
    with pytest.raises(TypeError):
        tf.fit(12345)  # type: ignore[arg-type]


def test_csv_input(tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("a,b,c\n1,,x\n2,3.5,y\n,4.0,\n")
    result = IFCTransformer().fit_transform(str(csv_file))
    assert isinstance(result, pd.DataFrame)
    assert result.isnull().sum().sum() == 0


def test_missing_csv_raises():
    tf = IFCTransformer()
    with pytest.raises(FileNotFoundError):
        tf.fit("/nonexistent/path/data.csv")
