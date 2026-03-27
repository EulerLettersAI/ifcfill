# User Guide

## Input formats

`IFCTransformer` accepts two input formats in `fit()`, `transform()`, and `fit_transform()`:

| Type | Example |
|---|---|
| `pandas.DataFrame` | `tf.fit_transform(df)` |
| CSV file path (str or `pathlib.Path`) | `tf.fit_transform("data.csv")` |

---

## Type inference

When no `col_types` override is provided, the package infers each column's type in
this order:

1. **`datetime`** — native `datetime64` dtype, or object column where ≥ 80 % of
   non-null values parse as dates
2. **`categorical`** — boolean dtype
3. **`integer`** — integer dtype, or float column whose non-null values are all whole
   numbers, or object column that coerces numerically with whole numbers (≥ 90 %
   success rate)
4. **`float`** — float dtype, or object column that coerces numerically (≥ 90 %
   success rate) but is not all whole numbers
5. **`categorical`** — everything else

### Override types manually

Use the `col_types` constructor argument to force a type for specific columns.
Other columns are still inferred automatically.

```python
tf = IFCTransformer(
    col_types={
        "zip_code": "categorical",  # prevent numeric inference
        "score":    "float",        # keep decimals even if all-whole
    }
)
```

---

## Imputation strategies

### Integer columns

| Strategy | Fill value |
|---|---|
| `"median"` *(default)* | `int(round(median))` |
| `"mean"` | `int(round(mean))` |
| `"mode"` | Most frequent value |
| `"zero"` | `0` |

### Float columns

| Strategy | Fill value |
|---|---|
| `"mean"` *(default)* | Arithmetic mean |
| `"median"` | Median |
| `"mode"` | Most frequent value |
| `"zero"` | `0.0` |

### Categorical columns

| Strategy | Fill value |
|---|---|
| `"mode"` *(default)* | Most frequent non-null value |
| `"constant"` | The string supplied in `cat_constant` (default `"missing"`) |

```python
tf = IFCTransformer(
    int_fill="mean",
    float_fill="median",
    cat_fill="constant",
    cat_constant="UNKNOWN",
)
```

---

## Datetime conversion

Datetime columns are converted to integers representing elapsed time from an anchor
date. Two parameters control this:

| Parameter | Default | Description |
|---|---|---|
| `datetime_anchor` | `"1970-01-01"` | Reference point; any date string or `pd.Timestamp` |
| `datetime_unit` | `"D"` (days) | Unit: `"D"`, `"s"`, `"ms"`, `"us"`, `"ns"` |

```python
# Days since 2000-01-01
tf = IFCTransformer(datetime_anchor="2000-01-01", datetime_unit="D")

# Seconds since Unix epoch
tf = IFCTransformer(datetime_anchor="1970-01-01", datetime_unit="s")
```

Missing datetime values (`NaT`) are filled with the median of the non-null values
in that column.

---

## Constant column removal

Any column with **≤ 1 unique non-null value** is automatically dropped from the
output of `transform()`. This includes:

- Columns where every row has the same value
- Columns that are entirely `NaN`/`None`

Dropped columns and their values are stored in `dropped_constants_`:

```python
tf.fit(df)
print(tf.dropped_constants_)
# {"flag": ("yes", 4)}  ← value and original column position
```

---

## Missing value tracking

After `fit()`, three attributes and one property describe the missing-data
distribution of the **original** (pre-transform) data:

```python
tf.missing_counts_     # dict: column → number of missing values
tf.missing_fractions_  # dict: column → fraction of missing values
tf.missing_report_     # pd.DataFrame with columns: column, type, missing_count, missing_fraction
```

Example report:

```
    column        type  missing_count  missing_fraction
0      age     integer              1          0.250000
1   salary       float              1          0.250000
2     city categorical              1          0.250000
3   joined    datetime              1          0.250000
4     flag    constant              0          0.000000
```

!!! tip
    Constant columns appear in the report with `type = "constant"` even though they
    are dropped from the transformed output.

---

## Inverse transform

`inverse_transform()` reverses the structural changes made by `transform()`:

1. **Re-inserts** dropped constant columns with their original values
2. **Reorders** columns to match the original input order
3. **Optionally re-introduces** missing values at the same rates as the original data

```python
restored = tf.inverse_transform(
    transformed_df,
    restore_missing=True,  # re-introduce NaN proportionally
    random_state=0,        # make it reproducible
)
```

!!! note
    `inverse_transform()` does **not** reverse numeric casting or imputation — it
    restores structure (column presence and order), not exact original values.

---

## Building the docs locally

```bash
pip install "ifcfill[docs]"
mkdocs serve          # live-reload preview at http://127.0.0.1:8000
mkdocs build          # static site in site/
mkdocs gh-deploy      # publish to GitHub Pages
```
