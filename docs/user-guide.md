# User Guide

## Synthetic-data workflow

`ifcfill` is intended for tabular synthetic-data generation workflows. It learns
all transformation rules from real data, then uses those same learned rules to
map both transformed real data and generated synthetic data back to the original
table structure.

The typical workflow is:

1. Fit `IFCTransformer` on real data.
2. Transform the real data into IFC variables.
3. Use any tabular generator to produce synthetic data in the transformed IFC
   space.
4. Call `inverse_transform()` on the synthetic output using the same fitted
   transformer, or save and load the fitted state when this happens elsewhere.

```python
tf = IFCTransformer(cat_encoding="label")

real_ifc = tf.fit_transform(real_df)

# Train or run your synthetic-data generator on real_ifc.
# synthetic_ifc = generator.sample(...)

synthetic_df = tf.inverse_transform(synthetic_ifc)
```

For a separate inverse-transform environment:

```python
tf.save("ifcfill-state.json")

# Later, or on another machine:
loaded_tf = IFCTransformer.load("ifcfill-state.json")
synthetic_df = loaded_tf.inverse_transform(synthetic_ifc)
```

`ifcfill` is intentionally unsupervised. It does not require a target variable,
and it avoids target-aware encodings so the same transformation contract can be
used across different tabular generators.

---

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
| `"constant"` *(default)* | The string supplied in `cat_constant` (default `"__ifcfill_missing__"`) |
| `"mode"` | Most frequent non-null value |

The default categorical strategy treats missing categorical values as their own
learnable category. If a generator later emits that category, `inverse_transform()`
converts it back to a missing value.

The default sentinel is deliberately namespaced as `"__ifcfill_missing__"` to
reduce collisions with real categories such as `"missing"` or `"unknown"`.

```python
tf = IFCTransformer(
    int_fill="mean",
    float_fill="median",
    cat_fill="constant",
    cat_constant="UNKNOWN",
)
```

---

## Parallel processing

`IFCTransformer` can process columns concurrently during `fit()` and
`transform()`:

```python
tf = IFCTransformer(n_jobs=-1)  # use all available CPUs
transformed = tf.fit_transform(df)
```

Use `n_jobs=1` or `n_jobs=None` for sequential execution. Negative values follow
the joblib convention: `-1` uses all available CPUs, `-2` leaves one CPU free,
and so on.

---

## Categorical encoding

By default, categorical columns are returned with pandas `object` dtype after
missing values are filled. This is broadly compatible with synthesizers that
reject pandas categorical dtypes. Missing categorical values are represented
by the learned missing category:

```python
tf = IFCTransformer(cat_encoding="none")  # default
```

To opt into pandas categorical output:

```python
tf = IFCTransformer(cat_output="category")
```

`inverse_transform()` accepts unencoded categorical columns with either object
or pandas categorical dtype, regardless of which dtype a synthesizer returns.

Encodings are designed to be unsupervised and invertible. Categorical missing
values are filled first, and label encoding is applied as a second layer over
the completed set of categories. This means the learned missing category is part
of the mapping even when it did not appear in the fit data.

For generative modeling workflows, categorical columns can also be label
encoded into integer codes:

```python
tf = IFCTransformer(cat_encoding="label")
transformed = tf.fit_transform(df)
```

The learned mappings are stored after `fit()`:

```python
tf.category_mappings_          # dict: column -> {category -> code}
tf.inverse_category_mappings_  # dict: column -> {code -> category}
tf.get_category_mappings()     # defensive copy of all forward mappings
tf.get_category_mapping("city") # defensive copy for one column
```

`inverse_transform()` decodes label-encoded columns back to their original
category values. This is designed for unsupervised synthetic-data workflows:
generated numeric category codes are rounded to the nearest integer and clipped
to the known code range before decoding. If the decoded value is the learned
categorical missing category, it is converted back to a missing value.

When new or unseen categories appear during `transform()`, they fall back to the
learned categorical fill value before encoding.

!!! note
    `ifcfill` intentionally does not include target-aware encodings. Encodings
    should stay unsupervised and invertible so synthetic data can be mapped back
    to meaningful table values.

---

## Saving fitted transformations

For workflows where inverse transformation happens later or on another machine,
save the fitted transformer state after `fit()` or `fit_transform()`:

```python
tf = IFCTransformer(cat_encoding="label")
real_ifc = tf.fit_transform(real_df)
tf.save("ifcfill-state.json")
```

Then load the same transformation state wherever generated data needs to be
restored:

```python
loaded_tf = IFCTransformer.load("ifcfill-state.json")
synthetic_df = loaded_tf.inverse_transform(synthetic_ifc)
```

The JSON state includes the learned column types, fill values, dropped
constants, missing-value statistics, categorical label mappings, datetime
settings, and original column order.

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

Categorical columns with one observed category and missing values are preserved
when `cat_fill="constant"` because the missing value becomes a second learned
category.

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
2. **Decodes** label-encoded categorical integer codes when `cat_encoding="label"`
3. **Converts** learned categorical missing categories back to missing values
4. **Reorders** columns to match the original input order
5. **Optionally re-introduces** non-categorical missing values at the same rates as the original data

```python
restored = tf.inverse_transform(
    transformed_df,
    restore_missing=True,  # statistical restoration for imputed non-categoricals
    random_state=0,        # make it reproducible
)
```

!!! note
    For categorical columns with `cat_fill="constant"`, missingness is restored
    deterministically when the missing category appears. For numeric and datetime
    columns, `restore_missing=True` statistically reintroduces missing values at
    the rates learned during `fit()`.

---

## Building the docs locally

```bash
pip install "ifcfill[docs]"
mkdocs serve          # live-reload preview at http://127.0.0.1:8000
mkdocs build          # static site in site/
mkdocs gh-deploy      # publish to GitHub Pages
```
