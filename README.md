# ifcfill

**ifcfill** is a Python library for preparing tabular data for synthetic-data
generation. It transforms real tabular data into **I**nteger, **F**loat, and
**C**ategorical (IFC) variables with fast NumPy-powered missing data imputation,
then applies the same learned inverse transformation to generated synthetic data
so it can be mapped back to the original table structure.

[![PyPI version](https://img.shields.io/pypi/v/ifcfill)](https://pypi.org/project/ifcfill/)
[![Python versions](https://img.shields.io/pypi/pyversions/ifcfill)](https://pypi.org/project/ifcfill/)
[![License](https://img.shields.io/pypi/l/ifcfill)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue)](https://eulerlettersai.github.io/ifcfill)

---

## Purpose

`ifcfill` is designed to sit around tabular synthetic-data generators:

1. Fit `IFCTransformer` on real data.
2. Transform the real data into generator-friendly IFC variables.
3. Train or run any tabular generator on the transformed data.
4. Apply `inverse_transform()` to the synthetic output using the mappings learned
   from the real data.

If inverse transformation happens later or on another machine, save the fitted
transformation state with `save()` and load it back with `IFCTransformer.load()`.

The package is intentionally unsupervised: it does not require or model a target
variable.

---

## Features

- **Flexible input** — accepts a `pandas.DataFrame` or a path to a CSV file
- **Automatic type inference** — detects integer, float, categorical, and datetime columns automatically; user overrides supported per column
- **Configurable imputation** — choose fill strategy independently for each type:
  - Integer: `mean`, `median`, `mode`, `zero`
  - Float: `mean`, `median`, `mode`, `zero`
  - Categorical: `constant` (default), `mode`
- **Categorical missingness as a category** — missing categorical values are transformed into a learnable category and converted back to missing values during `inverse_transform()`
- **Namespaced missing sentinel** — the default categorical missing category is `__ifcfill_missing__` to reduce collisions with real values
- **Synthesizer-friendly categorical output** — unencoded categoricals are object columns by default, with pandas `category` available via `cat_output="category"`
- **Optional categorical label encoding** — fill categorical values first, then encode categories as integer codes through a separate label-encoding layer with inverse mappings
- **Datetime → integer conversion** — converts date/time columns to integers relative to a configurable anchor date and time unit (days, seconds, ms, …)
- **Constant column removal** — automatically drops true constant columns while preserving categorical missing categories when they are learnable
- **Missing value tracking** — records the count and fraction of missing values per column at fit time, accessible via `missing_report_`
- **Full transformation bookkeeping** — `inverse_transform()` restores dropped constants, original column order, and optionally re-introduces missing values at the original rate
- **Portable fitted state** — save all learned transformations to JSON and load them later for consistent transform/inverse-transform workflows on another machine
- **Synthetic-data workflow support** — apply one inverse transformation consistently to both transformed real data and generated synthetic data

---

## Installation

```bash
pip install ifcfill
```

---

## Quick Start

```python
import pandas as pd
from ifcfill import IFCTransformer

df = pd.DataFrame({
    "age":    [25, 30, None, 40],
    "salary": [50_000.5, None, 75_000.0, 90_000.25],
    "city":   ["London", None, "Paris", "London"],
    "joined": pd.to_datetime(["2020-01-01", "2021-06-15", None, "2023-03-10"]),
    "flag":   ["yes", "yes", "yes", "yes"],   # constant → will be dropped
})

tf = IFCTransformer(
    int_fill="median",
    float_fill="mean",
    datetime_anchor="1970-01-01",
    datetime_unit="D",
)

transformed = tf.fit_transform(df)
print(transformed)

# Inspect missing-value distribution captured at fit time
print(tf.missing_report_)

# Restore original structure.
# Categorical missing categories are converted back to missing values.
restored = tf.inverse_transform(transformed, restore_missing=True, random_state=42)
print(restored)

# Save everything needed to transform or inverse-transform later.
tf.save("ifcfill-state.json")

# On another machine or in another process:
loaded_tf = IFCTransformer.load("ifcfill-state.json")
restored_again = loaded_tf.inverse_transform(transformed)
```

### From a CSV file

```python
transformed = IFCTransformer().fit_transform("data.csv")
```

### Parallel column processing

```python
# Use all available CPUs for per-column fit/transform work
tf = IFCTransformer(n_jobs=-1)
transformed = tf.fit_transform(df)
```

### Override column types

```python
tf = IFCTransformer(col_types={"age": "categorical", "score": "float"})
transformed = tf.fit_transform(df)
```

### Label encoding for generators

```python
tf = IFCTransformer(cat_encoding="label")
transformed = tf.fit_transform(df)

# Safe copies of learned category/code dictionaries
print(tf.get_category_mappings())

restored = tf.inverse_transform(transformed)
```

To return pandas categorical columns instead of the default object columns:

```python
tf = IFCTransformer(cat_output="category")
```

---

## Documentation

Full documentation including the API reference is available at:
**<https://eulerlettersai.github.io/ifcfill>**

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
