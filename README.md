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
- **Synthesis-safe categorical missingness** — missing categorical values use the internal `__ifcfill_missing__` sentinel during transformation
- **Missingness flags** — binary `<column>__ifc_missing` flags for integer, float, and datetime columns that contain nulls, so generators learn missingness jointly with the data
- **Four reconstruction modes** — generator-learned (default), hybrid (exact rates placed by a conditional missingness model), marginal, or empirical joint row patterns
- **Optional categorical label encoding** — fill categorical values first, then encode categories as integer codes through a separate label-encoding layer with inverse mappings
- **Datetime → integer conversion** — converts date/time columns to integers relative to a configurable anchor date and time unit (days, seconds, ms, …)
- **Constant column removal** — automatically drops true constant columns while preserving categorical missing categories when they are learnable
- **Missing value tracking** — records the count and fraction of missing values per column at fit time, accessible via `missing_report_`
- **Full transformation bookkeeping** — `inverse_transform()` restores semantic dtypes, dropped constants, original column order, and fitted missingness
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

# Restore original structure, using the missingness produced by the generator.
restored = tf.inverse_transform(
    transformed,
    missingness_restore="generated",
    random_state=42,
)
print(restored)

# Save everything needed to transform or inverse-transform later.
tf.save("ifcfill-state.json")

# On another machine or in another process:
loaded_tf = IFCTransformer.load("ifcfill-state.json")
restored_again = loaded_tf.inverse_transform(transformed)
```

## Missingness reconstruction

`inverse_transform()` always reconstructs missingness across every original
column, including categorical and internally dropped constant columns:

- **IFCFill-Generated** (`missingness_restore="generated"`, the default) keeps
  the missingness produced by the generator. `transform()` adds a binary
  `<column>__ifc_missing` flag for each integer, float, or datetime column that
  had missing values at fit time, and categorical missingness is carried by the
  sentinel category. The generator therefore learns missingness jointly with
  the other variables, and the final mask is read from the generated flags and
  sentinels. Missingness rates follow the generator and are not forced to match
  the real data exactly.
- **IFCFill-Hybrid** (`missingness_restore="hybrid"`) assigns exactly
  `round(fitted_fraction * n)` missing positions per column, like Marginal, but
  chooses the rows by weighted sampling without replacement from a
  regularized logistic model of `P(M_j | other values, other indicators)` fitted
  on the real data and evaluated on the synthetic rows. It preserves exact rates
  and the dependence of missingness on observed values.
- **IFCFill-Marginal** (`missingness_restore="marginal"`) preserves
  each fitted per-variable distribution, `P(M_j)`. For `n` synthetic rows it
  assigns exactly `round(fitted_fraction * n)` missing positions per column,
  independently of the values.
- **IFCFill-Joint** (`missingness_restore="joint"`) preserves the fitted
  empirical row-wise pattern distribution, `P(M_1, ..., M_p)`, using
  largest-remainder count allocation followed by a reproducible row shuffle.

Keep the flag columns in the data passed to the generator; the generated and
hybrid modes require them. Set `missing_flags=False` to omit them, in which
case use the marginal or joint mode.

The categorical sentinel `__ifcfill_missing__` only makes incomplete training
data synthesis-ready. It does not control final missingness on its own. IFCFill replaces
excess generated sentinels and applies its reconstructed mask as the authority.
Integers are rounded without clipping and returned as nullable pandas `Int64`;
floats remain floating-point and missing datetimes use `NaT`.

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

---

## Documentation

Full documentation including the API reference is available at:
**<https://eulerlettersai.github.io/ifcfill>**

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
