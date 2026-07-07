# ifcfill

**ifcfill** is a Python library for transforming tabular data into **I**nteger, **F**loat, and **C**ategorical (IFC) variables, with fast NumPy-powered missing data imputation and full transformation tracking.

[![PyPI version](https://img.shields.io/pypi/v/ifcfill)](https://pypi.org/project/ifcfill/)
[![Python versions](https://img.shields.io/pypi/pyversions/ifcfill)](https://pypi.org/project/ifcfill/)
[![License](https://img.shields.io/pypi/l/ifcfill)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue)](https://eulerlettersai.github.io/ifcfill)

---

## Features

- **Flexible input** — accepts a `pandas.DataFrame` or a path to a CSV file
- **Automatic type inference** — detects integer, float, categorical, and datetime columns automatically; user overrides supported per column
- **Configurable imputation** — choose fill strategy independently for each type:
  - Integer: `mean`, `median`, `mode`, `zero`
  - Float: `mean`, `median`, `mode`, `zero`
  - Categorical: `mode`, `constant`
- **Optional categorical label encoding** — encode categories as integer codes while keeping enough mapping metadata for `inverse_transform()`
- **Datetime → integer conversion** — converts date/time columns to integers relative to a configurable anchor date and time unit (days, seconds, ms, …)
- **Constant column removal** — automatically detects and drops columns with a single unique value
- **Missing value tracking** — records the count and fraction of missing values per column at fit time, accessible via `missing_report_`
- **Full transformation bookkeeping** — `inverse_transform()` restores dropped constants, original column order, and optionally re-introduces missing values at the original rate

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
    cat_fill="mode",
    cat_encoding="label",
    datetime_anchor="1970-01-01",
    datetime_unit="D",
)

transformed = tf.fit_transform(df)
print(transformed)

# Inspect missing-value distribution captured at fit time
print(tf.missing_report_)

# Restore original structure (constants + column order + optional missing values)
restored = tf.inverse_transform(transformed, restore_missing=True, random_state=42)
print(restored)
```

### From a CSV file

```python
transformed = IFCTransformer().fit_transform("data.csv")
```

### Override column types

```python
tf = IFCTransformer(col_types={"age": "categorical", "score": "float"})
transformed = tf.fit_transform(df)
```

---

## Documentation

Full documentation including the API reference is available at:
**<https://eulerlettersai.github.io/ifcfill>**

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
