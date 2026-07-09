# Getting Started

## Requirements

- Python 3.9–3.12
- `numpy >= 1.23`
- `pandas >= 1.5`

## Installation

Install from PyPI:

```bash
pip install ifcfill
```

To install the optional notebook dependencies for the examples:

```bash
pip install "ifcfill[examples]"
```

---

## Your First Transformation

### 1 — From a pandas DataFrame

```python
import numpy as np
import pandas as pd
from ifcfill import IFCTransformer

df = pd.DataFrame({
    "age":    [25, 30, None, 40],
    "salary": [50_000.5, None, 75_000.0, 90_000.25],
    "city":   ["London", None, "Paris", "London"],
    "joined": pd.to_datetime(["2020-01-01", "2021-06-15", None, "2023-03-10"]),
    "flag":   ["yes", "yes", "yes", "yes"],  # constant column
})

tf = IFCTransformer()
transformed = tf.fit_transform(df)
print(transformed)
```

Output:

```
   age       salary    city  joined
0   25   50000.500  London   18262
1   30   71666.917  __ifcfill_missing__  18793
2   30   75000.000   Paris   19063    ← NaT filled with median
3   40   90000.250  London   19426
```

!!! note
    The `flag` column is automatically dropped because it is constant.

---

### 2 — From a CSV file

```python
tf = IFCTransformer()
transformed = tf.fit_transform("path/to/data.csv")
```

---

### 3 — Fit once, transform many times

```python
tf = IFCTransformer()
tf.fit(train_df)

transformed_train = tf.transform(train_df)
transformed_test  = tf.transform(test_df)   # same rules applied
```

---

### 4 — Save the fitted transformation state

Save the fitted state when inverse transformation may happen later or on another
machine:

```python
tf.save("ifcfill-state.json")
```

Load it back without fitting again:

```python
loaded_tf = IFCTransformer.load("ifcfill-state.json")
restored = loaded_tf.inverse_transform(synthetic_ifc)
```

---

### 5 — Use label encoding for categorical generator inputs

```python
tf = IFCTransformer(cat_encoding="label")
transformed = tf.fit_transform(df)

print(tf.get_category_mappings())
```

Categorical values are filled first, then the label encoder maps the resulting
categories to integer codes. During inverse transformation, the integer codes are
decoded back to categories before missing categories are restored to missing
values.

---

### 6 — Check what happened

```python
# See types detected for each column
print(tf.column_types_)

# See fill values chosen for each column
print(tf.fill_values_)

# Full missing-value report
print(tf.missing_report_)
```

---

### 7 — Restore the original structure

```python
restored = tf.inverse_transform(
    transformed,
    restore_missing=True,   # statistical restoration for imputed non-categoricals
    random_state=42,        # reproducible
)
print(restored)
```

---

## Next Steps

- [Examples](examples.md) — notebook walkthrough of the main workflow
- [User Guide](user-guide.md) — detailed walkthrough of every feature
- [API Reference](api.md) — complete parameter and return-type documentation
