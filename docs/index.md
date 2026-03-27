# ifcfill

**ifcfill** transforms tabular data into **I**nteger, **F**loat, and **C**ategorical (IFC)
variables, fills missing values using fast NumPy operations, and keeps a full audit trail
of every transformation so you can restore the original table structure at any time.

---

## Key Features

| Feature | Description |
|---|---|
| Flexible input | `pandas.DataFrame` or CSV file path |
| Automatic type inference | Detects integer, float, categorical, and datetime columns |
| Per-column type overrides | Force a specific type for any column |
| Configurable imputation | Independent strategy per type (mean, median, mode, zero, constant) |
| Datetime conversion | Date/time → integer relative to a configurable anchor |
| Constant column removal | Silently drops columns with a single unique value |
| Missing value tracking | Records count and fraction per column via `missing_report_` |
| Inverse transform | Restores constants, column order, and optional missing-value distribution |

---

## Installation

```bash
pip install ifcfill
```

## Quick example

```python
from ifcfill import IFCTransformer

tf = IFCTransformer()
transformed = tf.fit_transform("data.csv")
print(tf.missing_report_)

restored = tf.inverse_transform(transformed, restore_missing=True, random_state=0)
```

---

[Get started →](getting-started.md){ .md-button .md-button--primary }
[API Reference →](api.md){ .md-button }
