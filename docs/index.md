# ifcfill

**ifcfill** prepares tabular data for synthetic-data generation. It transforms
real data into **I**nteger, **F**loat, and **C**ategorical (IFC) variables, fills
missing values using fast NumPy operations, and keeps the metadata needed to map
generated synthetic data back to the original table structure.

`ifcfill` is intentionally unsupervised. It does not require a target variable;
instead, it learns transformation and inverse-transformation rules from the real
data so the same inverse rules can be applied to synthetic outputs from any
tabular generator.

---

## Key Features

| Feature | Description |
|---|---|
| Flexible input | `pandas.DataFrame` or CSV file path |
| Automatic type inference | Detects integer, float, categorical, and datetime columns |
| Per-column type overrides | Force a specific type for any column |
| Configurable imputation | Independent strategy per type (mean, median, mode, zero, constant) |
| Categorical missingness | Treats missing categorical values as a learnable category and restores them on inverse transform |
| Categorical label encoding | Optionally encode filled categories as integer codes with a separate inverse-compatible encoder layer |
| Datetime conversion | Date/time → integer relative to a configurable anchor |
| Constant column removal | Drops true constants while preserving learnable categorical missing categories |
| Missing value tracking | Records count and fraction per column via `missing_report_` |
| Inverse transform | Restores constants, column order, categorical missing values, and optional non-categorical missing-value distribution |
| Portable fitted state | Save learned transformations to JSON and load them later on another machine |
| Generator support | Fit on real data, transform for a generator, inverse-transform generated synthetic data |

---

## Installation

```bash
pip install ifcfill
```

## Quick example

```python
from ifcfill import IFCTransformer

tf = IFCTransformer()
real_ifc = tf.fit_transform("real_data.csv")
print(tf.missing_report_)

tf.save("ifcfill-state.json")

# synthetic_ifc is produced by your tabular synthetic-data generator
loaded_tf = IFCTransformer.load("ifcfill-state.json")
synthetic_restored = loaded_tf.inverse_transform(synthetic_ifc)
```

---

[Get started →](getting-started.md){ .md-button .md-button--primary }
[Examples →](examples.md){ .md-button }
[API Reference →](api.md){ .md-button }
