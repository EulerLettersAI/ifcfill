# Examples

The repository includes a notebook that walks through the main `ifcfill`
workflow with a small pandas DataFrame:

- [Basic ifcfill usage notebook](https://github.com/EulerLettersAI/ifcfill/blob/main/examples/ifcfill_basic_usage.ipynb)

The notebook covers:

- Creating sample tabular data with numeric, categorical, datetime, missing, and constant columns
- Running `IFCTransformer.fit_transform`
- Label encoding categorical variables with inverse-compatible mappings
- Inspecting inferred column types, fill values, dropped constants, and `missing_report_`
- Restoring the original table structure with `inverse_transform`
- Reusing a fitted transformer on new data

To run it from a local clone:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[examples]"
jupyter notebook examples/ifcfill_basic_usage.ipynb
```

The base install stays lean:

```bash
pip install ifcfill
```

Use the `examples` extra only when you want the notebook dependencies.
