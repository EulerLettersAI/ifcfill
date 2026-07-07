# Examples

The repository includes two notebooks that walk through the main `ifcfill`
workflows with small pandas DataFrames:

- [Basic usage without categorical encoding](https://github.com/EulerLettersAI/ifcfill/blob/main/examples/ifcfill_basic_usage_without_encoding.ipynb)
- [Label encoding for synthetic data](https://github.com/EulerLettersAI/ifcfill/blob/main/examples/ifcfill_label_encoding_for_synthetic_data.ipynb)

The notebooks cover:

- Using `ifcfill` around tabular synthetic-data workflows without a target variable
- Creating sample tabular data with numeric, categorical, datetime, missing, and constant columns
- Running `IFCTransformer.fit_transform`
- Keeping categoricals unencoded with the default `cat_encoding="none"`
- Label encoding categorical variables with inverse-compatible mappings using `cat_encoding="label"`
- Inspecting inferred column types, fill values, dropped constants, and `missing_report_`
- Restoring the original table structure with `inverse_transform`

To run the notebooks:

```bash
python -m venv .venv
source .venv/bin/activate
pip install "ifcfill[examples]"
jupyter notebook
```

The base install stays lean:

```bash
pip install ifcfill
```

Use the `examples` extra only when you want the notebook dependencies.
