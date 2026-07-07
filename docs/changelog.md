# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- Optional categorical label encoding via `cat_encoding="label"`.
- Category mapping attributes for inverse-compatible label encoding:
  `category_mappings_` and `inverse_category_mappings_`.
- Categorical missing values are represented as a learnable category by default
  and converted back to missing values during `inverse_transform()`.

---

## [0.1.0] — 2026-03-27

### Added

- `IFCTransformer` class with `fit()`, `transform()`, `fit_transform()`, and `inverse_transform()`.
- Automatic type inference for integer, float, categorical, and datetime columns.
- Per-column type overrides via `col_types` constructor argument.
- Configurable imputation strategies:
  - Integer: `mean`, `median`, `mode`, `zero`
  - Float: `mean`, `median`, `mode`, `zero`
  - Categorical: `mode`, `constant`
- Datetime-to-integer conversion with configurable anchor date and time unit.
- Automatic detection and removal of constant columns.
- Missing value distribution tracking via `missing_counts_`, `missing_fractions_`, and `missing_report_`.
- `inverse_transform()` with optional missing-value restoration and reproducible `random_state`.
- Accepts `pandas.DataFrame` and CSV file paths as input.
- NumPy-based computation engine for fast processing.
