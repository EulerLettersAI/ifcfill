# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.3.2] — 2026-07-09

### Added

- Parallel per-column processing via the `n_jobs` constructor argument for
  `fit()` and `transform()`.
- Support for joblib-style negative `n_jobs` values, including `n_jobs=-1` to
  use all available CPUs.

---

## [0.3.1] — 2026-07-08

### Fixed

- `inverse_transform()` now converts datetime columns from their integer offset
  representation back to pandas timestamps.
- Datetime missing-value restoration now uses `NaT`, preserving datetime dtype
  when `restore_missing=True`.

---

## [0.3.0] — 2026-07-07

### Added

- Category mapping accessors:
  `get_category_mappings()` and `get_category_mapping(...)`.
- Portable fitted-state persistence with `save()` and `IFCTransformer.load()`
  for transform/inverse-transform workflows on another machine.

### Changed

- The default categorical missing category is now `__ifcfill_missing__` to
  reduce collisions with real categories.
- Label encoding now runs as a separate layer after categorical filling so
  learned fill categories are included in the mapping.
- Categorical columns with one observed category plus missing values are kept
  when `cat_fill="constant"` because the missing category is learnable.

---

## [0.2.0] — 2026-07-07

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
