# Notebooks

Exploratory analysis, factor diagnostics, and result plots (plotnine / altair)
live here, kept out of the `src/` library code. Notebooks should import from
`src` and treat it as the source of truth — keep reusable logic in modules, not
in cells.

Suggested:
- `01_eda.ipynb` — raw data sanity checks, coverage, accounting-break inspection
- `02_factor_diagnostics.ipynb` — IC / IR / quantile plots per factor
- `03_portfolio_results.ipynb` — backtest equity curve, drawdown, attribution
