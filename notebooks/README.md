# Notebooks

Exploratory analysis, factor diagnostics, and result plots (plotnine / altair)
live here, kept out of the `src/` library code. Notebooks should import from
`src` and treat it as the source of truth — keep reusable logic in modules, not
in cells.

Available:
- `pipeline_tour.ipynb` — end-to-end guided tour: ingestion → panel/universe → factor
  generation → regularize/neutralize → validation & selection (IC/IR, Fama-MacBeth,
  redundancy clustering, lasso) → final factor shortlist. Runs on a configurable
  country/date slice in well under a minute.
- `yield_curve_ns_fit.ipynb` — Nelson-Siegel curve-fit visual spot-check.
- `accounting_volatility_analysis.ipynb` — accounting-break / volatility inspection.

Suggested:
- `01_eda.ipynb` — raw data sanity checks, coverage, accounting-break inspection
- `02_factor_diagnostics.ipynb` — IC / IR / quantile plots per factor
- `03_portfolio_results.ipynb` — backtest equity curve, drawdown, attribution
