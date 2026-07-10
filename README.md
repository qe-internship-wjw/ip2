# Financials Quantitative Equity Strategy Pipeline

This repository is the codebase accompanying Individal Project 2 of WJW supervised by NIN. It features a robust out-of-sample, walk-forward backtesting engine with point-in-time (PIT) data constraints, dynamic factor selection, risk modeling, and a mean-variance optimizer (MVO) with L1 transaction cost penalties.

## Setup Guide

To get the pipeline running locally, follow these steps:

1. Set up conda environment by running `conda env create -f environment.yaml` followed by `conda activate ip2`.
2. Place data in a folder called `data/raw`.
3. Run `yield_curve_did.ipynb` and `build_processed.py`, which takes a few minutes to create data artifacts in the folder `data/processed`.
4. Run `run_backtest.py`, which takes about a minute.
5. You can now explore the results in `backtest_tour.ipynb` and `pipeline_tour.ipynb`!

---

## Repository Structure

The repository is modularly separated into executable scripts, exploratory/diagnostic notebooks, and the core library (`src/`).

### `src/` (Core Library)

The `src` directory contains the highly decoupled pipeline architecture. It is broken down into the following subpackages:

* **`src/data/`**: Data ingestion and processing.
    * Handles loading raw data, performing point-in-time (PIT) joins (`joins.py`), handling survivorship bias via delisting tracking (`delisting.py`), and factor preprocessing including winsorization, imputation, and standardization (`preprocess.py`).


* **`src/factors/`**: Declarative factor generation registry.
    * **`style/`**: Alpha signals grouped by sub-universe: `banks` (e.g., NIM, NPL Coverage, TBV/P), `insurance` (e.g., FIY, PI/P, Underwriting Margin), and `all_financials` (e.g., Size, E/P, Vol-Adjusted Momentum). Also includes `yield_curve` sensitivity factors and `structural_beta` signals.
    * **`nonstyle/`**: Structural risk factors (Market, Country, Industry) built via hierarchical factor-mimicking portfolios, and base Nelson-Siegel sovereign yield curve modeling.


* **`src/validation/`**: Factor evaluation and selection.
    * **`neutralization.py`**: Orthogonalizes style factors against structural non-style risk designs.
    * **`single_factor.py`**: Evaluates individual factors using Rank IC, IC decay, Quantile Returns, and pooled Fama-MacBeth regressions (with Newey-West HAC t-stats).
    * **`redundancy.py`**: Implements correlation clustering and LassoCV for parsimony.
    * **`selection.py`**: The 3-step point-in-time factor selection protocol (Single-factor gate → Redundancy clustering → Parsimony lasso).


* **`src/portfolio/`**: Portfolio construction algorithms.
    * **`expected_returns.py`**: Integrates systematic and behavioral factors, utilizing walk-forward expanding windows and James-Stein shrinkage.
    * **`risk_model.py`**: A structured covariance risk model using Ledoit-Wolf shrinkage and EMA smoothing across rebalances.
    * **`optimizer.py`**: Mean-variance optimization with constraints (gross leverage, net exposure) and an L1 transaction cost penalty to create the optimal no-trade region.
    * **`transaction_cost.py`**: Market-cap (free-float) dependent transaction cost models.


* **`src/backtest/`**: The walk-forward engine.
    * **`engine.py`**: The core step-by-step rebalance loop executing risk modeling, expected return generation, optimization, and drift tracking.
    * **`dynamic.py` & `schedule.py`**: Managers for the regime schedules, threading portfolio state and selected feature sets smoothly across dynamic re-selection boundaries.
    * **`metrics.py`**: P&L accounting, computing Sharpe ratios, Max Drawdown, Information Ratios, and factor selection stability metrics.



### `notebooks/` (Exploration & Diagnostics)

Notebooks are intended for exploratory data analysis (EDA), diagnostics, and visualization. They import from `src/` to ensure a single source of truth.

* **`pipeline_tour.ipynb`**: An end-to-end guided tour of the factor pipeline (ingestion → factors → neutralization → selection). Generates validation visualizations like IC-IR bars, Fama-MacBeth premia, and forward-return quantile ramps.
* **`backtest_tour.ipynb`**: Visualization layer over the dynamic walk-forward backtest outputs. Displays equity curves (net and hedged), drawdowns, book diagnostics (leverage, turnover), and factor selection stability metrics over time.
* **`ablation_tour.ipynb`**: Analysis of portfolio-construction hyperparameter sweeps (e.g., allocation methods, James-Stein shrinkage, risk aversion, and leverage caps), showcasing which levers most impact the hedged net Sharpe. **Note:** This requires a run of `run_ablation.py` which can take one to two hours.
* **`accounting_volatility_analysis.ipynb`**: Evaluates the impact of accounting standard transitions (IFRS 9 / CECL for banks; IFRS 17 / LDTI for insurers) on the fundamental volatility of the underlying assets.
* **`yield_curve_did.ipynb`**: A Difference-in-Differences (DiD) event study measuring whether the IBOR to RFR transitions shifted sovereign yield curve parameters (level/slope).
* **`yield_curve_ns_fit.ipynb`**: Visual spot-checker for the Nelson-Siegel sovereign yield curve fits applied to the raw tenor data.

### `scripts/` (Executable Drivers)

Command-line entry points for executing heavy compute tasks. They populate the `data/processed/` cache.

* **`build_processed.py`**: Precomputes downsampled artifacts (factor scores, loadings, delist events, neutralized frames).
    * Average run time: 3-5 minutes
* **`select_features.py`**: Performs point-in-time factor selection across dynamic regime cutoffs.
    * Average run time: 20 seconds
* **`run_backtest.py`**: The main driver for the end-to-end walk-forward backtest.
    * Average run time: 1 minute
* **`run_ablation.py`**: Driver for the comprehensive grid search over portfolio optimizer hyperparameters.
    * Average run time: 1-2 hours
* **`check_data_quality.py`**: Emits a data quality report checking for duplicates, stale price imputation flags, and null coverage rates across the dataset.
    * Average run time: 3-5 minutes