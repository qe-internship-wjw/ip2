"""Single-factor evaluation.

1. Information Coefficient
   - Rank IC between t exposure and t+1 excess return (net of non-style factors).
   - IC decay at t+1, t+2, t+3 to inform rebalancing frequency / turnover.
   - Information ratio of the IC series; consistent IR > 0.3 shortlists a factor.
2. Quantile Portfolio
   - Sort universe into quintiles; equal- and cap-weighted forward returns.
   - Monotonicity check and Long-Short (Q1-Q5) returns, t-stats, Sharpe.
3. Fama-MacBeth
   - Period-by-period cross-sectional regressions of forward returns on scores.
   - Aggregate coefficients with Newey-West adjusted t-stats for autocorrelation.
"""

from __future__ import annotations


def rank_ic(scores, fwd_returns, lags=(1, 2, 3)):
    """Rank IC series and IC decay across the given forward lags."""
    raise NotImplementedError


def information_ratio(ic_series):
    """IR = mean(IC) / std(IC) of the IC time series."""
    raise NotImplementedError


def quantile_portfolios(scores, fwd_returns, n=5, weighting="equal"):
    """Quintile forward returns plus the long-short (Q1-Q5) series."""
    raise NotImplementedError


def long_short_stats(ls_returns):
    """Cumulative return, t-statistic and Sharpe ratio of a long-short series."""
    raise NotImplementedError


def fama_macbeth(scores, fwd_returns, newey_west_lags=None):
    """Aggregate cross-sectional regression premia with Newey-West t-stats."""
    raise NotImplementedError
