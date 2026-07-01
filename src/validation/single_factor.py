"""Single-factor evaluation.

1. Information Coefficient
   - Rank IC between t exposure and t+1 excess return (net of non-style factors).
   - IC decay at t+1, t+2, t+3 to inform rebalancing frequency / turnover.
   - Information ratio of the IC series; consistent IR > 0.3 shortlists a factor.
2. Fama-MacBeth
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

def fama_macbeth(scores, fwd_returns, newey_west_lags=None):
    """Aggregate cross-sectional regression premia with Newey-West t-stats."""
    raise NotImplementedError
