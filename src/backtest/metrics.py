"""Performance metrics.

Evaluate the strategy on excess return over the risk-free rate, net of trading
costs and overall market beta, plus maximum drawdown, Sharpe ratio and
information ratio.

The engine's return series is already in **excess-of-RF** space (the pipeline's
``excess_return`` target) and **quarterly** (one row per rebalance period), so:

* ``risk_free`` is normally ``None`` here -- the subtraction already happened at
  the daily level;
* ``backtest.hedge_broad_market_beta: true`` means the broad-market beta is
  assumed hedged externally at no cost. That is an **accounting adjustment
  only** (per the strategy decision, it never constrains the optimizer): pass
  the engine's per-period ex-ante ``mkt_beta`` and the quarterly-compounded
  market return to :func:`excess_return`;
* annualization defaults to ``periods_per_year=252`` (daily convention, per the
  original stub) -- **callers evaluating the engine's quarterly series must pass
  ``periods_per_year=4``**.

All functions take anything ``np.asarray`` accepts (lists, numpy, polars Series).
"""

from __future__ import annotations

import numpy as np


def excess_return(returns, risk_free=None, market_beta=None, market_returns=None):
    """Return net of the risk-free rate and (optionally) the hedged market component.

    ``r_hedged_t = r_t - rf_t - beta_t * r_mkt_t``: ``beta_t`` is the portfolio's
    ex-ante market loading (``sum_i w_it * MKT_i``, reported by the engine) and
    ``r_mkt`` the market's excess return over the same periods -- the costless
    external hedge that ``backtest.hedge_broad_market_beta`` asserts. Pass
    ``market_beta`` and ``market_returns`` together or not at all.
    """
    r = np.asarray(returns, dtype=float).copy()
    if risk_free is not None:
        r = r - np.asarray(risk_free, dtype=float)
    if (market_beta is None) != (market_returns is None):
        raise ValueError("pass market_beta and market_returns together.")
    if market_beta is not None:
        r = r - np.asarray(market_beta, dtype=float) * np.asarray(
            market_returns, dtype=float
        )
    return r


def max_drawdown(returns):
    """Maximum peak-to-trough drawdown of the compounded return series (<= 0)."""
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return float("nan")
    curve = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(curve)
    return float((curve / peak - 1.0).min())


def sharpe_ratio(returns, risk_free=None, periods_per_year=252):
    """Annualized Sharpe ratio: ``mean / std * sqrt(periods_per_year)``.

    The engine's series is quarterly and already excess-of-RF: pass
    ``periods_per_year=4`` and leave ``risk_free=None``. Degenerate series
    (fewer than 2 points or zero dispersion) return NaN.
    """
    r = np.asarray(returns, dtype=float)
    if risk_free is not None:
        r = r - np.asarray(risk_free, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    std = r.std(ddof=1)
    if std == 0:
        return float("nan")
    return float(r.mean() / std * np.sqrt(periods_per_year))


def information_ratio(returns, benchmark, periods_per_year=252):
    """Annualized information ratio of active returns vs a benchmark.

    ``mean(r - b) / std(r - b) * sqrt(periods_per_year)`` over aligned series
    (pass ``periods_per_year=4`` for the engine's quarterly output). The default
    benchmark in the tour is the cap-weighted tradeable-universe return.
    """
    active = np.asarray(returns, dtype=float) - np.asarray(benchmark, dtype=float)
    return sharpe_ratio(active, risk_free=None, periods_per_year=periods_per_year)
