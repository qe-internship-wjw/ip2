"""Performance metrics.

Evaluate the strategy on excess return over the risk-free rate, net of trading
costs and overall market beta, plus maximum drawdown, Sharpe ratio and
information ratio.
"""

from __future__ import annotations


def excess_return(returns, risk_free, market_beta=None, market_returns=None):
    """Return net of risk-free rate and (optionally) overall market beta."""
    raise NotImplementedError


def max_drawdown(returns):
    """Maximum peak-to-trough drawdown of the cumulative return series."""
    raise NotImplementedError


def sharpe_ratio(returns, risk_free=None, periods_per_year=252):
    """Annualized Sharpe ratio."""
    raise NotImplementedError


def information_ratio(returns, benchmark):
    """Annualized information ratio of active returns vs a benchmark."""
    raise NotImplementedError
