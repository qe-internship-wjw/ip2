"""Backtest engine: rebalance loop.

Walks forward at the configured rebalancing frequency (default quarterly),
rebuilding expected returns and the risk model with information available as of
each rebalance date, solving the optimizer, applying transaction costs on the
turnover, and accruing portfolio returns between rebalances.
"""

from __future__ import annotations


def run(panel, factors, cfg):
    """Run the walk-forward backtest; return the realized portfolio return series."""
    raise NotImplementedError
