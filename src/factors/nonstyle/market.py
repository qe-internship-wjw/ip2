"""Market factor: exposure to overall market movement.

The market return is the cap-weighted excess return of every stock in the universe
(the undemeaned root of the structural-factor tree; :func:`market_returns`), also
the broad-market series used for beta hedging. A security's market factor is its
rolling beta on that series.

Currency risk is assumed hedged, so each stock's ``excess_return`` stays local and
USD market cap is used only for common-currency sizing.
"""

from __future__ import annotations

import polars as pl

from ..base import Applicability, Factor, FactorKind, register
from ._hierarchy import cap_weighted_return, stock_loadings


def market_returns(panel):
    """Cap-weighted daily market excess return, as ``[date, _factor]``."""
    lf = panel.lazy() if isinstance(panel, pl.DataFrame) else panel
    return cap_weighted_return(lf, ["date"]).select("date", pl.col("_r").alias("_factor"))


@register
class Market(Factor):
    name = "Market"
    shorthand = "MKT"
    sleeve = "NonStyle"
    kind = FactorKind.SYSTEMATIC
    applicability = Applicability.ALL_FINANCIALS

    def compute(self, panel, cfg):
        """Per-security rolling beta on the market return: ``[stock_id, date, MKT]``."""
        out = stock_loadings(panel, market_returns(panel), self.shorthand, cfg, ["date"])
        return out.lazy() if isinstance(panel, pl.LazyFrame) else out
