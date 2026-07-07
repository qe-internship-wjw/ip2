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
    input_frame = "market_frame"

    def compute(self, panel, cfg):
        """Per-security rolling beta on the market return: ``[stock_id, date, MKT]``.

        ``panel`` is the full-universe market frame (:func:`joins.build_market_frame`):
        the market return is cap-weighted over *every* security, but loadings are
        estimated only for the tradeable names.
        """
        lf = panel.lazy() if isinstance(panel, pl.DataFrame) else panel
        factor = market_returns(lf)
        tradeable = lf.filter(pl.col("tradeable"))
        out = stock_loadings(tradeable, factor, self.shorthand, cfg, ["date"])
        return out.collect() if isinstance(panel, pl.DataFrame) else out
