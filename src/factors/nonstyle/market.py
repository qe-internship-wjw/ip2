"""Market factor: global market movement aggregated across the universe.

The overall market return is the cap-weighted excess return of every stock in
the universe -- the root of the structural-factor tree (see ``_hierarchy``) and
the one series that is not demeaned. It is the broad-market series used for
market-beta hedging.

Currency risk is assumed hedged, so we focus on stock performance alone: each
stock's ``excess_return`` stays local and USD market cap is used only for
common-currency sizing.
"""

from __future__ import annotations

import polars as pl

from ..base import Applicability, Factor, FactorKind, register
from ._hierarchy import cap_weighted_return


@register
class Market(Factor):
    name = "Market"
    shorthand = "MKT"
    sleeve = "NonStyle"
    kind = FactorKind.SYSTEMATIC
    applicability = Applicability.ALL_FINANCIALS

    def compute(self, panel, cfg):
        """Cap-weighted daily market excess return, one row per date.

        Returns a single-factor table with columns ``date`` and ``market``,
        sorted chronologically.
        """
        lf = panel.lazy() if isinstance(panel, pl.DataFrame) else panel

        return (
            cap_weighted_return(lf, ["date"])
            .select("date", pl.col("_r").alias("market"))
            .sort("date")
            .collect()
        )
