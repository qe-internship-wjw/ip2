"""Market factor: global market movement aggregated across the universe.

The overall market return is the cap-weighted excess return of every stock in
the universe. It is the broad-market series used for market-beta hedging.

Currency risk is assumed hedged, so we focus on stock performance alone. Each
stock's ``excess_return`` stays local.
"""

from __future__ import annotations

import polars as pl

from ..base import Applicability, Factor, FactorKind, register


@register
class Market(Factor):
    name = "Market"
    shorthand = "MKT"
    sleeve = "NonStyle"
    kind = FactorKind.SYSTEMATIC
    applicability = Applicability.ALL_FINANCIALS

    def compute(self, panel, cfg):
        """Cap-weighted daily market excess return, one row per date.

        Returns a single-factor table with columns ``date`` and ``Market``,
        sorted chronologically.
        """
        lf = panel.lazy() if isinstance(panel, pl.DataFrame) else panel

        # Cap-weighted local excess return per date
        #   R_m = sum_i (mcap_i * excess_i) / sum_i mcap_i
        market = (
            lf.group_by("date")
            .agg(
                (pl.col("excess_return") * pl.col("mcap_usd")).sum().alias("_wsum"),
                pl.col("mcap_usd").sum().alias("_wtot"),
            )
            .with_columns((pl.col("_wsum") / pl.col("_wtot")).alias(self.name))
            .select("date", self.name)
            .sort("date")
        )

        return market
