"""Country and industry factors via hierarchical factor-mimicking portfolios.

Both are demeaned partitions of the structural-factor tree (see ``_hierarchy``):
each portfolio is the cap-weighted (USD-sized) excess return of a group of stocks
minus its parent's return, so the children of any node sum to zero when weighted
by USD market cap (orthogonal to the level above).

* **Country** -- selected by ``universe.country_granularity``:
    ``region``        one portfolio per region (APAC / EMEA / AMER), relative to
                      the whole market. Columns ``region:<code>``.
    ``home_country``  one portfolio per country, relative to its *region*.
                      Columns ``country:<code>``.
* **Industry** -- two flat layers:
    1. ``tradeable`` (banks + insurance, per :func:`universe.sector_set`) vs the
       ``rest`` of the universe, relative to the whole market. Columns
       ``universe:tradeable`` / ``universe:rest``.
    2. the sector/industry split *within* the tradeable universe, relative to the
       tradeable set's return. Columns ``sector:<label>`` or ``industry:<label>``
       per ``universe.industry_granularity`` (2 or 3 children respectively).
"""

from __future__ import annotations

import polars as pl

from ...universe import industry_labels, sector_set
from ..base import Applicability, Factor, FactorKind, register
from ._hierarchy import relative_factor, wide_factors


@register
class Country(Factor):
    name = "Country"
    shorthand = "CTRY"
    sleeve = "NonStyle"
    kind = FactorKind.SYSTEMATIC
    applicability = Applicability.ALL_FINANCIALS

    def compute(self, panel, cfg):
        """Hierarchical country/region factor-mimicking portfolio returns.

        Returns a wide table with a ``date`` column and one ``region:<code>``
        (``region`` granularity) or ``country:<code>`` (``home_country``
        granularity) column per group, sorted chronologically.
        """
        lf = panel.lazy() if isinstance(panel, pl.DataFrame) else panel

        if cfg["universe"]["country_granularity"] == "region":
            child, parent_keys, prefix = "region_code", (), "region"
        else:
            child, parent_keys, prefix = "country_code", ("region_code",), "country"

        return wide_factors(relative_factor(lf, child, parent_keys), child, prefix)


@register
class Industry(Factor):
    name = "Industry"
    shorthand = "IND"
    sleeve = "NonStyle"
    kind = FactorKind.SYSTEMATIC
    applicability = Applicability.ALL_FINANCIALS

    def compute(self, panel, cfg):
        """Hierarchical industry factor-mimicking portfolio returns.

        Returns a wide table with a ``date`` column plus, across the two layers,
        ``universe:{tradeable,rest}`` and ``{sector|industry}:<label>`` columns,
        sorted chronologically.
        """
        lf = panel.lazy() if isinstance(panel, pl.DataFrame) else panel

        # ── Layer 1: tradeable vs rest, relative to the whole market ──────────
        # Tradeable membership is a per-stock property (static classification +
        # a populated sector metric), so tag it via a semi-join on stock_id.
        tradeable_ids = (
            sector_set(lf, cfg)
            .select("stock_id")
            .unique()
            .with_columns(_tradeable=pl.lit(True))
        )
        lf = lf.join(tradeable_ids, on="stock_id", how="left").with_columns(
            _membership=pl.when(pl.col("_tradeable").fill_null(False))
            .then(pl.lit("tradeable"))
            .otherwise(pl.lit("rest"))
        )
        layer1 = wide_factors(
            relative_factor(lf, "_membership"), "_membership", "universe"
        )

        # ── Layer 2: sector/industry within the tradeable universe ────────────
        # Defined only on the tradeable set, relative to its own return.
        granularity = cfg["universe"].get("industry_granularity", "industry")
        tradeable = industry_labels(lf.filter(pl.col("_membership") == "tradeable"), cfg)
        layer2 = wide_factors(relative_factor(tradeable, "industry"), "industry", granularity)

        # Whole-market dates are a superset of tradeable dates, so left-join.
        return layer1.join(layer2, on="date", how="left").sort("date")
