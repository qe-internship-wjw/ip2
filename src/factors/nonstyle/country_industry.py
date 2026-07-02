"""Country and industry factor loadings via factor-mimicking portfolios.

Each factor return is the cap-weighted (USD-sized) excess return of a stock group
minus its parent's return (see ``_hierarchy``); a security's factor is its rolling
beta on the return of the group it belongs to.

* **Country** -- per ``universe.country_granularity``: the security's ``region``
  return (relative to the whole market) or its ``home_country`` return (relative
  to its region).
* **Industry** -- the security's industry return within the tradeable universe
  (banks + insurance), relative to that set, at ``universe.industry_granularity``.
  The ``tradeable`` flag and ``industry`` label are precomputed on the market frame
  (:func:`src.data.joins.build_market_frame`).

Country/Market returns span the full universe; a loading is attached only to the
tradeable names (``panel.filter(tradeable)`` in each ``compute``).
"""

from __future__ import annotations

import polars as pl

from ..base import Applicability, Factor, FactorKind, register
from ._hierarchy import expand_by_group, relative_factor, stock_loadings


@register
class Country(Factor):
    name = "Country"
    shorthand = "CTRY"
    sleeve = "NonStyle"
    kind = FactorKind.SYSTEMATIC
    applicability = Applicability.ALL_FINANCIALS

    def compute(self, panel, cfg):
        """Per-country/region betas + region dummies, one column block per group.

        The country/region return is cap-weighted over *all* securities in that
        country/region (:func:`joins.build_market_frame`); loadings are estimated
        only for the tradeable names, then expanded per group so each country/region
        carries its own slope (``beta_{group}``) and baseline (``is_{group}``).
        """
        lf = panel.lazy() if isinstance(panel, pl.DataFrame) else panel

        if cfg["universe"]["country_granularity"] == "region":
            child, parent_keys = "region_code", ()
        else:
            child, parent_keys = "country_code", ("region_code",)

        long = relative_factor(lf, child, parent_keys)
        tradeable = lf.filter(pl.col("tradeable"))
        loadings = stock_loadings(tradeable, long, self.shorthand, cfg, ["date", child])
        out = expand_by_group(loadings, self.shorthand, child)
        return out.collect() if isinstance(panel, pl.DataFrame) else out


@register
class Industry(Factor):
    name = "Industry"
    shorthand = "IND"
    sleeve = "NonStyle"
    kind = FactorKind.SYSTEMATIC
    applicability = Applicability.ALL_FINANCIALS

    def compute(self, panel, cfg):
        """Per-industry betas + industry dummies, one column block per group.

        Industry is handled hierarchically: within the tradeable financials set the
        industry return is taken relative to that set's aggregate, isolating the
        industry (bank vs insurer) tilt. The ``tradeable`` flag and ``industry``
        label are precomputed on the market frame (:func:`joins.build_market_frame`),
        so no re-filtering against the universe is needed here. The beta is expanded
        per industry so each carries its own slope (``beta_{group}``) and baseline
        (``is_{group}``).
        """
        lf = panel.lazy() if isinstance(panel, pl.DataFrame) else panel
        tradeable = lf.filter(pl.col("tradeable"))
        long = relative_factor(tradeable, "industry")
        loadings = stock_loadings(tradeable, long, self.shorthand, cfg, ["date", "industry"])
        out = expand_by_group(loadings, self.shorthand, "industry")
        return out.collect() if isinstance(panel, pl.DataFrame) else out
