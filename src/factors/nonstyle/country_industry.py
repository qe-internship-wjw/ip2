"""Country and industry factor loadings via factor-mimicking portfolios.

Each factor return is the cap-weighted (USD-sized) excess return of a stock group
minus its parent's return (see ``_hierarchy``); a security's factor is its rolling
beta on the return of the group it belongs to.

* **Country** -- per ``universe.country_granularity``: the security's ``region``
  return (relative to the whole market) or its ``home_country`` return (relative
  to its region).
* **Industry** -- the security's industry return within the tradeable universe
  (banks + insurance, per :func:`universe.sector_set`), relative to that set, at
  ``universe.industry_granularity``.
"""

from __future__ import annotations

import polars as pl

from ...universe import industry_labels, sector_set
from ..base import Applicability, Factor, FactorKind, register
from ._hierarchy import relative_factor, stock_loadings


@register
class Country(Factor):
    name = "Country"
    shorthand = "CTRY"
    sleeve = "NonStyle"
    kind = FactorKind.SYSTEMATIC
    applicability = Applicability.ALL_FINANCIALS

    def compute(self, panel, cfg):
        """Per-security rolling beta on its country/region factor return."""
        lf = panel.lazy() if isinstance(panel, pl.DataFrame) else panel

        if cfg["universe"]["country_granularity"] == "region":
            child, parent_keys = "region_code", ()
        else:
            child, parent_keys = "country_code", ("region_code",)

        long = relative_factor(lf, child, parent_keys)
        out = stock_loadings(panel, long, self.shorthand, cfg, ["date", child])
        return out.lazy() if isinstance(panel, pl.LazyFrame) else out


@register
class Industry(Factor):
    name = "Industry"
    shorthand = "IND"
    sleeve = "NonStyle"
    kind = FactorKind.SYSTEMATIC
    applicability = Applicability.ALL_FINANCIALS

    def compute(self, panel, cfg):
        """Per-security rolling beta on its industry factor return.

        The industry return is defined within the tradeable set (relative to that
        set's return), so the loading is estimated on the tradeable universe.
        """
        lf = panel.lazy() if isinstance(panel, pl.DataFrame) else panel
        tradeable = industry_labels(sector_set(lf, cfg), cfg)
        long = relative_factor(tradeable, "industry")
        out = stock_loadings(tradeable, long, self.shorthand, cfg, ["date", "industry"])
        return out.lazy() if isinstance(panel, pl.LazyFrame) else out
