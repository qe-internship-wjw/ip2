"""Country and (sub)industry factors via factor-mimicking portfolios.

The design is hierarchical, selected by ``universe.country_granularity``:

* ``region``  -- one portfolio per region (``region_code`` from
  ``country_mapping``: APAC / EMEA / AMER). Each is the region's cap-weighted
  excess return minus the overall market return, so the region portfolios sum to
  zero when market-cap weighted (orthogonal to the market factor).
* ``home_country`` (a.k.a. country) -- one portfolio per country, each the
  country's cap-weighted excess return minus its *region's* return, so the
  countries within a region sum to zero when cap-weighted (orthogonal to that
  region's factor).
"""

from __future__ import annotations

import polars as pl

from ..base import Applicability, Factor, FactorKind, register


@register
class Country(Factor):
    name = "Country"
    shorthand = "CTRY"
    sleeve = "NonStyle"
    kind = FactorKind.SYSTEMATIC
    applicability = Applicability.ALL_FINANCIALS

    def compute(self, panel, cfg):
        """Hierarchical country/region factor-mimicking portfolio returns.

        Returns a wide table with one ``date`` column and one column per region
        (``region`` granularity) or per country (``home_country`` granularity),
        sorted chronologically. Within each date the columns sum to zero when
        weighted by the group's USD market cap.
        """
        granularity = cfg["universe"]["country_granularity"]

        lf = panel.lazy() if isinstance(panel, pl.DataFrame) else panel

        if granularity == "region":
            child, parent = "region_code", None
        else:
            child, parent = "country_code", "region_code"
            lf = lf.filter(pl.col("country_code").is_not_null())

        # Cap-weighted excess return per group (numerator/denominator kept split
        # so the parent baseline can be re-aggregated from the same pieces).
        group_keys = ["date", child] if parent is None else ["date", parent, child]
        comp = lf.group_by(group_keys).agg(
            _w=(pl.col("excess_return") * pl.col("mcap_usd")).sum(),
            _m=pl.col("mcap_usd").sum(),
        )

        # Baseline return the group is demeaned against: the whole market (per
        # date) for regions, or the parent region (per date) for countries.
        base_keys = ["date"] if parent is None else ["date", parent]
        base = comp.group_by(base_keys).agg(
            _wb=pl.col("_w").sum(),
            _mb=pl.col("_m").sum(),
        ).with_columns(_r_base=pl.col("_wb") / pl.col("_mb"))

        comp = (
            comp.with_columns(_r=pl.col("_w") / pl.col("_m"))
            .join(base.select(*base_keys, "_r_base"), on=base_keys, how="left")
            .with_columns(_factor=pl.col("_r") - pl.col("_r_base"))
        )

        return (
            comp.collect()
            .pivot(values="_factor", index="date", on=child)
            .sort("date")
        )
