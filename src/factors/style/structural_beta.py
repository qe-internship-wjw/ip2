"""Structural-beta signals: the non-style loadings traded as style factors.

The structural factors (:mod:`src.factors.nonstyle`) estimate each security's
rolling beta on the market / its country / its industry factor-mimicking return.
Those loadings enter the pipeline in two representations:

* **expanded** (``beta_{g}`` slopes + ``is_{g}`` dummies, ``expand_by_group``) --
  the neutralization design and the risk model's ``B``. Unchanged by this module.
* **stacked** (this module) -- the security's own-group beta as a *tradeable
  signal*: the cross-sectional low-beta anomaly (betting-against-beta) within
  financials. The raw score is the beta itself; the sign of the premium is
  discovered downstream (Fama-MacBeth), so a flat/inverted security-market line
  simply prices as ``lambda <= 0``.

Both representations come from the very same :func:`~src.factors.nonstyle.
_hierarchy.stock_loadings` estimate (same ``factors.loadings.window_months``
window), so the alpha view and the risk-model view of a stock's beta never drift
apart.

The signals set ``neutralize = False`` (:class:`~src.factors.base.Factor`): the
stacked beta equals the row-sum of its expanded ``beta_{g}`` design block, so the
neutralization OLS would annihilate it -- and economically these factors are the
*explicit* systematic bets the styles are stripped of. They are still regularized
(winsorized / imputed / z-scored) like any style factor. The hierarchy already
demeans each country/industry factor return against its parent, so the loadings
need no further cross-sectional neutralization; what the hierarchy cannot provide
is *comparability of loadings*, handled at two levels:

* **across groups** -- a beta on group ``g``'s factor scales as ``1/var(f_g)``,
  so the country / industry signals are median-centered per (calendar month,
  group) before regularization. The market beta has a single group: no centering.
* **across sub-universes** -- market and country betas are all-financials factors
  (one pooled cross-section); the industry beta is registered per sector, since
  banks and insurers load on structurally different industry factors, so its
  z-scores never pool across the sub-universe boundary.

Unlike the other style factors these read the **market frame**
(:func:`src.data.joins.build_market_frame`), not the sector panel: the
factor-mimicking returns are cap-weighted over the full universe, while loadings
attach to the tradeable names only.
"""

from __future__ import annotations

import polars as pl

from ..base import Applicability, Factor, FactorKind, register
from ..nonstyle._hierarchy import relative_factor, stock_loadings
from ..nonstyle.market import market_returns


def center_by_group(loadings, name, group_col):
    """Median-center a stacked loading per (calendar month, group).

    Betas on different mimicking portfolios sit at mechanically different levels,
    so cross-group comparability requires removing the per-group location first.
    The cross-section key is the calendar month, not the raw date -- month-end
    trading days differ per security under staggered calendars.
    """
    month = pl.col("date").dt.truncate("1mo")
    return loadings.with_columns(
        (pl.col(name) - pl.col(name).median().over(month, group_col)).alias(name)
    )


class _StructuralBeta(Factor):
    """Base: stacked own-group rolling beta, group-centered when grouped.

    Subclasses implement :meth:`_stacked` returning the lazy loading frame (the
    beta column named by ``shorthand``) and its group column (``None`` for the
    market). ``compute`` takes the **market frame** and returns the whole
    tradeable cross-section; sector applicability is enforced downstream by the
    regularization masks.
    """

    sleeve = "Beta"
    kind = FactorKind.SYSTEMATIC
    applicability = Applicability.ALL_FINANCIALS
    neutralize = False

    def _stacked(self, lf, cfg):
        raise NotImplementedError

    def compute(self, panel, cfg):
        lf = panel.lazy() if isinstance(panel, pl.DataFrame) else panel
        loadings, group_col = self._stacked(lf, cfg)
        if group_col is not None:
            loadings = center_by_group(loadings, self.shorthand, group_col)
        out = loadings.select("stock_id", "date", self.shorthand)
        return out.collect() if isinstance(panel, pl.DataFrame) else out


@register
class MarketBetaSignal(_StructuralBeta):
    name = "Market Beta"
    shorthand = "Beta~MKT"

    def _stacked(self, lf, cfg):
        tradeable = lf.filter(pl.col("tradeable"))
        return stock_loadings(tradeable, market_returns(lf), self.shorthand, cfg, ["date"]), None


@register
class CountryBetaSignal(_StructuralBeta):
    name = "Country Beta"
    shorthand = "Beta~CTRY"

    def _stacked(self, lf, cfg):
        # Mirror nonstyle.country_industry.Country: the same granularity switch,
        # so signal and design are betas on the same mimicking portfolios.
        if cfg["universe"]["country_granularity"] == "region":
            child, parent_keys = "region_code", ()
        else:
            child, parent_keys = "country_code", ("region_code",)
        long = relative_factor(lf, child, parent_keys)
        tradeable = lf.filter(pl.col("tradeable"))
        return stock_loadings(tradeable, long, self.shorthand, cfg, ["date", child]), child


class _IndustryBetaSignal(_StructuralBeta):
    """Industry beta, registered per sector so z-scores never pool across it.

    Both variants compute the identical stacked beta over the whole tradeable
    set; the sector applicability masks in ``preprocess`` null out the other
    sub-universe, so each column is standardized strictly within its own sector.
    """

    def _stacked(self, lf, cfg):
        tradeable = lf.filter(pl.col("tradeable"))
        long = relative_factor(tradeable, "industry")
        return stock_loadings(tradeable, long, self.shorthand, cfg, ["date", "industry"]), "industry"


@register
class BankIndustryBetaSignal(_IndustryBetaSignal):
    name = "Industry Beta (Banks)"
    shorthand = "Beta~IND (B)"
    applicability = Applicability.BANKS


@register
class InsuranceIndustryBetaSignal(_IndustryBetaSignal):
    name = "Industry Beta (Insurers)"
    shorthand = "Beta~IND (I)"
    applicability = Applicability.INSURANCE
