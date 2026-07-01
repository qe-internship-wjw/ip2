"""All-financials style factors (defined on banks *and* insurers).

Value:      Earnings Yield (E/P), Dividend Yield (DY).
Low Vol:    Trailing Return Volatility (TRV).
Momentum:   Volatility-Adjusted 12M-1M Return.
Size:       Log Market Capitalization.

Formulas follow the research plan's "All Financials Style Factors" table. The two
trailing-window factors (TRV, Momentum) are time-series per security and override
``compute`` to roll over the daily panel; the rest are pure cross-sectional
ratios expressed via :class:`RatioFactor`.
"""

from __future__ import annotations

import polars as pl

from ..base import Applicability, Factor, FactorKind, register
from ._base import RatioFactor, finalize, to_lazy

# Trailing-window sizes in trading days (~21 per month, ~252 per year).
_MONTH = 21
_YEAR = 252
_TRV_YEARS = 3


@register
class EarningsYield(RatioFactor):
    name = "Earnings Yield"
    shorthand = "E/P"
    sleeve = "Value"
    kind = FactorKind.SYSTEMATIC
    applicability = Applicability.ALL_FINANCIALS

    def expr(self, cfg) -> pl.Expr:
        return pl.col("earnings_ltm") / pl.col("security_mcap_local")


@register
class DividendYield(RatioFactor):
    name = "Dividend Yield"
    shorthand = "DY"
    sleeve = "Value"
    kind = FactorKind.SYSTEMATIC
    applicability = Applicability.ALL_FINANCIALS

    def expr(self, cfg) -> pl.Expr:
        return pl.col("dividend_ltm") / pl.col("security_mcap_local")


@register
class LogMarketCap(RatioFactor):
    name = "Log Market Capitalization"
    shorthand = "ln(MCap)"
    sleeve = "Size"
    kind = FactorKind.SYSTEMATIC
    applicability = Applicability.ALL_FINANCIALS

    def expr(self, cfg) -> pl.Expr:
        return pl.col("mcap_usd").log()


@register
class TrailingReturnVolatility(Factor):
    name = "Trailing Return Volatility"
    shorthand = "TRV"
    sleeve = "Low Vol"
    kind = FactorKind.SYSTEMATIC
    applicability = Applicability.ALL_FINANCIALS

    def compute(self, panel, cfg):
        """sqrt(Var(total_return)) over a trailing 3-year window, per security."""
        lf = to_lazy(panel).sort("stock_id", "date")
        window = _TRV_YEARS * _YEAR
        score = (
            pl.col("total_return")
            .rolling_std(window_size=window, min_samples=_YEAR)
            .over("stock_id")
        )
        return finalize(lf, score, self.shorthand, panel)


@register
class VolAdjustedMomentum(Factor):
    name = "Volatility-Adjusted Momentum"
    shorthand = "Momentum"
    sleeve = "Momentum"
    # Momentum is the transparent behavioural case (see research plan).
    kind = FactorKind.BEHAVIOURAL
    applicability = Applicability.ALL_FINANCIALS

    def compute(self, panel, cfg):
        """(prod_{t-12m}^{t-1m}(1+r) - 1) / sigma(r), per security.

        The 12M-1M cumulative return is the trailing-12M log return minus the
        trailing-1M log return (so the most recent month is skipped), exponentiated
        back to a simple return, then scaled by the trailing-12M return volatility.
        """
        lf = to_lazy(panel).sort("stock_id", "date")

        log1p = (pl.col("total_return") + 1.0).log()
        sum_12m = log1p.rolling_sum(window_size=12 * _MONTH, min_samples=9 * _MONTH).over("stock_id")
        sum_1m = log1p.rolling_sum(window_size=_MONTH, min_samples=_MONTH // 2).over("stock_id")
        cum_return = (sum_12m - sum_1m).exp() - 1.0

        vol = (
            pl.col("total_return")
            .rolling_std(window_size=12 * _MONTH, min_samples=6 * _MONTH)
            .over("stock_id")
        )
        return finalize(lf, cum_return / vol, self.shorthand, panel)
