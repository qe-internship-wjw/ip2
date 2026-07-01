"""Insurance-industry style factors.

Value:         Book to Price (B/P), Premium Income to Price (PI/P).
Leverage:      Reserve Leverage Ratio (RLR).
Quality:       Float Investment Yield (FIY), Liquid Assets to Technical Reserves
               (LA/TR), Return on Equity (ROE).
Profitability: Underwriting Margin (UWM).

Formulas follow the research plan's "Insurance Industry Style Factors" table. All
are pure cross-sectional ratios.
"""

from __future__ import annotations

import polars as pl

from ..base import Applicability, FactorKind, register
from ._base import RatioFactor


@register
class BookToPrice(RatioFactor):
    name = "Book to Price"
    shorthand = "B/P"
    sleeve = "Value"
    kind = FactorKind.SYSTEMATIC
    applicability = Applicability.INSURANCE

    def expr(self, cfg) -> pl.Expr:
        return pl.col("book_value") / pl.col("security_mcap_local")


@register
class PremiumIncomeToPrice(RatioFactor):
    name = "Premium Income to Price"
    shorthand = "PI/P"
    sleeve = "Value"
    kind = FactorKind.SYSTEMATIC
    applicability = Applicability.INSURANCE

    def expr(self, cfg) -> pl.Expr:
        return pl.col("insurance_premium_ltm") / pl.col("security_mcap_local")


@register
class ReserveLeverageRatio(RatioFactor):
    name = "Reserve Leverage Ratio"
    shorthand = "RLR"
    sleeve = "Leverage"
    kind = FactorKind.SYSTEMATIC
    applicability = Applicability.INSURANCE

    def expr(self, cfg) -> pl.Expr:
        return pl.col("insurance_reserves") / pl.col("total_equity")


@register
class FloatInvestmentYield(RatioFactor):
    name = "Float Investment Yield"
    shorthand = "FIY"
    sleeve = "Quality"
    kind = FactorKind.SYSTEMATIC
    applicability = Applicability.INSURANCE

    def expr(self, cfg) -> pl.Expr:
        return pl.col("insurance_investment_income_ltm") / pl.col("insurance_reserves")


@register
class LiquidAssetsToReserves(RatioFactor):
    name = "Liquid Assets to Technical Reserves"
    shorthand = "LA/TR"
    sleeve = "Quality"
    kind = FactorKind.SYSTEMATIC
    applicability = Applicability.INSURANCE

    def expr(self, cfg) -> pl.Expr:
        return pl.col("cash_and_st_investments") / pl.col("insurance_reserves")


@register
class ReturnOnEquity(RatioFactor):
    name = "Return on Equity"
    shorthand = "ROE"
    sleeve = "Quality"
    kind = FactorKind.SYSTEMATIC
    applicability = Applicability.INSURANCE

    def expr(self, cfg) -> pl.Expr:
        return pl.col("earnings_ltm") / pl.col("total_equity")


@register
class UnderwritingMargin(RatioFactor):
    name = "Underwriting Margin"
    shorthand = "UWM"
    sleeve = "Profitability"
    kind = FactorKind.SYSTEMATIC
    applicability = Applicability.INSURANCE

    def expr(self, cfg) -> pl.Expr:
        return (
            pl.col("insurance_premium_ltm")
            - pl.col("insurance_loss_ltm")
            - pl.col("underwriting_expense_ltm")
        ) / pl.col("insurance_premium_ltm")
