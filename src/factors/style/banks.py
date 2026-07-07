"""Banks-industry style factors.

Value:         Tangible Book Value to Price (TBV/P), Pre-Provision Net Revenue
               to Price (PPNR/P).
Leverage:      Tier 1 Capital Ratio (T1CR).
Quality:       NPL Coverage, NPL Ratio, Core Deposit Ratio (CDR).
Profitability: Total Pretax Expense to Average Assets (PTX/AA), Efficiency Ratio
               (ER), Return on Tangible Common Equity (ROTCE), Net Interest
               Margin (NIM).

Formulas follow the research plan's "Banks Industry Style Factors" table. All are
pure cross-sectional ratios.
"""

from __future__ import annotations

import polars as pl

from ..base import Applicability, FactorKind, register
from ._base import RatioFactor

@register
class TangibleBookToPrice(RatioFactor):
    name = "Tangible Book Value to Price"
    shorthand = "TBV/P"
    sleeve = "Value"
    kind = FactorKind.SYSTEMATIC
    applicability = Applicability.BANKS

    def expr(self, cfg) -> pl.Expr:
        return (pl.col("book_value") - pl.col("goodwill")) / pl.col("security_mcap_local")


@register
class PPNRToPrice(RatioFactor):
    name = "Pre-Provision Net Revenue to Price"
    shorthand = "PPNR/P"
    sleeve = "Value"
    kind = FactorKind.SYSTEMATIC
    applicability = Applicability.BANKS

    def expr(self, cfg) -> pl.Expr:
        return (pl.col("operating_income_ltm") + pl.col("loan_loss_provision_ltm")) / pl.col("security_mcap_local")


@register
class Tier1CapitalRatio(RatioFactor):
    name = "Tier 1 Capital Ratio"
    shorthand = "T1CR"
    sleeve = "Leverage"
    kind = FactorKind.SYSTEMATIC
    applicability = Applicability.BANKS

    def expr(self, cfg) -> pl.Expr:
        return pl.col("tier1_capital_ratio")


@register
class NPLCoverage(RatioFactor):
    name = "Nonperforming Loan Coverage Ratio"
    shorthand = "NPL Coverage"
    sleeve = "Quality"
    kind = FactorKind.SYSTEMATIC
    applicability = Applicability.BANKS

    def expr(self, cfg) -> pl.Expr:
        return pl.col("loan_loss_reserve") / pl.col("nonperforming_loans")


@register
class NPLRatio(RatioFactor):
    name = "Nonperforming Loan Ratio"
    shorthand = "NPL Ratio"
    sleeve = "Quality"
    kind = FactorKind.SYSTEMATIC
    applicability = Applicability.BANKS

    def expr(self, cfg) -> pl.Expr:
        return pl.col("nonperforming_loans") / pl.col("net_loans")


@register
class PretaxExpenseToAssets(RatioFactor):
    name = "Total Pretax Expense to Average Assets"
    shorthand = "PTX/AA"
    sleeve = "Profitability"
    kind = FactorKind.SYSTEMATIC
    applicability = Applicability.BANKS

    def expr(self, cfg) -> pl.Expr:
        return (
            pl.col("interest_ltm") + pl.col("operating_expenses_ltm") + pl.col("loan_loss_provision_ltm")
        ) / pl.col("assets")


@register
class EfficiencyRatio(RatioFactor):
    name = "Efficiency Ratio"
    shorthand = "ER"
    sleeve = "Profitability"
    kind = FactorKind.SYSTEMATIC
    applicability = Applicability.BANKS

    def expr(self, cfg) -> pl.Expr:
        return pl.col("operating_expenses_ltm") / pl.col("sales_ltm")


@register
class ReturnOnTangibleCommonEquity(RatioFactor):
    name = "Return on Tangible Common Equity"
    shorthand = "ROTCE"
    sleeve = "Profitability"
    kind = FactorKind.SYSTEMATIC
    applicability = Applicability.BANKS

    def expr(self, cfg) -> pl.Expr:
        return pl.col("earnings_ltm") / (pl.col("total_equity") - pl.col("goodwill"))


@register
class NetInterestMargin(RatioFactor):
    name = "Net Interest Margin"
    shorthand = "NIM"
    sleeve = "Profitability"
    kind = FactorKind.SYSTEMATIC
    applicability = Applicability.BANKS

    def expr(self, cfg) -> pl.Expr:
        return pl.col("net_interest_margin")
