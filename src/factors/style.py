"""Style factors for financials.

Three tiers of applicability:

All-financials:  P/E, Dividend Yield, Trailing Return Volatility, Volatility-
                 Adjusted Momentum, Size (ln MCap), Amihud Illiquidity.
Banks:           TBV/P, PPNR/P, Tier 1 Capital Ratio, NPL Coverage, NPL Ratio,
                 Core Deposit Ratio, PTX/AA, Efficiency Ratio, ROTCE, NIM.
Insurance:       P/B, P/PI, Reserve Leverage, Float Investment Yield, LA/TR,
                 ROE, Underwriting Margin.

Each factor is a `Factor` subclass registered via @register; formulas map to the
columns in data README files (e.g. earnings_ltm, book_value, goodwill,
net_interest_margin, insurance_reserves). Start with the all-financials set, then
layer the industry-specific factors.
"""

from __future__ import annotations

from .base import Applicability, Factor, FactorKind, register

# TODO: implement concrete factors, e.g.
#
# @register
# class PriceToEarnings(Factor):
#     name = "Price to Earnings"
#     shorthand = "P/E"
#     sleeve = "Value"
#     kind = FactorKind.BEHAVIOURAL
#     applicability = Applicability.ALL_FINANCIALS
#
#     def compute(self, panel, cfg):
#         return panel["security_mcap_local"] / panel["earnings_ltm"]
