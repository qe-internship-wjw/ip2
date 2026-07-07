"""Style factors for financials.

Three tiers of applicability, each in its own module:

* :mod:`all_financials` -- defined on banks *and* insurers: Earnings Yield,
  Dividend Yield, Trailing Return Volatility, Volatility-Adjusted Momentum, Size.
* :mod:`banks`          -- TBV/P, PPNR/P, Tier 1 Capital Ratio, NPL Coverage,
  NPL Ratio, Core Deposit Ratio, PTX/AA, Efficiency Ratio, ROTCE, NIM.
* :mod:`insurance`      -- B/P, PI/P, Reserve Leverage, Float Investment Yield,
  LA/TR, ROE, Underwriting Margin.
* :mod:`yield_curve`    -- interest-rate sensitivities across all three tiers:
  Return sensitivity (all-financials), NIM sensitivity (banks), FIY sensitivity
  (insurance).
* :mod:`structural_beta` -- the structural loadings traded as signals
  (betting-against-beta): Market and Country beta (all-financials), Industry
  beta (per sector). Exempt from neutralization (``neutralize = False``) and,
  unlike the other style factors, computed from the **market frame**.

Every factor is a :class:`~src.factors.base.Factor` subclass registered via
``@register``; each returns a long ``[stock_id, date, <shorthand>]`` frame of raw
cross-sectional scores (see :mod:`._base`). Importing this package imports the
three submodules, which is what populates the registry.
"""

from __future__ import annotations

from . import (  # noqa: F401  (registration side-effect)
    all_financials,
    banks,
    insurance,
    structural_beta,
    yield_curve,
)
