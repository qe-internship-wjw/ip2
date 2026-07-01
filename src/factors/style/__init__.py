"""Style factors for financials.

Three tiers of applicability, each in its own module:

* :mod:`all_financials` -- defined on banks *and* insurers: Earnings Yield,
  Dividend Yield, Trailing Return Volatility, Volatility-Adjusted Momentum, Size.
* :mod:`banks`          -- TBV/P, PPNR/P, Tier 1 Capital Ratio, NPL Coverage,
  NPL Ratio, Core Deposit Ratio, PTX/AA, Efficiency Ratio, ROTCE, NIM.
* :mod:`insurance`      -- B/P, PI/P, Reserve Leverage, Float Investment Yield,
  LA/TR, ROE, Underwriting Margin.

Every factor is a :class:`~src.factors.base.Factor` subclass registered via
``@register``; each returns a long ``[stock_id, date, <shorthand>]`` frame of raw
cross-sectional scores (see :mod:`._base`). Importing this package imports the
three submodules, which is what populates the registry.
"""

from __future__ import annotations

from . import all_financials, banks, insurance  # noqa: F401  (registration side-effect)
