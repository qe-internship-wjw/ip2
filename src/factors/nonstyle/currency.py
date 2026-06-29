"""Currency factor: cash-flow sensitivity to a trade-weighted FX index.

Direct FX-rate sensitivity of stock prices appears unpriced or negative, but
sensitivity of *cash flows* to FX reveals a positive currency risk premium
(Du & Hu, 2011). We model, per security:

    delta Metric_i = alpha + Sensitivity * (delta FX Rate) + eps_i

where Metric adapts operating cash flow to the subsector (Net Interest Margin
for banks, Underwriting Margin for insurers) and FX Rate is a discretionary
trade-weighted index keyed to the institution's home country. The estimated
Sensitivity is the factor exposure. Direct currency *translation* effects are an
implementation artifact and are not rewarded (handled in evaluation, not here).
"""

from __future__ import annotations


def trade_weighted_index(fx_rates, home_country, cfg):
    """Build the discretionary trade-weighted FX index for a home country."""
    raise NotImplementedError


def cashflow_fx_sensitivity(panel, fx_index, cfg):
    """Regress delta(subsector cash-flow metric) on delta(FX index) -> Sensitivity."""
    raise NotImplementedError
