"""Point-in-time joins and currency / risk-free normalization.

Builds the analysis panel from the raw tables while preventing look-ahead bias.
Join rules:
    price        <- security_master      on stock_id
    price        <- fundamental_master   on stock_id, observation_date <= date
    price        <- fx_rates             on currency_code, date  (local -> USD)
    price        <- risk_free_rate       on country_code, date   (-> excess_return)
    security     <- country_mapping      on country_code
    security     <- industry_mapping     on stock_id

Stocks are evaluated in their own local currency by default.
"""

from __future__ import annotations


def attach_fundamentals(prices, fundamentals):
    """As-of join fundamentals onto prices using observation_date <= date."""
    raise NotImplementedError


def attach_reference(prices, security_master, country_mapping, industry_mapping):
    """Attach static security/country/industry reference columns."""
    raise NotImplementedError


def to_excess_return(panel, risk_free):
    """Subtract the matched daily risk-free rate to produce excess_return."""
    raise NotImplementedError


def build_panel(raw, cfg):
    """Assemble the full point-in-time joined panel from raw tables."""
    raise NotImplementedError
