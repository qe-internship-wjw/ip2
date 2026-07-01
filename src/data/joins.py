"""Point-in-time joins and currency / risk-free normalization.

Builds the analysis panel from the raw tables while preventing look-ahead bias.
Join rules:
    price        <- security_master      on stock_id
    price        <- fundamental_master,
                    fundamental_extended on stock_id, observation_date <= date
    price        <- fx_rates             on currency_code, date  (local -> USD)
    price        <- risk_free_rate       on country_code, date   (-> excess_return)
    security     <- country_mapping      on country_code
    security     <- industry_mapping     on stock_id

Stocks are evaluated in their own local currency by default.
"""

from __future__ import annotations

import polars as pl

from ..factors.nonstyle.yield_curve import attach_nelson_siegel, fit_nelson_siegel


def attach_fundamentals(prices, fundamentals, fundamentals_extended):
    """As-of join fundamentals onto prices using observation_date <= date.
    """
    combined = fundamentals.join(
        fundamentals_extended,
        on=["stock_id", "date_fundamental"],
        how="left",
    )

    return prices.sort("date").join_asof(
        combined.sort("observation_date"),
        left_on="date",
        right_on="observation_date",
        by="stock_id",
        strategy="backward",
    )


def attach_reference(prices, security_master, country_mapping, industry_mapping):
    """Attach static security/country/industry reference columns."""
    return (
        prices
        .join(security_master, on="stock_id", how="left")
        .join(country_mapping, on="country_code", how="left")
        .join(industry_mapping, on="stock_id", how="left")
    )


def to_excess_return(panel, risk_free):
    """Subtract the matched daily risk-free rate to produce excess_return.

    The risk_free table date column is cast from Datetime to Date to align with
    the price panel.
    """
    rfr = risk_free.with_columns(pl.col("date").cast(pl.Date))
    return (
        panel
        .join(rfr, on=["country_code", "date"], how="left")
        .with_columns(
            excess_return=(pl.col("total_return") - pl.col("risk_free_rate"))
        )
    )


def build_panel(raw, cfg):
    """Assemble the full point-in-time joined panel from raw tables."""
    fx = raw["fx_rates"].with_columns(pl.col("date").cast(pl.Date))

    panel = attach_reference(
        raw["price"],
        raw["security_master"],
        raw["country_mapping"],
        raw["industry_mapping"],
    )
    panel = panel.join(fx, on=["currency_code", "date"], how="left")
    # Precompute USD market cap now that fx is attached; many factors weight by
    # it, so materialize it once here rather than recomputing per factor.
    panel = panel.with_columns(
        mcap_usd=pl.col("security_mcap_local") * pl.col("fx_to_usd")
    )
    panel = attach_fundamentals(
        panel,
        raw["fundamental_master"],
        raw["fundamental_master_extended"],
    )
    panel = to_excess_return(panel, raw["risk_free_rate"])

    # Summarise each sovereign curve (Nelson-Siegel) and attach parameters.
    ns_params = fit_nelson_siegel(raw["zero_curve"], cfg)
    panel = attach_nelson_siegel(panel, ns_params)
    
    return panel
