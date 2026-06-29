"""Raw data IO and caching.

Use lazy reads (scan_ipc()) and an optional processed-data cache to avoid
re-joining on every run.

Tables:
    price-001.feather                   daily price/return/mcap/volume
    security_master.feather             static security reference (GICS, country, currency)
    fundamental_master.feather          point-in-time fundamentals
    fundamental_master_extended.feather
    fx_rates.feather                    daily FX to USD
    risk_free_rate.feather              daily risk-free rate by country
    country_mapping.feather             country -> region
    industry_mapping.feather            SIC + FactSet industry codes
    zero_curve.feather                  sovereign zero rates (yield-curve factors)
"""

from __future__ import annotations


def load_table(name, cfg):
    """Load a single raw table by logical name from the configured data root."""
    raise NotImplementedError


def load_all(cfg):
    """Load every raw table required by the pipeline into a dict of frames."""
    raise NotImplementedError
