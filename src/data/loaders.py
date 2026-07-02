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

from pathlib import Path

import polars as pl

# Tables that live in a named subfolder under data/raw
_SUBDIRS: dict[str, str] = {
    "fundamental_master_extended": "Industry Fundamentals Data",
    "zero_curve": "Zero Rates Data",
}

_ALL_TABLES = [
    "price",
    "security_master",
    "fundamental_master",
    "fundamental_master_extended",
    "fx_rates",
    "risk_free_rate",
    "country_mapping",
    "industry_mapping",
    "zero_curve",
]

# Canonical categorical schema. Identifier / code / classification columns are
# semantically categorical and must load as ``pl.Categorical``
_CATEGORICAL_COLS: dict[str, list[str]] = {
    "price": ["stock_id"],
    "security_master": [
        "stock_id",
        "name",
        "country_code",
        "currency_code",
        "bloomberg_composite_ticker",
        "gics_sector_name",
        "gics_industry_group_name",
        "gics_industry_name",
    ],
    "fundamental_master": ["stock_id"],
    "fundamental_master_extended": ["stock_id"],
    "fx_rates": ["currency_code"],
    "risk_free_rate": ["country_code"],
    "country_mapping": ["country_code", "region_code"],
    "industry_mapping": [
        "stock_id",
        "sic_code",
        "sic_name",
        "factset_industry_code",
        "factset_industry_name",
    ],
    "zero_curve": ["currency"],
}


def _cast_categoricals(name: str, lf: pl.LazyFrame) -> pl.LazyFrame:
    """Cast the table's canonical categorical columns to ``pl.Categorical``.

    Only columns actually present in the frame are cast, so a schema change in a
    raw feather degrades gracefully rather than raising.
    """
    wanted = _CATEGORICAL_COLS.get(name, [])
    if not wanted:
        return lf
    present = set(lf.collect_schema().names())
    casts = [pl.col(c).cast(pl.Categorical) for c in wanted if c in present]
    return lf.with_columns(casts) if casts else lf


def _resolve_path(name: str, cfg) -> Path:
    root = Path(cfg["data"]["root"])
    if name == "price":
        return root / cfg["data"]["price_glob"]
    subdir = _SUBDIRS.get(name)
    if subdir:
        return root / subdir / f"{name}.feather"
    return root / f"{name}.feather"


def load_table(name: str, cfg) -> pl.LazyFrame:
    """Load a single raw table by logical name from the configured data root."""
    return _cast_categoricals(name, pl.scan_ipc(_resolve_path(name, cfg)))


def load_all(cfg) -> dict[str, pl.LazyFrame]:
    """Load every raw table required by the pipeline into a dict of frames."""
    return {name: load_table(name, cfg) for name in _ALL_TABLES}
