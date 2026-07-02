"""Universe definition and data splits.

Central to this project: the strategy is restricted to financials. We distinguish
the *market set* (all securities) from the *sector set* (banks + insurance), and
tag subindustries (retail vs investment banks; life vs P&C insurers) using GICS,
SIC and FactSet codes.
"""

from __future__ import annotations

import polars as pl

# ---------------------------------------------------------------------------
# FactSet industry name → canonical label
# Two granularity levels only:
#   sector   : "bank" | "insurance"
#   industry : "bank" | "insurance_life" | "insurance_multiline" | "insurance_pc"
#
# Excluded from universe:
#   - Investment Banks/Brokers
#   - Insurance Brokers/Services
# Financial Conglomerates are merged into "bank".
# ---------------------------------------------------------------------------

_FACTSET_TO_SECTOR: dict[str, str] = {
    "Major Banks":                 "bank",
    "Regional Banks":              "bank",
    "Savings Banks":               "bank",
    "Financial Conglomerates":     "bank",
    "Finance/Rental/Leasing":      "bank",
    "Life/Health Insurance":       "insurance",
    "Property/Casualty Insurance": "insurance",
    "Multi-Line Insurance":        "insurance",
    "Specialty Insurance":         "insurance",
    "Managed Health Care":         "insurance",
}

_FACTSET_TO_INDUSTRY: dict[str, str] = {
    "Major Banks":                 "bank",
    "Regional Banks":              "bank",
    "Savings Banks":               "bank",
    "Financial Conglomerates":     "bank",
    "Finance/Rental/Leasing":      "bank",
    "Life/Health Insurance":       "insurance_life",
    "Property/Casualty Insurance": "insurance_pc",
    "Multi-Line Insurance":        "insurance_life",
    "Specialty Insurance":         "insurance_pc",
    "Managed Health Care":         "insurance_life",
}

_GRANULARITY_MAP = {
    "sector":   _FACTSET_TO_SECTOR,
    "industry": _FACTSET_TO_INDUSTRY,
}

# GICS industry names that define the tradeable sector (banks + insurance)
_GICS_BANKS     = "Banks"
_GICS_INSURANCE = "Insurance"

# Extended-fundamentals columns that confirm sector membership
_BANK_METRIC      = "net_interest_margin"
_INSURANCE_METRIC = "insurance_premium_ltm"


def tradable_ids(raw, cfg):
    """Tradeable-universe ``stock_id`` set (banks + insurance), from reference tables.

    Same membership gate as :func:`sector_set`, but evaluated on the small
    reference / fundamentals tables (48k security rows, ~8.5M fundamental rows)
    rather than the 150M-row price panel.

    Returns a lazy ``[stock_id]`` frame of the tradeable securities.
    """
    ucfg = cfg["universe"]
    gics_keep = []
    if ucfg.get("include_banks", True):
        gics_keep.append(_GICS_BANKS)
    if ucfg.get("include_insurance", True):
        gics_keep.append(_GICS_INSURANCE)

    known_factset = list(_FACTSET_TO_SECTOR.keys())
    bank_factset = [k for k, v in _FACTSET_TO_SECTOR.items() if v == "bank"]
    insurance_factset = [k for k, v in _FACTSET_TO_SECTOR.items() if v == "insurance"]

    # "metric ever populated" per security, straight off the fundamentals tables.
    has_bank = (
        raw["fundamental_master_extended"]
        .group_by("stock_id")
        .agg(_has_bank=pl.col(_BANK_METRIC).is_not_null().any())
    )
    has_ins = (
        raw["fundamental_master"]
        .group_by("stock_id")
        .agg(_has_ins=pl.col(_INSURANCE_METRIC).is_not_null().any())
    )

    ref = (
        raw["security_master"].select("stock_id", "gics_industry_name")
        .join(raw["industry_mapping"].select("stock_id", "factset_industry_name"),
              on="stock_id", how="left")
        .join(has_bank, on="stock_id", how="left")
        .join(has_ins, on="stock_id", how="left")
    )

    keep = (
        pl.col("gics_industry_name").is_in(gics_keep)
        & pl.col("factset_industry_name").is_in(known_factset)
        & (
            (pl.col("factset_industry_name").is_in(bank_factset)
             & pl.col("_has_bank").fill_null(False))
            | (pl.col("factset_industry_name").is_in(insurance_factset)
               & pl.col("_has_ins").fill_null(False))
        )
    )
    return ref.filter(keep).select("stock_id").unique()


def market_set(panel, cfg):
    """All securities used for estimating market/country/industry factors."""
    return panel


def sector_set(panel, cfg):
    """Securities restricted to banks and insurance (the tradable universe).

    Filters by GICS industry name; respects include_banks / include_insurance
    flags from config. Noisy cross-classified securities are gated by requiring
    a FactSet label present in the known mappings. Companies are further required
    to have their corresponding sector metric populated in the extended
    fundamentals (net_interest_margin for banks, insurance_premium_ltm for
    insurance), ensuring only operationally active entities are included.
    """
    ucfg = cfg["universe"]
    gics_keep = []
    if ucfg.get("include_banks", True):
        gics_keep.append(_GICS_BANKS)
    if ucfg.get("include_insurance", True):
        gics_keep.append(_GICS_INSURANCE)

    known_factset = set(_FACTSET_TO_SECTOR.keys())

    bank_factset      = {k for k, v in _FACTSET_TO_SECTOR.items() if v == "bank"}
    insurance_factset = {k for k, v in _FACTSET_TO_SECTOR.items() if v == "insurance"}

    return (
        panel
        .filter(pl.col("gics_industry_name").is_in(gics_keep))
        .filter(pl.col("factset_industry_name").is_in(known_factset))
        .filter(
            pl.when(pl.col("factset_industry_name").is_in(bank_factset))
            .then(pl.col(_BANK_METRIC).is_not_null().any().over("stock_id"))
            .when(pl.col("factset_industry_name").is_in(insurance_factset))
            .then(pl.col(_INSURANCE_METRIC).is_not_null().any().over("stock_id"))
            .otherwise(False)
        )
    )


def industry_labels(panel, cfg):
    """Tag each security with a canonical industry label.

    Adds an 'industry' column using FactSet industry names as the primary
    source. Granularity is controlled by cfg['universe']['industry_granularity']:
        sector   → "bank" | "insurance"
        industry → "bank" | "insurance_life" | "insurance_multiline" | "insurance_pc"
    Securities whose FactSet label is not in the mapping get null.
    """
    granularity = cfg["universe"].get("industry_granularity", "industry")
    mapping = _GRANULARITY_MAP[granularity]

    # ``industry`` is intentionally a plain String (not Categorical) due to downstream
    # string operations like str.contains()

    return panel.with_columns(
        pl.col("factset_industry_name")
        .cast(pl.Utf8)
        .replace_strict(mapping, default=None)
        .alias("industry")
    )
