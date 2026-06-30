"""Data quality checks on the joined panel.

Two passes over the ~150M-row panel:
  Pass 1 - single scan: shape, null-coverage rates, price/mcap sanity, return stats.
  Pass 2 - groupby:     duplicate (stock_id, date) primary-key check.

All intermediate joins stay lazy; only the final aggregated scalars are collected.
"""

from __future__ import annotations

import polars as pl

from src.config import load
from src.data.loaders import load_all
from src.data.joins import build_panel


# (column, join-source label) pairs checked for null coverage
_COVERAGE_COLS: list[tuple[str, str]] = [
    ("country_code",     "security_master"),
    ("currency_code",    "security_master"),
    ("fx_to_usd",        "fx_rates"),
    ("risk_free_rate",   "risk_free_rate"),
    ("region_code",      "country_mapping"),
    ("gics_sector_name", "security_master / GICS (~54% expected)"),
    ("sic_code",         "industry_mapping"),
    ("sales_ltm",        "fundamental_master (asof)"),
    ("excess_return",    "computed: total_return - risk_free_rate"),
]


def run(panel_lf: pl.LazyFrame) -> None:
    """Run quality checks and print a formatted report to stdout."""
    if not isinstance(panel_lf, pl.LazyFrame):
        panel_lf = panel_lf.lazy()

    # ------------------------------------------------------------------ #
    # Pass 1 – single full scan                                           #
    # ------------------------------------------------------------------ #
    exprs: list[pl.Expr] = [
        # shape
        pl.len().alias("n_rows"),
        pl.col("date").min().alias("date_min"),
        pl.col("date").max().alias("date_max"),
        pl.col("stock_id").n_unique().alias("n_securities"),
        pl.col("date").n_unique().alias("n_dates"),

        # join coverage: fraction of rows where the joined column is null
        *[
            pl.col(col).is_null().mean().alias(f"null_pct_{col}")
            for col, _ in _COVERAGE_COLS
        ],

        # price / market-cap sanity
        (pl.col("price_local") <= 0).sum().alias("n_nonpositive_price"),
        (pl.col("entity_mcap_local") <= 0).sum().alias("n_nonpositive_mcap"),
        (pl.col("volume") < 0).sum().alias("n_negative_volume"),

        # total_return distribution
        pl.col("total_return").min().alias("tr_min"),
        pl.col("total_return").max().alias("tr_max"),
        pl.col("total_return").mean().alias("tr_mean"),
        pl.col("total_return").is_null().sum().alias("tr_null"),
        (pl.col("total_return") == 0.0).sum().alias("tr_zero"),
        (pl.col("total_return").abs() > 0.50).sum().alias("tr_extreme_50pct"),

        # excess_return distribution
        pl.col("excess_return").min().alias("er_min"),
        pl.col("excess_return").max().alias("er_max"),
        pl.col("excess_return").mean().alias("er_mean"),
        pl.col("excess_return").is_null().sum().alias("er_null"),
    ]

    stats = panel_lf.select(exprs).collect()
    row = stats.row(0, named=True)
    n: int = row["n_rows"]

    # ------------------------------------------------------------------ #
    # Pass 2 – duplicate primary-key check                                #
    # ------------------------------------------------------------------ #
    dup_frame = (
        panel_lf
        .group_by(["stock_id", "date"])
        .agg(pl.len().alias("cnt"))
        .filter(pl.col("cnt") > 1)
    ).collect()

    n_dup_keys: int = len(dup_frame)
    n_dup_rows: int = int(dup_frame["cnt"].sum()) if n_dup_keys > 0 else 0

    # ------------------------------------------------------------------ #
    # Report                                                               #
    # ------------------------------------------------------------------ #
    sep = "=" * 72

    print(sep)
    print("  PANEL DATA QUALITY REPORT")
    print(sep)

    print("\n  [Shape]")
    print(f"    Rows        : {n:>15,}")
    print(f"    Securities  : {row['n_securities']:>15,}")
    print(f"    Dates       : {row['n_dates']:>15,}")
    print(f"    Date range  : {row['date_min']}  to  {row['date_max']}")

    print("\n  [Join Coverage - null rates]")
    for col, source in _COVERAGE_COLS:
        pct = row[f"null_pct_{col}"] * 100
        warn = "  <-- WARNING" if pct > 20 and "expected" not in source else ""
        print(f"    {col:<26} {pct:6.2f}%  [{source}]{warn}")

    print("\n  [Price / Market-Cap Sanity]")
    for label, key in [
        ("Non-positive price_local",   "n_nonpositive_price"),
        ("Non-positive entity_mcap",   "n_nonpositive_mcap"),
        ("Negative volume",            "n_negative_volume"),
    ]:
        v: int = row[key]
        print(f"    {label:<30} {v:>12,}  ({v / n * 100:.3f}%)")

    print("\n  [Return Distribution]")
    print(f"    total_return  min / mean / max : "
          f"{row['tr_min']:>10.4f} / {row['tr_mean']:>10.6f} / {row['tr_max']:>10.4f}")
    print(f"    excess_return min / mean / max : "
          f"{row['er_min']:>10.4f} / {row['er_mean']:>10.6f} / {row['er_max']:>10.4f}")
    print(f"    Null total_return              : "
          f"{row['tr_null']:>12,}  ({row['tr_null'] / n * 100:.3f}%)")
    print(f"    Zero total_return              : "
          f"{row['tr_zero']:>12,}  ({row['tr_zero'] / n * 100:.3f}%)")
    print(f"    Null excess_return             : "
          f"{row['er_null']:>12,}  ({row['er_null'] / n * 100:.3f}%)")
    print(f"    |total_return| > 50%           : "
          f"{row['tr_extreme_50pct']:>12,}  ({row['tr_extreme_50pct'] / n * 100:.3f}%)")

    print("\n  [Primary Key Integrity - (stock_id, date)]")
    if n_dup_keys == 0:
        print("    No duplicate (stock_id, date) pairs found.")
    else:
        print(f"    Duplicate key pairs  : {n_dup_keys:>12,}  <-- WARNING")
        print(f"    Excess rows          : {n_dup_rows - n_dup_keys:>12,}  <-- WARNING")

    print(sep)


if __name__ == "__main__":
    cfg = load()
    raw = load_all(cfg)
    panel_lf = build_panel(raw, cfg)
    run(panel_lf)
