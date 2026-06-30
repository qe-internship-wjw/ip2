"""
Regularization: cleaning, winsorization, standardization.
"""

from __future__ import annotations

import polars as pl


def clean(panel, cfg):
    """Repair null and erroneous raw values prior to factor generation.

    Steps applied in order:
      1. Impute null total_return:
           - genuine price change (corporate action) → price_local / prev_price_local - 1
           - stale price or no price data            → 0
      2. Drop rows where risk_free_rate is null (SG/KR/TW before their
         rate series begins; ~2% of rows, ~3.1M).
      3. Recompute excess_return from the now-complete total_return.
    """
    # Sort required for shift(1).over() to produce the chronological lag.
    panel = panel.sort(["stock_id", "date"])

    # ── 1. Impute null total_return ───────────────────────────────────────
    panel = (
        panel
        .with_columns(
            pl.col("price_local").shift(1).over("stock_id").alias("_prev_price")
        )
        .with_columns(
            pl.when(
                pl.col("total_return").is_null()
                & pl.col("price_local").is_not_null()
                & pl.col("_prev_price").is_not_null()
                & (pl.col("price_local") != pl.col("_prev_price"))
            )
            .then(pl.col("price_local") / pl.col("_prev_price") - 1.0)
            .when(pl.col("total_return").is_null())
            .then(pl.lit(0.0))
            .otherwise(pl.col("total_return"))
            .alias("total_return")
        )
        .drop("_prev_price")
    )

    # ── 2. Remove rows with no risk-free rate ─────────────────────────────
    panel = panel.filter(pl.col("risk_free_rate").is_not_null())

    # ── 3. Recompute excess_return with the filled total_return ───────────
    panel = panel.with_columns(
        excess_return=(pl.col("total_return") - pl.col("risk_free_rate"))
    )

    return panel


def winsorize(scores, limits, by=None):
    """Clip cross-sectional outliers to the given quantile limits."""
    raise NotImplementedError


def fill_missing(scores, by=None):
    """Fill missing factor scores (e.g. cross-sectional median)."""
    raise NotImplementedError


def standardize(scores, by=None):
    """Cross-sectional z-score (optionally within group), the factor's z_k."""
    raise NotImplementedError


def regularize(scores, cfg):
    """Run winsorize -> fill_missing -> standardize on raw factor scores."""
    raise NotImplementedError
