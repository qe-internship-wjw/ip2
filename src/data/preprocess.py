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


def _value_columns(scores, by):
    """Numeric factor-score columns to regularize (excludes ids / the group key).

    Works on both eager and lazy frames.
    """
    schema = (
        scores.collect_schema()
        if isinstance(scores, pl.LazyFrame)
        else scores.schema
    )
    skip = {"stock_id", by}
    return [c for c, dtype in schema.items() if c not in skip and dtype.is_numeric()]


def winsorize(scores, limits, by=None):
    """Clip cross-sectional outliers to the given quantile limits.

    For each factor column, values below the ``limits[0]`` quantile or above the
    ``limits[1]`` quantile of the cross-section (the ``by`` group, default
    ``"date"``) are clipped to those bounds. Nulls are preserved (imputed later by
    :func:`fill_missing`). Uses ``.over(by)`` so no information crosses dates.
    """
    by = by or "date"
    lo, hi = float(limits[0]), float(limits[1])
    exprs = [
        pl.col(c)
        .clip(
            lower_bound=pl.col(c).quantile(lo).over(by),
            upper_bound=pl.col(c).quantile(hi).over(by),
        )
        .alias(c)
        for c in _value_columns(scores, by)
    ]
    return scores.with_columns(exprs)


def fill_missing(scores, by=None):
    """Impute missing factor scores with the cross-sectional median.

    The median of the same-date cross-section is a neutral fill that introduces no
    look-ahead (it never reaches across dates, unlike a forward-fill) and, once
    scores are standardized, sits at ~0 -- i.e. a missing stock takes no active
    bet on that factor.
    """
    by = by or "date"
    exprs = [
        pl.col(c).fill_null(pl.col(c).median().over(by)).alias(c)
        for c in _value_columns(scores, by)
    ]
    return scores.with_columns(exprs)


def standardize(scores, by=None):
    """Cross-sectional z-score per factor: the factor's z_k.

    Subtracts the cross-sectional mean and divides by the cross-sectional standard
    deviation within each ``by`` group. Degenerate cross-sections (zero or null
    dispersion) map to 0 rather than inf/nan.
    """
    by = by or "date"

    def zscore(c: str) -> pl.Expr:
        mean = pl.col(c).mean().over(by)
        std = pl.col(c).std().over(by)
        return (
            pl.when(std.is_null() | (std == 0))
            .then(pl.lit(0.0))
            .otherwise((pl.col(c) - mean) / std)
            .alias(c)
        )

    return scores.with_columns([zscore(c) for c in _value_columns(scores, by)])


def regularize(scores, cfg):
    """Run winsorize -> fill_missing -> standardize on raw factor scores.

    Reads ``preprocess.winsorize_limits`` and ``preprocess.group_by`` from config
    (defaults: ``[0.01, 0.99]`` and ``"date"``).
    """
    pcfg = cfg["preprocess"]
    limits = pcfg.get("winsorize_limits", [0.01, 0.99])
    by = pcfg.get("group_by", "date")

    scores = winsorize(scores, limits, by=by)
    scores = fill_missing(scores, by=by)
    scores = standardize(scores, by=by)
    return scores
