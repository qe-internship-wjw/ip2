"""
Regularization: cleaning, winsorization, standardization.

Winsorization / imputation / standardization are **sub-universe aware**. Banks and
insurers carry mutually-exclusive (and several jointly-computable) metrics.
"""

from __future__ import annotations

import polars as pl

from ..factors.base import Applicability, registry

# Sector applicability -> the ``industry``-label prefix that selects its
# sub-universe (``universe.industry_labels``: "bank", "insurance", "insurance_*").
# All-financials factors are absent here: they use the whole cross-section.
_APPLICABILITY_PREFIX = {
    Applicability.BANKS: "bank",
    Applicability.INSURANCE: "insurance",
}


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


def rebalance_grid(panel, cfg):
    """Per (security, rebalancing period) the period-end trading date + period key.

    Buckets each security's trading days into non-overlapping calendar periods of
    ``backtest.rebalancing_frequency_months`` (quarterly by default) and keeps the
    last trading day of each -- the rebalance date. Returns a lazy ``[stock_id,
    date, period]`` frame: the grid used by :func:`to_rebalance` to downsample the
    daily style scores and the monthly loadings to the rebalancing cross-sections.
    """
    pm = int(cfg.get("backtest", {}).get("rebalancing_frequency_months", 3))
    lf = panel if isinstance(panel, pl.LazyFrame) else panel.lazy()
    return (
        lf.select("stock_id", "date")
        .with_columns(period=pl.col("date").dt.truncate(f"{pm}mo"))
        .group_by("stock_id", "period")
        .agg(date=pl.col("date").max())
        .select("stock_id", "date", "period")
    )


def to_rebalance(frame, grid):
    """Downsample a ``[stock_id, date, ...]`` frame to the rebalance cross-sections.

    Inner-joins ``frame`` to :func:`rebalance_grid` on ``[stock_id, date]`` so only
    period-end rows survive, and attaches the common ``period`` key. Grouping the
    downstream cross-sectional steps (regularize / neutralize) on ``period`` -- not
    the raw ``date`` -- keeps every security of a rebalancing period in one
    cross-section despite staggered trading calendars. Returns a lazy frame.
    """
    lf = frame if isinstance(frame, pl.LazyFrame) else frame.lazy()
    g = grid if isinstance(grid, pl.LazyFrame) else grid.lazy()
    return lf.join(g, on=["stock_id", "date"], how="inner")


def _columns(scores):
    """Column names of an eager or lazy frame."""
    return (
        scores.collect_schema().names()
        if isinstance(scores, pl.LazyFrame)
        else scores.columns
    )


def _factor_columns(scores):
    """Registered factor-score columns present in ``scores`` (keyed by shorthand).

    Cross-references the frame's columns against the factor registry rather than
    duck-typing numeric columns, so metadata (``mcap_usd``, the ``industry``
    label, ...) is never regularized as if it were a signal.
    """
    known = registry()
    return [c for c in _columns(scores) if c in known]


def _subuniverse_mask(col: str, universe_col: str):
    """Boolean expr for the sub-universe factor ``col`` is defined on, or ``None``.

    ``None`` means the factor is all-financials and uses the whole cross-section.
    """
    prefix = _APPLICABILITY_PREFIX.get(registry()[col].applicability)
    return None if prefix is None else pl.col(universe_col).str.starts_with(prefix)


def _require_universe_col(scores, factors, universe_col):
    """Ensure the membership label is present when any sector factor needs it."""
    needs_split = any(
        registry()[c].applicability in _APPLICABILITY_PREFIX for c in factors
    )
    if needs_split and universe_col not in _columns(scores):
        raise ValueError(
            f"preprocess needs the '{universe_col}' sub-universe label column to "
            "partition Bank/Insurer factors; otherwise sector medians and z-scores "
            "leak across the structurally-disjoint sub-universes. Attach it via "
            "universe.industry_labels before regularizing."
        )


def winsorize(scores, limits, by=None, universe_col="industry"):
    """Clip cross-sectional outliers to the given quantile limits.

    For each factor column, values below the ``limits[0]`` quantile or above the
    ``limits[1]`` quantile of its cross-section (the ``by`` group, default
    ``"date"``) are clipped to those bounds. The cross-section is restricted to the
    factor's sub-universe (bank / insurer / all-financials): the quantiles come
    only from applicable rows, and rows outside the sub-universe are set to null.
    Nulls are preserved (imputed later by :func:`fill_missing`). Uses ``.over(by)``
    so no information crosses dates.
    """
    by = by or "date"
    lo, hi = float(limits[0]), float(limits[1])
    factors = _factor_columns(scores)
    _require_universe_col(scores, factors, universe_col)

    def clip_expr(c: str) -> pl.Expr:
        mask = _subuniverse_mask(c, universe_col)
        if mask is None:
            lob = pl.col(c).quantile(lo).over(by)
            hib = pl.col(c).quantile(hi).over(by)
            return pl.col(c).clip(lower_bound=lob, upper_bound=hib).alias(c)
        lob = pl.col(c).filter(mask).quantile(lo).over(by)
        hib = pl.col(c).filter(mask).quantile(hi).over(by)
        clipped = pl.col(c).clip(lower_bound=lob, upper_bound=hib)
        return pl.when(mask).then(clipped).otherwise(None).alias(c)

    return scores.with_columns([clip_expr(c) for c in factors])


def fill_missing(scores, by=None, universe_col="industry"):
    """Impute missing factor scores with the cross-sectional median.

    The median of the same-date cross-section is a neutral fill that introduces no
    look-ahead (it never reaches across dates, unlike a forward-fill) and, once
    scores are standardized, sits at ~0 -- i.e. a missing stock takes no active
    bet on that factor. The cross-section is restricted to the factor's
    sub-universe, so a Bank's missing Bank-factor is filled with the *Bank* median
    (never the Insurer median); rows outside the sub-universe stay null.
    """
    by = by or "date"
    factors = _factor_columns(scores)
    _require_universe_col(scores, factors, universe_col)

    def fill_expr(c: str) -> pl.Expr:
        mask = _subuniverse_mask(c, universe_col)
        if mask is None:
            return pl.col(c).fill_null(pl.col(c).median().over(by)).alias(c)
        median = pl.col(c).filter(mask).median().over(by)
        return (
            pl.when(mask).then(pl.col(c).fill_null(median)).otherwise(None).alias(c)
        )

    return scores.with_columns([fill_expr(c) for c in factors])


def standardize(scores, by=None, universe_col="industry"):
    """Cross-sectional z-score per factor: the factor's z_k.

    Subtracts the cross-sectional mean and divides by the cross-sectional standard
    deviation within each ``by`` group, restricted to the factor's sub-universe so
    a sector factor is standardized only against its own sub-universe. Degenerate
    cross-sections (zero or null dispersion) map to 0 rather than inf/nan; rows
    outside the sub-universe stay null.
    """
    by = by or "date"
    factors = _factor_columns(scores)
    _require_universe_col(scores, factors, universe_col)

    def zscore(c: str) -> pl.Expr:
        mask = _subuniverse_mask(c, universe_col)
        if mask is None:
            mean = pl.col(c).mean().over(by)
            std = pl.col(c).std().over(by)
            return (
                pl.when(std.is_null() | (std == 0))
                .then(pl.lit(0.0))
                .otherwise((pl.col(c) - mean) / std)
                .alias(c)
            )
        mean = pl.col(c).filter(mask).mean().over(by)
        std = pl.col(c).filter(mask).std().over(by)
        core = (
            pl.when(std.is_null() | (std == 0))
            .then(pl.lit(0.0))
            .otherwise((pl.col(c) - mean) / std)
        )
        return pl.when(mask).then(core).otherwise(None).alias(c)

    return scores.with_columns([zscore(c) for c in factors])


def regularize(scores, cfg):
    """Run winsorize -> fill_missing -> standardize on raw factor scores.

    Reads ``preprocess.winsorize_limits``, ``preprocess.group_by`` and
    ``preprocess.universe_col`` from config (defaults: ``[0.01, 0.99]``, ``"date"``
    and ``"industry"``). The input must carry the ``universe_col`` sub-universe
    label whenever any bank/insurer factor is present (see :func:`fill_missing`).
    """
    pcfg = cfg["preprocess"]
    limits = pcfg.get("winsorize_limits", [0.01, 0.99])
    by = pcfg.get("group_by", "date")
    universe_col = pcfg.get("universe_col", "industry")

    scores = winsorize(scores, limits, by=by, universe_col=universe_col)
    scores = fill_missing(scores, by=by, universe_col=universe_col)
    scores = standardize(scores, by=by, universe_col=universe_col)
    return scores
