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

    The raw vendor feed carries physically-impossible returns -- both directly
    (a ``total_return`` of +1,999,900%) and latently, as vanishingly small
    "prices" (a near-delisting / relisting stub of a fraction of a cent) that
    manufacture thousand-percent moves the moment they anchor a ratio
    (RETURNS_DATA_QUALITY.md). Left alone they flow into ``ret_raw``, the
    cap-weighted market series, and the return-derived style factors
    (volatility, momentum), so the scrub operates on ``total_return`` itself --
    upstream of every one of them.

    Steps applied in order:
      1. **Price-validity gate.** A mark below ``preprocess.min_price_usd``
         (converted to USD via ``fx_to_usd``), non-positive, or non-finite is a
         vendor stub, not a tradeable price. It is blanked so it can never
         anchor a return -- neither as the current mark nor as a later row's
         previous mark -- which neutralises a near-zero round-trip (stub → back)
         end to end.
      2. **Impute null total_return** across *trustworthy* fills only:
           - the vendor ``price_return`` when present (never null with a null
             ``total_return`` in practice, but the correct preference);
           - else ``price_local / prev_price − 1``, but only when both marks are
             valid (step 1) and no more than ``preprocess.max_impute_gap_days``
             apart -- a wider gap spans a halt / relisting / unadjusted action
             whose raw ratio is not a one-day return;
           - else 0 (stale price, untrustworthy gap, or no usable price).
      3. **Return-validity gate.** After imputation, a return that spans a
         blanked stub, is non-finite, or exceeds ``preprocess.max_abs_daily_return``
         in magnitude is a data error: null it and book the day flat (``0``).
         This is a validity gate, not a winsorization -- it removes impossible
         values while leaving legitimate tails (incl. the −100% delisting
         settlement, applied downstream) intact.
      4. Drop rows where risk_free_rate is null (SG/KR/TW before their
         rate series begins; ~2% of rows, ~3.1M).
      5. Recompute excess_return from the now-clean total_return.

    ``_px_blanked`` / ``_ret_gated`` boolean columns ride along so a caller
    (``scripts.build_processed``) can log how many rows each gate touched; they
    are transient and ignored by every downstream consumer.
    """
    pcfg = cfg.get("preprocess", {}) or {}
    min_price_usd = pcfg.get("min_price_usd")
    max_gap_days = pcfg.get("max_impute_gap_days")
    max_abs_ret = pcfg.get("max_abs_daily_return")

    # Sort required for shift(1).over() to produce the chronological lag.
    panel = panel.sort(["stock_id", "date"])

    # ── 1. Price-validity gate (USD floor + basic sanity) ─────────────────
    valid_price = (
        pl.col("price_local").is_not_null()
        & pl.col("price_local").is_finite()
        & (pl.col("price_local") > 0.0)
    )
    if min_price_usd is not None:
        price_usd = pl.col("price_local") * pl.col("fx_to_usd")
        # A null price_usd means fx is missing, not that the price is tiny -- keep
        # the basic-sanity verdict and let the USD floor apply only when convertible.
        valid_price = valid_price & (
            price_usd.is_null() | (price_usd >= float(min_price_usd))
        )
    panel = panel.with_columns(
        _orig_tr_nonnull=pl.col("total_return").is_not_null(),
        _valid_price=valid_price,
    )
    # The validated mark (null on a stub) plus its lagged value / date / validity.
    # The lag is null exactly when the previous mark was itself a stub, so a stub
    # can never sit on either end of an imputed one-day return.
    panel = panel.with_columns(
        _px=pl.when("_valid_price").then(pl.col("price_local")).otherwise(None),
    ).with_columns(
        _prev_px=pl.col("_px").shift(1).over("stock_id"),
        _prev_date=pl.col("date").shift(1).over("stock_id"),
        # First row has no predecessor to span, so treat its "previous" as valid.
        _prev_valid=pl.col("_valid_price").shift(1).over("stock_id").fill_null(True),
    )

    # ── 2. Impute null total_return across trustworthy fills only ─────────
    within_gap = pl.lit(True)
    if max_gap_days is not None:
        within_gap = (
            pl.col("date") - pl.col("_prev_date")
        ).dt.total_days() <= int(max_gap_days)
    can_impute = (
        pl.col("_px").is_not_null()
        & pl.col("_prev_px").is_not_null()
        & (pl.col("_px") != pl.col("_prev_px"))
        & within_gap
    )
    panel = panel.with_columns(
        pl.when(pl.col("total_return").is_null() & pl.col("price_return").is_not_null())
        .then(pl.col("price_return"))
        .when(pl.col("total_return").is_null() & can_impute)
        .then(pl.col("_px") / pl.col("_prev_px") - 1.0)
        .when(pl.col("total_return").is_null())
        .then(pl.lit(0.0))
        .otherwise(pl.col("total_return"))
        .alias("total_return")
    )

    # ── 3. Return-validity gate: null impossible values, book them flat ───
    tr = pl.col("total_return")
    is_bad = (~pl.col("_valid_price")) | (~pl.col("_prev_valid"))  # spans a stub
    is_bad = is_bad | (tr.is_not_null() & ~tr.is_finite())         # NaN / inf
    if max_abs_ret is not None:
        is_bad = is_bad | (tr.is_not_null() & (tr.abs() > float(max_abs_ret)))
    panel = panel.with_columns(
        pl.when(is_bad).then(None).otherwise(tr).fill_null(0.0).alias("total_return"),
        _px_blanked=~pl.col("_valid_price"),
        _ret_gated=pl.col("_orig_tr_nonnull") & is_bad,
    ).drop("_valid_price", "_orig_tr_nonnull", "_px", "_prev_px", "_prev_date", "_prev_valid")

    # ── 4. Remove rows with no risk-free rate ─────────────────────────────
    panel = panel.filter(pl.col("risk_free_rate").is_not_null())

    # ── 5. Recompute excess_return with the cleaned total_return ──────────
    panel = panel.with_columns(
        excess_return=(pl.col("total_return") - pl.col("risk_free_rate"))
    )

    return panel


def rebalance_grid(panel, cfg, period_months=None):
    """Per (security, rebalancing period) the period-end trading date + period key.

    Buckets each security's trading days into non-overlapping calendar periods of
    ``backtest.rebalancing_frequency_months`` (quarterly by default) and keeps the
    last trading day of each -- the rebalance date. Returns a lazy ``[stock_id,
    date, period]`` frame: the grid used by :func:`to_rebalance` to downsample the
    daily style scores and the monthly loadings to the rebalancing cross-sections.

    ``period_months`` overrides the config frequency -- the risk model estimates
    its factor returns on **monthly** cross-sections (``period_months=1``) while
    the rebalance itself stays quarterly (see PORTFOLIO_PLAN.md §2).
    """
    pm = int(period_months or cfg.get("backtest", {}).get("rebalancing_frequency_months", 3))
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


def _winsorize_expr(col: str, limits, by: str, mask=None) -> pl.Expr:
    """Clip ``col`` to its ``[lo, hi]`` cross-sectional quantiles within each ``by`` group.

    When ``mask`` is given the quantile bounds are computed only over masked rows and
    rows outside the mask are nulled (the sub-universe restriction :func:`winsorize`
    needs); otherwise the bounds span the whole ``by`` cross-section. ``.over(by)`` so
    no information crosses the group boundary.
    """
    lo, hi = float(limits[0]), float(limits[1])
    src = pl.col(col).filter(mask) if mask is not None else pl.col(col)
    clipped = pl.col(col).clip(
        lower_bound=src.quantile(lo).over(by), upper_bound=src.quantile(hi).over(by)
    )
    if mask is None:
        return clipped.alias(col)
    return pl.when(mask).then(clipped).otherwise(None).alias(col)


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
    factors = _factor_columns(scores)
    _require_universe_col(scores, factors, universe_col)
    return scores.with_columns(
        [_winsorize_expr(c, limits, by, _subuniverse_mask(c, universe_col)) for c in factors]
    )


def winsorize_cross_section(frame, cols, limits, by="date"):
    """Clip each of ``cols`` to its cross-sectional ``[lo, hi]`` quantiles per ``by`` group.

    A sub-universe-agnostic companion to :func:`winsorize` for **non-factor** series
    such as forward returns: a return carries no factor/sub-universe label, so its
    bounds are taken over the whole ``by`` cross-section. Feed it the same
    ``preprocess.winsorize_limits`` the factor winsorizer uses. Accepts eager or lazy
    frames and leaves nulls untouched.
    """
    return frame.with_columns([_winsorize_expr(c, limits, by, None) for c in cols])


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
