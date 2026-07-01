"""Shared machinery for the hierarchical structural factors.

``market``, ``country`` and ``industry`` build on a tree of stock groupings:

* a *node* series is the cap-weighted (USD-sized) excess return of its members
  (:func:`cap_weighted_return`), and
* a *factor return* is a node's return minus its parent's return
  (:func:`relative_factor`) -- so the children of any node sum to zero when
  weighted by USD market cap.

    market                    root, absolute                 (Market)
    +- region / country       relative to its parent         (Country)
    +- industry               relative to the tradeable set  (Industry)

A security's *exposure* to a structural factor is its rolling time-series beta on
that factor's return (:func:`stock_loadings`), so every non-style factor exposes a
per-security ``[stock_id, date, <shorthand>]`` loading -- the same shape as a
style score, which lets the pipeline treat all factors uniformly.
"""

from __future__ import annotations

import polars as pl

_YEAR = 252  # trading days


def cap_weighted_return(lf, keys):
    """Cap-weighted excess return per ``keys`` group.

    Returns a lazy frame with ``keys`` plus the split weighted numerator ``_w``
    (= sum of ``excess_return * mcap_usd``) and denominator ``_m`` (= sum of
    ``mcap_usd``) and their ratio ``_r``. The pieces are kept so a coarser
    (parent) return re-aggregates from the very same sums.
    """
    return (
        lf.group_by(keys)
        .agg(
            _w=(pl.col("excess_return") * pl.col("mcap_usd")).sum(),
            _m=pl.col("mcap_usd").sum(),
        )
        .with_columns(_r=pl.col("_w") / pl.col("_m"))
    )


def relative_factor(lf, child, parent_keys=()):
    """Parent-neutral cap-weighted excess return per ``child`` group.

    Each ``child`` group's cap-weighted excess return is demeaned against the
    cap-weighted return over its parent scope (``["date", *parent_keys]``), so
    the children within each parent sum to zero when weighted by USD market cap.
    ``parent_keys`` empty means the parent is the whole market.

    Returns a long lazy frame with columns ``date``, ``child`` and ``_factor``.
    """
    parent_keys = list(parent_keys)
    base_keys = ["date", *parent_keys]

    child_ret = cap_weighted_return(lf, [*base_keys, child])
    # Re-aggregate the parent return from the child sums (consistent + cheap).
    base = (
        child_ret.group_by(base_keys)
        .agg(_wb=pl.col("_w").sum(), _mb=pl.col("_m").sum())
        .with_columns(_r_base=pl.col("_wb") / pl.col("_mb"))
    )
    return (
        child_ret.join(base.select(*base_keys, "_r_base"), on=base_keys, how="left")
        .with_columns(_factor=pl.col("_r") - pl.col("_r_base"))
        .select("date", child, "_factor")
    )


def _loading_window(cfg) -> tuple[int, int]:
    """Rolling-beta window / min-periods (trading days) from ``factors.loadings``."""
    months = cfg["factors"].get("loadings", {}).get("window_months", 24)
    window = max(_YEAR, round(months / 12 * _YEAR))
    return window, window // 2


def stock_loadings(panel, factor_long, name, cfg, join_on):
    """Per-security rolling beta of ``excess_return`` on its own factor return.

    ``factor_long`` carries a factor-mimicking portfolio return keyed by
    ``join_on`` (``["date"]`` for the market, ``["date", <member>]`` for
    country/industry). It is aligned to each ``(stock, date)`` and a rolling
    time-series OLS gives the security's loading, exposed (eagerly) as ``name``.
    """
    import polars_ols  # noqa: F401 -- registers the `.least_squares` namespace

    lf = panel.lazy() if isinstance(panel, pl.DataFrame) else panel
    window, min_periods = _loading_window(cfg)
    keys = [k for k in join_on if k != "date"]

    aligned = (
        lf.select("stock_id", "date", "excess_return", *keys)
        .join(factor_long.rename({"_factor": "_x"}), on=join_on, how="left")
        .sort("stock_id", "date")
    )
    beta = (
        pl.col("excess_return")
        .least_squares.rolling_ols(
            "_x", window_size=window, min_periods=min_periods,
            add_intercept=True, mode="coefficients", null_policy="drop",
        )
        .over("stock_id")
        .struct.field("_x")
    )
    return aligned.select(
        "stock_id", "date",
        pl.when(beta.is_finite()).then(beta).otherwise(None).alias(name),
    ).collect()
