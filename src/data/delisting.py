"""Delisting events and point-in-time (PIT) tradeability windows.

The raw panel carries **no delisting flag and no delisting return** -- a security that
leaves simply stops emitting rows, and how the record ends (silent stop, null-price
tail, floor sentinel) is a country/vendor encoding convention, not a statement about
*why* the stock left (see DELISTING_HANDLING.md). Detection therefore keys on the
**price path**, which is currency-agnostic:

1. Trim the zombie/stale tail: an *active* row has ``volume > 0`` or a non-zero
   ``price_return``; ``last_active_date`` is the last active row's date.
2. Right-censoring guard: a name still active within ``censor_buffer_days`` of the
   panel end is alive, not delisted -- it emits no event.
3. Classify the exit: **wipeout** when the exit price sits at or below
   ``drawdown_threshold`` x the trailing ``peak_window_days``-active-row peak, or when
   the auxiliary floor test ``min(price_local) <= floor_price`` fires (the floor
   sentinel may live in the post-active zombie tail, so the min spans the full
   record); otherwise **acquisition-like**.
4. Terminal return: ``backtest.delisting_return`` (~ -1.0) for wipeouts; ``0.0``
   (settle at the last real print) for acquisitions.

Build the events from the *same* price window the analysis runs on, so censoring at
the window edge stays consistent. Thresholds live under ``backtest.delisting`` in
config.yaml and should be swept during validation, not treated as truths.
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl


def _knobs(cfg) -> dict:
    """Classifier thresholds from ``backtest`` config, defaulted per DELISTING_HANDLING.md."""
    section = cfg.get("backtest", {}) or {}
    dl = section.get("delisting", {}) or {}
    return {
        "drawdown_threshold": float(dl.get("drawdown_threshold", 0.30)),
        "peak_window_days": int(dl.get("peak_window_days", 252)),
        "floor_price": float(dl.get("floor_price", 1e-5)),
        "censor_buffer_days": int(dl.get("censor_buffer_days", 15)),
        "delisting_return": float(section.get("delisting_return", -1.0)),
    }


def delist_events(price, cfg) -> pl.DataFrame:
    """Point-in-time delisting-event table: one row per security that left the panel.

    Parameters
    ----------
    price : eager or lazy frame with ``stock_id, date, price_local, price_return,
        volume`` -- the price table, pre-filtered by the caller to the universe and
        analysis window (censoring is judged against this frame's own last date).
    cfg : Config (reads ``backtest.delisting_return`` and ``backtest.delisting.*``).

    Returns
    -------
    pl.DataFrame
        ``[stock_id, last_active_date, delist_date, reason, delist_return]`` with
        ``reason in {"wipeout", "acquisition"}`` and ``delist_date =
        last_active_date`` -- the last day the position could be marked at a real
        print; weights are forced to zero after it. Right-censored (still-alive)
        names emit no row.
    """
    k = _knobs(cfg)
    lf = price.lazy().select("stock_id", "date", "price_local", "price_return", "volume")

    panel_end = lf.select(pl.col("date").max()).collect().item()
    if panel_end is None:
        raise ValueError("delist_events: the price frame has no rows.")
    cutoff = panel_end - timedelta(days=k["censor_buffer_days"])

    # A row is *active* when the stock really traded: positive volume or a non-zero
    # price move. Flat/null carry-forward (zombie) tails fail both tests.
    is_active = (pl.col("volume") > 0) | (
        pl.col("price_return").is_not_null() & (pl.col("price_return") != 0.0)
    )

    active = (
        lf.filter(is_active)
        .group_by("stock_id")
        .agg(
            last_active_date=pl.col("date").max(),
            # Trailing peak over the last `peak_window_days` *active* rows (row count,
            # not calendar), exit row included.
            peak_price=pl.col("price_local")
            .sort_by("date")
            .tail(k["peak_window_days"])
            .max(),
        )
    )

    exit_price = (
        lf.join(active.select("stock_id", "last_active_date"), on="stock_id", how="inner")
        .filter(pl.col("date") <= pl.col("last_active_date"))
        .group_by("stock_id")
        .agg(exit_price=pl.col("price_local").sort_by("date").drop_nulls().last())
    )

    # The floor sentinel may be written into the post-active zombie tail, so the min
    # deliberately spans the full record, not just rows up to `last_active_date`.
    floor = lf.group_by("stock_id").agg(min_price=pl.col("price_local").min())

    is_wipeout = (
        pl.col("exit_price") <= k["drawdown_threshold"] * pl.col("peak_price")
    ) | (pl.col("min_price") <= k["floor_price"])

    return (
        active.join(exit_price, on="stock_id", how="left")
        .join(floor, on="stock_id", how="left")
        .filter(pl.col("last_active_date") < cutoff)
        .with_columns(
            reason=pl.when(is_wipeout)
            .then(pl.lit("wipeout"))
            .otherwise(pl.lit("acquisition"))
        )
        .with_columns(
            delist_date=pl.col("last_active_date"),
            delist_return=pl.when(pl.col("reason") == "wipeout")
            .then(k["delisting_return"])
            .otherwise(0.0),
        )
        .select("stock_id", "last_active_date", "delist_date", "reason", "delist_return")
        .sort("stock_id")
        .collect()
    )


def tradeability_window(security_master, events):
    """PIT tradeability window per security: lazy ``[stock_id, start_date, end_date]``.

    ``start_date`` = ``security_master.stock_start_date`` (not investable before);
    ``end_date`` = the inferred ``delist_date`` (null while listed / right-censored).
    A security belongs in a rebalance-``t`` cross-section iff
    ``start_date <= t <= end_date`` (:func:`pit_filter`): delisted names stay in
    history over their *live* window -- which is what kills survivorship bias --
    while never entering before listing or after removal. Apply the same window to
    the risk-model estimation universe, not just the investable set.
    """
    sm = security_master.lazy().select(
        "stock_id", pl.col("stock_start_date").cast(pl.Date).alias("start_date")
    )
    ev = events.lazy().select("stock_id", pl.col("delist_date").alias("end_date"))
    return sm.join(ev, on="stock_id", how="left")


def pit_filter(frame, window, date_col="date"):
    """Restrict ``frame`` to each security's PIT tradeability window.

    Rows with ``date_col`` outside ``[start_date, end_date]`` are dropped. A null
    bound is unbounded on that side, and a security absent from ``window`` is kept
    unchanged (no information to gate on). Accepts and returns eager or lazy,
    matching the input's laziness.
    """
    lazy_in = isinstance(frame, pl.LazyFrame)
    joined = frame.lazy().join(window.lazy(), on="stock_id", how="left")
    keep = (
        pl.col("start_date").is_null() | (pl.col(date_col) >= pl.col("start_date"))
    ) & (pl.col("end_date").is_null() | (pl.col(date_col) <= pl.col("end_date")))
    out = joined.filter(keep).drop("start_date", "end_date")
    return out if lazy_in else out.collect()
