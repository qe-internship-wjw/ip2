"""Shared machinery for the hierarchical structural factors.

``market``, ``country`` and ``industry`` are all the same operation on a tree of
stock groupings:

* a *node* series is the cap-weighted (USD-sized) excess return of its members
  (:func:`cap_weighted_return`), and
* a *factor* is a node's return minus its parent's return
  (:func:`relative_factor`) -- so the children of any node sum to zero when
  weighted by USD market cap.

    market                    root, absolute                 (Market)
    +- region                 relative to market             (Country: region)
    |   +- country            relative to its region         (Country: home_country)
    +- tradeable / rest       relative to market             (Tradeable Sector)
        +- industry           relative to the tradeable set  (Industry)

Columns in the returned tables follow a single ``"dimension:label"`` scheme (e.g.
``region:APAC``, ``country:JP``, ``universe:tradeable``, ``industry:bank``); the
lone exception is the root, named ``market``.
"""

from __future__ import annotations

import polars as pl


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


def wide_factors(long, child, prefix):
    """Collect a long factor frame into a wide, name-prefixed table.

    Pivots ``child`` into one column per value, named ``f"{prefix}:{value}"``,
    with a single ``date`` column. Returns an eager, chronologically sorted
    DataFrame.
    """
    wide = long.collect().pivot(values="_factor", index="date", on=child)
    renamed = {c: f"{prefix}:{c}" for c in wide.columns if c != "date"}
    return wide.rename(renamed).sort("date")
