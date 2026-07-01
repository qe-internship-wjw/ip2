"""Shared helpers for the validation modules.

Data contracts used across neutralization / single-factor / redundancy:

* **scores**       -- a (wide) frame with ``stock_id``, ``date`` and one or more
  numeric factor-score columns (as produced by ``factors`` + ``preprocess``).
* **fwd_returns**  -- a frame with ``stock_id``, ``date`` and a per-period return
  column (``excess_return`` by default). "Forward" returns are formed here by
  shifting the return series backward within each security, so the value stored
  at date *t* is the return realised over the period(s) *after* t.

Everything is keyed on ``(stock_id, date)`` and every temporal shift is taken
``.over("stock_id")`` on a date-sorted frame, so no information leaks across
securities or backward in time.
"""

from __future__ import annotations

import polars as pl

ID_COLS = ("stock_id", "date")
_RETURN_PREFERENCE = ("excess_return", "total_return", "fwd_return", "ret", "return")


def as_df(frame) -> pl.DataFrame:
    """Materialise a frame (collect if lazy)."""
    return frame.collect() if isinstance(frame, pl.LazyFrame) else frame


def factor_columns(df: pl.DataFrame, extra_skip=()) -> list[str]:
    """Numeric factor columns of ``df`` (excludes ids and anything in extra_skip)."""
    skip = set(ID_COLS) | set(extra_skip)
    return [c for c, dtype in df.schema.items() if c not in skip and dtype.is_numeric()]


def infer_return_column(df: pl.DataFrame) -> str:
    """Best guess at the return column: a known name, else the lone numeric column."""
    for name in _RETURN_PREFERENCE:
        if name in df.columns:
            return name
    numeric = factor_columns(df)
    if len(numeric) != 1:
        raise ValueError(
            "Could not infer the return column; pass a frame with a recognised "
            f"return name {_RETURN_PREFERENCE} or a single numeric column."
        )
    return numeric[0]


def forward_returns(fwd_returns, lags=(1,), ret_col=None):
    """Per-security forward returns at each horizon in ``lags``.

    Returns ``(frame, ret_col)`` where ``frame`` has ``stock_id``, ``date`` and a
    ``_fwd{lag}`` column per lag holding the single-period return ``lag`` periods
    ahead (``ret.shift(-lag)`` within each security). Rebalancing periods are the
    rows of the input, so "lag" counts periods, not calendar days.
    """
    df = as_df(fwd_returns)
    ret_col = ret_col or infer_return_column(df)
    df = df.sort("stock_id", "date")
    fwd = [
        pl.col(ret_col).shift(-lag).over("stock_id").alias(f"_fwd{lag}")
        for lag in lags
    ]
    return df.select("stock_id", "date", *fwd), ret_col
