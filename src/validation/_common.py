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


def design_matrix(exposures, by="date"):
    """Coerce non-style exposures into a numeric regressor design matrix.

    Numeric columns pass through; non-numeric membership labels (e.g. a
    ``region_code`` / ``industry`` column) are one-hot encoded with a reference
    level dropped (``drop_first``) so the dummies are not collinear with the
    regression intercept. Returns ``(frame, regressor_columns)`` with
    ``stock_id`` / ``by`` retained.
    """
    exposures = as_df(exposures)
    skip = {"stock_id", by}
    cat_cols = [
        c for c, dtype in exposures.schema.items()
        if c not in skip and not dtype.is_numeric()
    ]
    if cat_cols:
        exposures = exposures.to_dummies(columns=cat_cols, drop_first=True)
    reg_cols = [
        c for c, dtype in exposures.schema.items()
        if c not in skip and dtype.is_numeric()
    ]
    return exposures, reg_cols


def cross_sectional_residuals(frame, target_cols, exposures, by="date"):
    """Replace each ``target_cols`` column with its per-``by`` OLS residual.

    Runs a cross-sectional regression (one per ``by`` group, intercept added) of
    each target on the non-style ``exposures`` design and keeps the residuals --
    the neutralised series. This is a single vectorised ``polars-ols`` expression
    per target evaluated ``.over(by)`` (no per-date Python loop), with the SVD
    solver so rank-deficient cross-sections (e.g. a date missing a dummy level)
    stay numerically stable. Non-target columns are preserved; a row whose target
    or regressors are null yields a null residual.
    """
    import polars_ols  # noqa: F401 -- registers the `.least_squares` namespace

    frame = as_df(frame)
    design, reg_cols = design_matrix(exposures, by)
    joined = frame.join(
        design.select("stock_id", by, *reg_cols), on=["stock_id", by], how="left"
    )
    resid = [
        pl.col(c)
        .least_squares.ols(
            *reg_cols,
            mode="residuals",
            add_intercept=True,
            null_policy="drop",
            solve_method="svd",
        )
        .over(by)
        .alias(c)
        for c in target_cols
    ]
    keep = [c for c in frame.columns if c not in target_cols]
    return joined.select(*keep, *resid)
