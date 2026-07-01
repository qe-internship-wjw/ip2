"""Shared helpers for the validation modules.

Data contracts used across neutralization / single-factor / redundancy:

* **scores**       -- a (wide) frame with ``stock_id``, ``date``, one column per
  registered factor (named by its ``shorthand``), and the ``industry``
  sub-universe label (``universe.industry_labels``: ``bank`` / ``insurance*``).
* **fwd_returns**  -- a daily frame with ``stock_id``, ``date`` and a per-period
  excess-return column (``excess_return`` by default; pass ``target_col``).

Forward returns are formed by compounding the daily target into rebalancing-period
buckets and shifting ``.over("stock_id")``, so the value stored at a period-end
date is the compounded return realised over the period(s) *after* it -- no
information leaks across securities or backward in time.
"""

from __future__ import annotations

import polars as pl

from ..factors.base import Applicability, registry

ID_COLS = ("stock_id", "date")

# Sub-universe -> the factor applicabilities defined on it. Banks and insurers
# each carry their own sector factors plus the shared all-financials factors.
SUBUNIVERSES: dict[str, tuple[Applicability, ...]] = {
    "bank": (Applicability.ALL_FINANCIALS, Applicability.BANKS),
    "insurance": (Applicability.ALL_FINANCIALS, Applicability.INSURANCE),
}


def as_df(frame) -> pl.DataFrame:
    """Materialise a frame (collect if lazy)."""
    return frame.collect() if isinstance(frame, pl.LazyFrame) else frame


def factor_columns(df: pl.DataFrame, extra_skip=()) -> list[str]:
    """Columns of ``df`` that are registered alpha factors (keyed by shorthand).
    """
    known = registry()
    skip = set(ID_COLS) | set(extra_skip)
    return [c for c in df.columns if c in known and c not in skip]


def applicable_factor_columns(df, applicabilities, extra_skip=()) -> list[str]:
    """Registered factor columns of ``df`` whose applicability is in the set.

    Used to assemble the dense regressor set for a single sub-universe (e.g.
    all-financials + bank factors for the bank cross-section).
    """
    known = registry()
    skip = set(ID_COLS) | set(extra_skip)
    wanted = set(applicabilities)
    return [
        c for c in df.columns
        if c in known and c not in skip and known[c].applicability in wanted
    ]


def subuniverse_mask(sub: str, universe_col: str = "industry") -> pl.Expr:
    """Boolean expr selecting the ``sub`` sub-universe ('bank' | 'insurance').
    """
    return pl.col(universe_col).str.starts_with(sub)


def forward_returns(fwd_returns, lags=(1,), target_col="excess_return", period_months=3):
    """Compounded forward returns at each rebalancing-period horizon in ``lags``.

    The daily panel is downsampled to non-overlapping calendar buckets of
    ``period_months`` (the rebalancing frequency; quarterly by default), the
    per-period excess return is compounded within each ``(security, period)``
    bucket, and each bucket is stamped with its last trading date. ``_fwd{lag}``
    then holds the compounded return realised ``lag`` *periods* ahead
    (``shift(-lag)`` within each security).

    Returns a frame ``[stock_id, date, _fwd{lag}...]`` keyed on the period-end
    dates; an inner join to a daily ``scores`` frame therefore keeps exactly the
    rebalancing rows.
    """
    df = as_df(fwd_returns).sort("stock_id", "date")
    periodic = (
        df.select("stock_id", "date", target_col)
        .with_columns(_period=pl.col("date").dt.truncate(f"{period_months}mo"))
        .group_by("stock_id", "_period")
        .agg(
            _date=pl.col("date").last(),
            _ret=(pl.col(target_col) + 1.0).product() - 1.0,
        )
        .sort("stock_id", "_period")
    )
    fwd = [
        pl.col("_ret").shift(-lag).over("stock_id").alias(f"_fwd{lag}")
        for lag in lags
    ]
    return periodic.select("stock_id", pl.col("_date").alias("date"), *fwd)


def design_matrix(exposures, by="date"):
    """Registered non-style loadings of ``exposures`` as the regressor matrix.

    Returns ``(frame, regressor_columns)``; regressors are the registered factor
    columns (the per-security structural loadings), selected via the registry.
    """
    exposures = as_df(exposures)
    return exposures, factor_columns(exposures, extra_skip=(by,))


def cross_sectional_residuals(frame, target_cols, exposures, by="date"):
    """Replace each ``target_cols`` column with its per-``by`` OLS residual.

    Runs a cross-sectional regression (one per ``by`` group, intercept added) of
    each target on the non-style loadings and keeps the residuals -- the
    neutralised series. A single vectorised ``polars-ols`` expression per target
    evaluated ``.over(by)`` (no per-date loop), with the SVD solver so
    rank-deficient cross-sections stay stable. Non-target columns are preserved; a
    row whose target or regressors are null yields a null residual.
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
