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

    Returns ``[stock_id, date, period, _fwd{lag}...]``: ``date`` is the security's
    period-end trading day (join it to a daily ``scores`` frame to keep exactly the
    rebalancing rows) and ``period`` is the common calendar bucket -- group
    cross-sections on ``period``, since securities' period-end days differ under
    staggered trading calendars.
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
    return periodic.select(
        "stock_id",
        pl.col("_date").alias("date"),
        pl.col("_period").alias("period"),
        *fwd,
    )


def design_matrix(exposures, by="date"):
    """The non-style regressor matrix: every non-identifier column of ``exposures``.

    Returns ``(frame, regressor_columns)``. The ``exposures`` frame is purpose-built
    as the neutralization design -- the market beta plus the *expanded* country and
    industry blocks (a distinct ``beta_{group}`` slope per country/industry group and
    one-hot ``is_{group}`` dummies; see
    :func:`src.factors.nonstyle._hierarchy.expand_by_group`). So every column that is
    not an identifier (``stock_id`` / ``date`` / ``period`` / the ``by`` key) is a
    regressor.
    """
    exposures = as_df(exposures)
    skip = set(ID_COLS) | {by, "period"}
    reg_cols = [c for c in exposures.columns if c not in skip]
    return exposures, reg_cols


def impute_loadings(frame, reg_cols, by):
    """Cross-sectionally fill missing structural loadings before they are regressed.

    A structural beta is null throughout its rolling-beta warm-up (~2 years). Each
    null is filled with the per-``by`` median of the regressor's own comparable group.
    """
    return frame.with_columns(
        [
            pl.when(pl.col(c).is_null())
            .then(pl.when(pl.col(c) != 0).then(pl.col(c)).median().over(by))
            .otherwise(pl.col(c))
            .alias(c)
            for c in reg_cols
        ]
    )


def cross_sectional_residuals(frame, target_cols, exposures, by="date"):
    """Replace each ``target_cols`` column with its per-``by`` OLS residual.

    Runs a cross-sectional regression (one per ``by`` group, intercept added) of
    each target on the non-style regressors (:func:`design_matrix` -- the market
    beta, the per-group country/industry betas, and their one-hot dummies) and keeps
    the residuals: the neutralised series, net of both the group risk *slopes* and
    the group *baselines*. A single vectorised ``polars-ols`` expression per target
    evaluated ``.over(by)`` (no per-date loop), with the SVD solver so rank-deficient
    cross-sections (a dropped-baseline dummy set that is still collinear on a thin
    date, a singleton group) stay stable.

    Non-target columns are preserved; a row whose target -- or a
    regressor still null after imputation -- is null yields a null residual.
    """
    import polars_ols  # noqa: F401 -- registers the `.least_squares` namespace

    frame = as_df(frame)
    design, reg_cols = design_matrix(exposures, by)
    joined = frame.join(
        design.select("stock_id", by, *reg_cols), on=["stock_id", by], how="left"
    )
    joined = impute_loadings(joined, reg_cols, by)
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
