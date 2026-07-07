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

from ..data.preprocess import winsorize_cross_section
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


def _weight_list(weight_col):
    """Normalize ``weight_col`` (None | str | sequence) to a list of column names."""
    if weight_col is None:
        return []
    if isinstance(weight_col, str):
        return [weight_col]
    return list(weight_col)


def periodic_returns(
    fwd_returns, target_col="excess_return", period_months=3,
    winsorize_limits=(0.01, 0.99), weight_col=None, *, delist_events,
):
    """Per-(security, period) compounded returns: the periodic core of
    :func:`forward_returns` without the forward shift.

    Runs trim -> winsorize -> compound -> settle (see :func:`forward_returns`
    for the full semantics of each step and of the **required**
    ``delist_events`` keyword) and returns ``[stock_id, date, period,
    (weight cols,) _ret]`` -- ``date`` is the (delisting-aware) period-end
    trading day and ``_ret`` the return compounded over that period.
    ``weight_col`` may be a single name or a sequence: each column is sampled
    at the period end (``.last()``, a formation-date level -- never compounded
    or winsorized) and passed through untouched.

    ``scripts/build_processed.py`` persists this frame per rebalancing
    frequency so downstream consumers shift it with
    :func:`forward_from_periodic` instead of recompounding the dailies.
    """
    df = as_df(fwd_returns).sort("stock_id", "date")
    weight = _weight_list(weight_col)

    events = None
    if delist_events is not None:
        events = as_df(delist_events).select("stock_id", "delist_date", "delist_return")
        if events.schema["stock_id"] != df.schema["stock_id"]:
            events = events.with_columns(pl.col("stock_id").cast(df.schema["stock_id"]))
        # A security ceases to exist after its delist date: drop the zombie tail so
        # post-delist periods never form (they would otherwise enter cross-sections
        # as phantom 0-return observations).
        df = (
            df.join(events.select("stock_id", "delist_date"), on="stock_id", how="left")
            .filter(
                pl.col("delist_date").is_null()
                | (pl.col("date") <= pl.col("delist_date"))
            )
            .drop("delist_date")
        )

    if winsorize_limits is not None:
        df = winsorize_cross_section(df, [target_col], winsorize_limits, by="date")
    periodic = (
        df.select("stock_id", "date", target_col, *weight)
        .with_columns(_period=pl.col("date").dt.truncate(f"{period_months}mo"))
        .group_by("stock_id", "_period")
        .agg(
            pl.col("date").last().alias("_date"),
            ((pl.col(target_col) + 1.0).product() - 1.0).alias("_ret"),
            # Weight is a level at the formation date, not a return: take the
            # period-end value as-is (no compounding).
            *[pl.col(w).last() for w in weight],
        )
        .sort("stock_id", "_period")
    )
    if events is not None:
        # Book the terminal settlement in the period spanning the delist -- after
        # winsorization (a -100% print must not be clipped as an outlier) and before
        # the forward shift. A null compounded return in the delist period (no usable
        # prints) settles from the last mark: treat it as 0, then apply the terminal.
        settled = (1.0 + pl.col("_ret").fill_null(0.0)) * (
            1.0 + pl.col("delist_return")
        ) - 1.0
        periodic = (
            periodic.join(
                events.with_columns(
                    _period=pl.col("delist_date").dt.truncate(f"{period_months}mo")
                ).select("stock_id", "_period", "delist_return"),
                on=["stock_id", "_period"],
                how="left",
            )
            .with_columns(
                _ret=pl.when(pl.col("delist_return").is_not_null())
                .then(settled)
                .otherwise(pl.col("_ret"))
            )
            .drop("delist_return")
            .sort("stock_id", "_period")
        )
    return periodic.select(
        "stock_id",
        pl.col("_date").alias("date"),
        pl.col("_period").alias("period"),
        *[pl.col(w) for w in weight],
        "_ret",
    )


def forward_from_periodic(periodic, lags=(1,), ret_col="_ret"):
    """Forward-shift a per-period return frame into ``_fwd{lag}`` columns.

    ``periodic`` is a :func:`periodic_returns`-shaped frame (``stock_id``,
    ``date``, ``period``, the ``ret_col`` return column, any passthrough
    columns). ``_fwd{lag}`` holds ``ret_col`` realised ``lag`` periods ahead
    (``shift(-lag)`` within each security, ordered by ``period``); ``ret_col``
    itself is dropped, passthrough columns survive.
    """
    df = as_df(periodic).sort("stock_id", "period")
    keep = [c for c in df.columns if c != ret_col]
    fwd = [
        pl.col(ret_col).shift(-lag).over("stock_id").alias(f"_fwd{lag}")
        for lag in lags
    ]
    return df.select(*keep, *fwd)


def forward_returns(
    fwd_returns, lags=(1,), target_col="excess_return", period_months=3,
    winsorize_limits=(0.01, 0.99), weight_col=None, *, delist_events,
):
    """Compounded forward returns at each rebalancing-period horizon in ``lags``.

    **Precomputed periodic input**: a frame that already carries a ``period``
    column (a ``scripts/build_processed.py`` artifact or a
    :func:`periodic_returns` result) skips construction entirely -- trimming,
    winsorization and delisting settlement are baked into the artifact, so
    ``period_months`` / ``winsorize_limits`` / ``delist_events`` are inert and
    ``target_col`` picks the return variant column (e.g. ``"ret_wins"`` for the
    winsorized panel, ``"ret_raw"`` for the raw one). Only the forward shift
    (:func:`forward_from_periodic`) is applied. Daily input follows the
    documented pipeline below.

    The daily target is first **winsorized cross-sectionally per date** (to the
    ``winsorize_limits`` quantiles -- ``preprocess.winsorize_limits`` by default; pass
    ``None`` to skip): a single extreme daily print would otherwise compound into a
    period return that dominates the mean of a quantile portfolio or a Fama-MacBeth
    cross-section. Returns are not sub-universe-specific, so the bounds span the whole
    per-date cross-section (:func:`src.data.preprocess.winsorize_cross_section`).

    The daily panel is then downsampled to non-overlapping calendar buckets of
    ``period_months`` (the rebalancing frequency; quarterly by default), the
    per-period excess return is compounded within each ``(security, period)``
    bucket, and each bucket is stamped with its last trading date. ``_fwd{lag}``
    then holds the compounded return realised ``lag`` *periods* ahead
    (``shift(-lag)`` within each security).

    ``weight_col`` (e.g. ``"mcap_usd"``) surfaces a point-in-time weighting column
    to the validators. It is sampled at the **period-end** trading day (``.last()``,
    the formation date) and passed straight through untouched -- it is never
    winsorized or compounded -- so a downstream cap-weighted quantile portfolio
    weights each name by its market cap *at rebalance*. It must be present in
    ``fwd_returns``; ``None`` (default) carries no weight.

    ``delist_events`` (**required keyword**) threads the survivorship fix
    (DELISTING_HANDLING.md): rows after a security's ``delist_date`` are dropped, so
    no phantom post-delist period ever forms, and the reason-dependent terminal
    return is compounded into the period spanning the delist -- *after*
    winsorization, so a wipeout's -100% settlement is never clipped away as an
    outlier. Pass the frame from :func:`src.data.delisting.delist_events`
    (needs ``stock_id, delist_date, delist_return``), built on the same price
    window; events for securities absent from ``fwd_returns`` are inert. Pass
    ``None`` to opt out explicitly (legacy, survivorship-biased behaviour: a
    wiped-out name silently escapes its terminal loss).

    Returns ``[stock_id, date, period, (weight_col,) _fwd{lag}...]``: ``date`` is the
    security's period-end trading day (join it to a daily ``scores`` frame to keep
    exactly the rebalancing rows) and ``period`` is the common calendar bucket --
    group cross-sections on ``period``, since securities' period-end days differ under
    staggered trading calendars.
    """
    df = as_df(fwd_returns)

    if "period" in df.columns:
        if target_col not in df.columns:
            raise ValueError(
                "forward_returns: input has a 'period' column (precomputed "
                f"periodic panel) but no '{target_col}' return column -- pass "
                "target_col as one of the artifact's return variants "
                "(e.g. 'ret_wins' / 'ret_raw')."
            )
        keep = ["stock_id", "date", "period", *_weight_list(weight_col), target_col]
        return forward_from_periodic(df.select(keep), lags, ret_col=target_col)

    periodic = periodic_returns(
        df, target_col=target_col, period_months=period_months,
        winsorize_limits=winsorize_limits, weight_col=weight_col,
        delist_events=delist_events,
    )
    return forward_from_periodic(periodic, lags)


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
