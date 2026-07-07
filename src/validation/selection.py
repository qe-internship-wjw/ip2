"""Three-step point-in-time factor selection (DYNAMIC_SELECTION_PLAN.md).

Nested filters, each keeping fewer factors than the last:

1. **Single-factor gate** -- a factor passes iff its lag-1 IC information ratio
   clears ``validation.ic.ir_shortlist_threshold`` in absolute value **or** its
   Fama-MacBeth Newey-West t-stat clears
   ``validation.fama_macbeth.t_stat_threshold`` (pooled architecture: each
   factor tested once, on the universe it is applicable to -- see
   :func:`single_factor.fama_macbeth` ``pooled=True``).
2. **Redundancy** -- correlation clustering over the *survivors only*
   (``validation.redundancy.correlation_threshold``); each cluster is
   represented by its largest-|FM coefficient| member
   (``cluster_representative: fm_gradient``).
3. **Parsimony** -- ``LassoCV`` over the *representatives only*; the final
   shortlist is the lasso survivors.

Structural-beta signals (registry ``neutralize = False``) are handled per the
Stage-5 caveat: their ICs are measured against **raw** forward returns
(residualising against the loadings would annihilate them by construction) and
they bypass the lasso (exactly collinear with the design the lasso target is
residualised against) -- a beta signal is selected iff it passes the
single-factor gate and is a cluster representative.

Point-in-time contract: with a ``cutoff`` (a formation period), every input is
sliced to ``period <= cutoff`` *before* the forward shift, so an IC/FM
observation at period ``s`` only exists once its return (realised over
``s+1..s+lag``) has completed by the end of the cutoff period -- selection at
the cutoff's formation date sees no future data. Inputs are the
``data/processed`` artifacts (or same-shaped frames): the quarterly ``neu_*``
panel, the periodic ``returns_*`` panel (must carry ``period``; ``target_col``
picks the variant) and the raw ``loadings_*`` design.
"""

from __future__ import annotations

import datetime as dt
import warnings
from dataclasses import dataclass, field

import polars as pl

from ..factors.base import registry
from ._common import as_df, factor_columns
from .redundancy import (
    average_correlation,
    cluster_factors,
    lasso_select,
    select_cluster_representatives,
)
from .single_factor import fama_macbeth, information_ratio, rank_ic

_EMPTY_PAIRS_SCHEMA = {"factor_a": pl.Utf8, "factor_b": pl.Utf8, "rho": pl.Float64}


@dataclass
class SelectionResult:
    """Shortlist + scorecard + every intermediate frame for diagnostics/plots."""

    shortlist: list[str]
    scorecard: pl.DataFrame
    cutoff: dt.date | None
    ic: pl.DataFrame                      # [period, factor, lag, ic]
    ir: pl.DataFrame                      # [factor, lag, ir]
    fm: pl.DataFrame                      # [sub_universe, factor, mean_coef, t_stat, nw_se, n_periods]
    pairs: pl.DataFrame                   # [factor_a, factor_b, rho] (survivors only)
    flagged: pl.DataFrame                 # subset of pairs above the threshold
    clusters: list[list[str]] = field(default_factory=list)
    representatives: list[str] = field(default_factory=list)
    lasso_survivors: list[str] = field(default_factory=list)


def _as_date(value) -> dt.date | None:
    if value is None or isinstance(value, dt.date) and not isinstance(value, dt.datetime):
        return value
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, str):
        return dt.date.fromisoformat(value)
    raise TypeError(f"expected a date, datetime or ISO string, got {type(value)!r}")


def _window(df: pl.DataFrame, train_start, cutoff) -> pl.DataFrame:
    if train_start is not None:
        df = df.filter(pl.col("period") >= train_start)
    if cutoff is not None:
        df = df.filter(pl.col("period") <= cutoff)
    return df


def select_features(
    neu,
    returns_panel,
    loadings,
    cfg,
    *,
    cutoff=None,
    train_start=None,
    target_col="ret_wins",
    delist_events=None,
    universe_col="industry",
) -> SelectionResult:
    """Run the three-step selection on the ``[train_start, cutoff]`` window.

    Parameters
    ----------
    neu : quarterly neutralized panel ``[stock_id, date, period, industry,
        <factor shorthands...>]`` (the ``neu_3m`` artifact). ``date`` is the
        join key to the returns panel's period-end rows.
    returns_panel : periodic returns panel carrying ``period`` (the
        ``returns_3m`` artifact); ``target_col`` picks the return variant
        (``"ret_wins"`` for selection). Trim/winsorization/delisting settlement
        are baked in, so ``delist_events`` is inert for this input (pass it
        only when supplying a daily panel *without* ``cutoff``).
    loadings : raw structural design ``[stock_id, date, period, MKT, beta_*,
        is_*]`` -- the residualisation design for the style-factor ICs and the
        lasso target.
    cfg : config mapping (thresholds under ``validation.*``; see module doc).
    cutoff : formation period (date / ISO string) ending the selection window;
        ``None`` uses the full sample (in-sample demo mode).
    train_start : first period entering the window (``backtest.schedule.
        train_start``); ``None`` keeps everything from the panel start.
    """
    neu, loadings = as_df(neu), as_df(loadings)
    returns_panel = as_df(returns_panel)
    cutoff, train_start = _as_date(cutoff), _as_date(train_start)

    if "date" not in neu.columns:
        raise ValueError(
            "select_features needs a 'date' column on the neutralized panel "
            "(the period-end join key to the returns panel) -- use the "
            "data/processed neu_* artifact or keep 'date' when neutralizing."
        )
    if (cutoff is not None or train_start is not None) and "period" not in returns_panel.columns:
        raise ValueError(
            "select_features windows on the 'period' column: pass a periodic "
            "returns panel (a data/processed returns_* artifact or a "
            "periodic_returns frame), not a daily panel."
        )

    neu_w = _window(neu, train_start, cutoff)
    loads_w = _window(loadings, train_start, cutoff)
    returns_w = (
        _window(returns_panel, train_start, cutoff)
        if "period" in returns_panel.columns
        else returns_panel
    )

    val = cfg.get("validation", {}) or {}
    thr_ir = float(val.get("ic", {}).get("ir_shortlist_threshold", 0.2))
    thr_t = float(val.get("fama_macbeth", {}).get("t_stat_threshold", 1.0))
    lags = tuple(val.get("ic", {}).get("decay_lags", (1, 2, 3, 4)))
    corr_thr = float(val.get("redundancy", {}).get("correlation_threshold", 0.4))
    rep_method = val.get("redundancy", {}).get("cluster_representative", "fm_gradient")
    min_warn = int(val.get("selection", {}).get("min_shortlist_warn", 0))

    reg = registry()
    fac_cols = factor_columns(neu_w)
    style_cols = [c for c in fac_cols if reg[c].neutralize]
    beta_cols = [c for c in fac_cols if not reg[c].neutralize]

    # ── Step 1a: rank IC / IR (lag 1 gates; other lags ride along for decay
    # diagnostics). Style factors correlate against loadings-residualised
    # returns; beta signals against raw returns (module docstring).
    ic_frames = []
    if style_cols:
        ic_frames.append(
            rank_ic(
                neu_w.select("stock_id", "date", *style_cols), returns_w,
                lags=lags, nonstyle_exposures=loads_w, target_col=target_col,
                delist_events=delist_events,
            )
        )
    if beta_cols:
        ic_frames.append(
            rank_ic(
                neu_w.select("stock_id", "date", *beta_cols), returns_w,
                lags=lags, nonstyle_exposures=None, target_col=target_col,
                delist_events=delist_events,
            )
        )
    ic = pl.concat(ic_frames) if ic_frames else pl.DataFrame(
        schema={"period": pl.Date, "factor": pl.Utf8, "lag": pl.Int32, "ic": pl.Float64}
    )
    ir = information_ratio(ic) if ic.height else pl.DataFrame(
        schema={"factor": pl.Utf8, "lag": pl.Int32, "ir": pl.Float64}
    )
    ir1 = (
        ir.filter(pl.col("lag") == 1)
        .select("factor", ir_lag1=pl.col("ir").fill_nan(None))
    )

    # ── Step 1b: pooled Fama-MacBeth (one test row per factor).
    fm = fama_macbeth(
        neu_w.select("stock_id", "date", universe_col, *fac_cols), returns_w,
        target_col=target_col, universe_col=universe_col, pooled=True,
        delist_events=delist_events,
    )
    fm_best = (
        fm.filter(pl.col("factor") != "const")
        .with_columns(
            fm_t=pl.col("t_stat").fill_nan(None),
            fm_coef=pl.col("mean_coef").fill_nan(None),
        )
        .sort(pl.col("fm_t").abs(), descending=True, nulls_last=True)
        .unique("factor", keep="first")
        .select("factor", fm_sub="sub_universe", fm_coef="fm_coef", fm_t="fm_t")
    )

    gates = (
        pl.DataFrame({"factor": fac_cols})
        .join(ir1, on="factor", how="left")
        .join(fm_best, on="factor", how="left")
        .with_columns(
            ir_pass=(pl.col("ir_lag1").abs() >= thr_ir).fill_null(False),
            fm_pass=(pl.col("fm_t").abs() >= thr_t).fill_null(False),
        )
        .with_columns(single_pass=pl.col("ir_pass") | pl.col("fm_pass"))
    )
    survivors = gates.filter(pl.col("single_pass"))["factor"].to_list()

    # ── Step 2: correlation clustering over the survivors only.
    if len(survivors) >= 2:
        pairs, flagged = average_correlation(
            neu_w.select("stock_id", "period", *survivors), threshold=corr_thr
        )
        clusters = cluster_factors((pairs, flagged), threshold=corr_thr)
    else:
        pairs = pl.DataFrame(schema=_EMPTY_PAIRS_SCHEMA)
        flagged = pl.DataFrame(schema=_EMPTY_PAIRS_SCHEMA)
        clusters = [[f] for f in survivors]
    representatives = select_cluster_representatives(
        clusters, ic_ir=ir.filter(pl.col("lag") == 1), fm=fm, method=rep_method
    )

    # ── Step 3: lasso over the style representatives; beta signals bypass.
    style_reps = [f for f in fac_cols if f in representatives and reg[f].neutralize]
    beta_reps = [f for f in fac_cols if f in representatives and not reg[f].neutralize]
    lasso_survivors = (
        lasso_select(
            returns_w, neu_w.select("stock_id", "date", universe_col, *style_reps),
            cfg, nonstyle_exposures=loads_w, target_col=target_col,
            universe_col=universe_col, delist_events=delist_events,
        )
        if style_reps
        else []
    )
    selected = set(lasso_survivors) | set(beta_reps)
    shortlist = [f for f in fac_cols if f in selected]

    if min_warn and len(shortlist) < min_warn:
        warnings.warn(
            f"select_features(cutoff={cutoff}): shortlist has "
            f"{len(shortlist)} < {min_warn} factors",
            stacklevel=2,
        )

    cluster_id = {f: i for i, cluster in enumerate(clusters) for f in cluster}
    sleeve = {f: reg[f].sleeve for f in fac_cols}
    applic = {f: reg[f].applicability.value for f in fac_cols}
    scorecard = (
        gates.with_columns(
            cutoff=pl.lit(cutoff, dtype=pl.Date),
            sleeve=pl.col("factor").replace_strict(sleeve, default=None),
            applicability=pl.col("factor").replace_strict(applic, default=None),
            cluster_id=(
                pl.col("factor").replace_strict(cluster_id, default=None)
                if cluster_id
                else pl.lit(None, dtype=pl.Int64)
            ),
            representative=pl.col("factor").is_in(representatives),
            # True/False = entered the lasso and survived/dropped; null = never
            # entered (failed an earlier step, or a beta signal that bypasses).
            lasso=pl.when(pl.col("factor").is_in(style_reps or [""]))
            .then(pl.col("factor").is_in(lasso_survivors or [""]))
            .otherwise(None),
            selected=pl.col("factor").is_in(shortlist or [""]),
        )
        .select(
            "cutoff", "factor", "sleeve", "applicability", "fm_sub",
            "ir_lag1", "fm_coef", "fm_t", "ir_pass", "fm_pass", "single_pass",
            "cluster_id", "representative", "lasso", "selected",
        )
        .sort(
            "selected", "single_pass", pl.col("fm_t").abs(),
            descending=True, nulls_last=True,
        )
    )

    return SelectionResult(
        shortlist=shortlist, scorecard=scorecard, cutoff=cutoff,
        ic=ic, ir=ir, fm=fm, pairs=pairs, flagged=flagged, clusters=clusters,
        representatives=representatives, lasso_survivors=lasso_survivors,
    )
