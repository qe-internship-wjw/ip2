"""Expected-return estimation and factor integration.

Per-factor expected returns depend on the factor's kind (``factors.base.FactorKind``):

    systematic:   mu_k,i = lambda_k(t) * z_k,i       (priced risk exposure)
    behavioural:  mu_k,i = IC_k(t) * sigma_i * z_k,i  (Grinold: mispricing x idio vol)

Factors are combined by the *integration* approach: a strategic weight vector
``alpha_k`` aggregates the per-factor expected returns,

    mu = sum_k alpha_k * mu_k

with ``alpha_k = 1/K`` by default (``portfolio.strategic_allocation: equal``) or
IR-weighted.

Everything is estimated **walk-forward** on the precompute-then-slice pattern:
:func:`premia_series` produces the per-period Fama-MacBeth coefficient series once
(each observation only uses contemporaneous data), and :func:`walk_forward_means`
turns any such series into a point-in-time expanding mean under the availability
rule -- the observation for period ``s`` regresses on the return realised over
``s+1``, so at rebalance ``t`` only ``s <= t-1`` has entered. Sample means are the
noisiest MVO input, so the per-factor means are **James-Stein-shrunk** toward their
grand mean before entering mu (MVO_PARAMETER_ESTIMATION.md §1). ``sigma_i`` comes
from the risk model's shrunk idio diagonal (``RiskModel.sigma_idio``) -- the engine
estimates risk *before* expected returns each rebalance.
"""

from __future__ import annotations

from functools import reduce

import numpy as np
import polars as pl

from ..factors.base import Applicability, FactorKind, registry
from ..validation._common import (
    SUBUNIVERSES,
    applicable_factor_columns,
    as_df,
    factor_columns,
    forward_returns,
    subuniverse_mask,
)


# ── per-factor formulas (stub contracts, used by the cross-section builder) ──────


def systematic_mu(beta, z):
    """Expected return for a systematic factor: ``mu_k = lambda_k * z_k``."""
    return np.asarray(beta, dtype=float) * np.asarray(z, dtype=float)


def behavioural_mu(ic, sigma, z):
    """Expected return for a behavioural factor: ``mu = IC * sigma * z``."""
    return (
        np.asarray(ic, dtype=float)
        * np.asarray(sigma, dtype=float)
        * np.asarray(z, dtype=float)
    )


def strategic_weights(factors, cfg, ir=None):
    """Strategic factor weights ``alpha_k``.

    ``portfolio.strategic_allocation: equal`` -> ``1/K``; ``ir_weighted`` ->
    ``|ir_k| / sum|ir|``, requiring an ``ir`` mapping ``factor -> IR``.
    """
    factors = list(factors)
    if not factors:
        raise ValueError("strategic_weights: empty factor list.")
    method = cfg.get("portfolio", {}).get("strategic_allocation", "equal")
    if method == "equal":
        return {f: 1.0 / len(factors) for f in factors}
    if method == "ir_weighted":
        if ir is None:
            raise ValueError("strategic_allocation=ir_weighted needs an `ir` mapping.")
        gross = sum(abs(ir.get(f, 0.0)) for f in factors)
        if gross == 0:
            return {f: 1.0 / len(factors) for f in factors}
        return {f: abs(ir.get(f, 0.0)) / gross for f in factors}
    raise ValueError(f"unknown strategic_allocation '{method}'")


def integrate(per_factor_mu, alpha):
    """Aggregate per-factor expected returns: ``mu = sum_k alpha_k * mu_k``.

    ``per_factor_mu`` maps factor -> aligned array of per-stock mu; a NaN
    contribution counts as 0 (no view).
    """
    total = None
    for f, m in per_factor_mu.items():
        arr = np.nan_to_num(np.asarray(m, dtype=float), nan=0.0) * float(alpha[f])
        total = arr if total is None else total + arr
    if total is None:
        raise ValueError("integrate: no per-factor expected returns supplied.")
    return total


# ── walk-forward estimation ──────────────────────────────────────────────────────


def premia_series(scores, fwd_returns, *, delist_events, target_col="excess_return",
                  period_months=3, winsorize_limits=(0.01, 0.99),
                  universe_col="industry", pooled=False):
    """Per-period Fama-MacBeth cross-sectional coefficients, per sub-universe.

    The same per-period regression as :func:`src.validation.single_factor.
    fama_macbeth`, but exposing the coefficient **time series** instead of its
    Newey-West aggregate -- the walk-forward premia are expanding means of this
    series. Each period's coefficient uses only that period's cross-section and
    the return realised one period ahead, so the series is point-in-time by
    construction. ``delist_events`` threads the survivorship fix into the target.

    ``pooled=True`` mirrors ``fama_macbeth(pooled=True)``: each factor's
    coefficient is estimated **once, on the universe it is applicable to** --
    all-financials factors on the pooled cross-section (``sub_universe="all"``,
    sector factors zero-filled off-sector as controls), sector factors on their
    own sub-universe (all-financials factors as controls). The traded beliefs
    then match the selection tests (DYNAMIC_SELECTION_PLAN.md §1.5);
    :func:`expected_return_cross_section` applies an ``"all"`` premium to every
    row. ``pooled=False`` is the legacy two-estimates-per-all-financials-factor
    architecture.

    Returns ``[period, sub_universe, factor, coef]`` (``const`` included).
    """
    import polars_ols  # noqa: F401 -- registers the `.least_squares` namespace

    scores = as_df(scores)
    if universe_col not in scores.columns:
        raise ValueError(f"premia_series needs the '{universe_col}' label column.")
    fwd = forward_returns(
        fwd_returns, lags=(1,), target_col=target_col, period_months=period_months,
        winsorize_limits=winsorize_limits, delist_events=delist_events,
    ).rename({"_fwd1": "_y"})
    # Join on the formation date when available (the scores' own `period` wins);
    # otherwise on the common period key. Either way the result carries `period`.
    if "date" in scores.columns:
        keep = ["stock_id", "date", "_y"]
        if "period" not in scores.columns:
            keep.insert(2, "period")
        df = scores.join(fwd.select(keep), on=["stock_id", "date"], how="inner")
    elif "period" in scores.columns:
        df = scores.join(
            fwd.select("stock_id", "period", "_y"), on=["stock_id", "period"], how="inner"
        )
    else:
        raise ValueError("premia_series: scores needs a 'date' or 'period' column.")

    # (sub_universe label, row mask, regressors, reported factors, zero-fill exprs)
    runs = []
    if pooled:
        all_fin = applicable_factor_columns(scores, (Applicability.ALL_FINANCIALS,))
        sector_cols = {
            "bank": applicable_factor_columns(scores, (Applicability.BANKS,)),
            "insurance": applicable_factor_columns(scores, (Applicability.INSURANCE,)),
        }
        if all_fin:
            zero_off_sector = [
                pl.when(subuniverse_mask(sub, universe_col))
                .then(pl.col(c)).otherwise(0.0).alias(c)
                for sub, cols in sector_cols.items() for c in cols
            ]
            pooled_mask = subuniverse_mask("bank", universe_col) | subuniverse_mask(
                "insurance", universe_col
            )
            fac = all_fin + [c for cols in sector_cols.values() for c in cols]
            runs.append(("all", pooled_mask, fac, ["const", *all_fin], zero_off_sector))
        for sub, applic in SUBUNIVERSES.items():
            if sector_cols[sub]:
                runs.append((
                    sub, subuniverse_mask(sub, universe_col),
                    applicable_factor_columns(scores, applic),
                    ["const", *sector_cols[sub]], None,
                ))
    else:
        for sub, applic in SUBUNIVERSES.items():
            fac_cols = applicable_factor_columns(scores, applic)
            runs.append((sub, subuniverse_mask(sub, universe_col), fac_cols, None, None))

    frames = []
    for sub, mask, fac_cols, reported, zero_fill in runs:
        sub_df = df.filter(mask)
        if zero_fill:
            sub_df = sub_df.with_columns(zero_fill)
        sub_df = sub_df.drop_nulls(["_y", *fac_cols])
        if sub_df.height == 0 or not fac_cols:
            continue
        coef = (
            pl.col("_y")
            .least_squares.ols(
                *fac_cols, mode="coefficients", add_intercept=True,
                null_policy="drop", solve_method="svd",
            )
            .over("period")
        )
        series = (
            sub_df.select("period", _coef=coef)
            .unique(subset="period", keep="first")
            .sort("period")
            .unnest("_coef")
            .unpivot(index="period", variable_name="factor", value_name="coef")
            .with_columns(sub_universe=pl.lit(sub))
        )
        if reported is not None:
            series = series.filter(pl.col("factor").is_in(reported))
        frames.append(series)
    if not frames:
        raise ValueError("premia_series: no sub-universe produced a coefficient series.")
    return (
        pl.concat(frames)
        .select("period", "sub_universe", "factor", "coef")
        .sort("sub_universe", "factor", "period")
    )


def walk_forward_means(series, value_col, keys, min_periods=8, shift_periods=1,
                       mode="walk_forward"):
    """Point-in-time expanding mean (and its sampling variance) of a period series.

    Availability rule: the observation for period ``s`` becomes known
    ``shift_periods`` periods later (an FM coefficient / lag-1 IC for period ``s``
    uses the return over ``s+1``), so the estimate *usable at* period ``t``
    averages observations with ``s <= t - shift_periods``. The series must carry
    one row per (key, period) on a contiguous period calendar (null values are
    fine; missing rows would misalign the shift). ``mode="full_sample"`` returns
    the leaky full-sample mean at every period -- debug only, never for results.

    Returns ``[*keys, period, mean, se2, n]`` where ``se2`` is the sampling
    variance of the mean (feeds James-Stein) and ``mean`` is null until
    ``min_periods`` observations are available.
    """
    keys = list(keys)
    df = as_df(series).sort([*keys, "period"])
    v = pl.col(value_col)
    if mode == "full_sample":
        n = v.count().over(keys)
        return df.with_columns(
            n=n,
            mean=v.mean().over(keys),
            se2=(v.var().over(keys) / n),
        ).select([*keys, "period", "mean", "se2", "n"])
    if mode != "walk_forward":
        raise ValueError(f"unknown estimation mode '{mode}'")

    min_periods = max(int(min_periods), 1)
    n = v.is_not_null().cum_sum().shift(shift_periods).over(keys)
    s1 = v.fill_null(0.0).cum_sum().shift(shift_periods).over(keys)
    sq = (v.fill_null(0.0) ** 2).cum_sum().shift(shift_periods).over(keys)
    mean = s1 / n
    var = (sq - s1**2 / n) / (n - 1)
    return df.with_columns(
        n=n.fill_null(0),
        mean=pl.when(n >= min_periods).then(mean).otherwise(None),
        se2=pl.when((n >= min_periods) & (n >= 2)).then(var / n).otherwise(None),
    ).select([*keys, "period", "mean", "se2", "n"])


def james_stein(estimates, value_col="mean", se2_col="se2", by=("period",)):
    """Shrink each ``by`` group's vector of means toward its grand mean (James-Stein).

    ``b = clip((K - 3) * avg(se2) / sum_k (m_k - m_bar)^2, 0, 1)`` and
    ``m_js = m_bar + (1 - b) (m - m_bar)``: extreme, likely spurious estimates are
    pulled toward the cross-factor central tendency. ``K <= 3`` -> no shrinkage
    (JS undefined); zero dispersion -> full collapse to the grand mean. Adds
    ``{value_col}_js`` and ``js_b``.
    """
    by = list(by)
    m = pl.col(value_col)
    k = m.count().over(by)
    m_bar = m.mean().over(by)
    ssq = ((m - m_bar) ** 2).sum().over(by)
    avg_se2 = pl.col(se2_col).mean().over(by)
    b = (
        pl.when(k <= 3)
        .then(0.0)
        .when(ssq <= 0)
        .then(1.0)
        .otherwise(((k - 3).cast(pl.Float64) * avg_se2 / ssq).clip(0.0, 1.0))
    )
    return estimates.with_columns(
        (m_bar + (1 - b) * (m - m_bar)).alias(f"{value_col}_js"), js_b=b
    )


def _est_cfg(cfg):
    ecfg = cfg.get("portfolio", {}).get("expected_returns", {}) or {}
    return (
        ecfg.get("estimation", "walk_forward"),
        int(ecfg.get("min_estimation_periods", 8)),
        bool(ecfg.get("james_stein", True)),
    )


def walk_forward_premia(coef_series, cfg):
    """``lambda_k(t)`` per (period, sub-universe, factor) from the FM series.

    Expanding mean under the availability rule (``shift=1``), James-Stein-shrunk
    across factors within each (period, sub-universe) -- the intercept is excluded
    (it is not a premium and would poison the grand mean). Returns
    ``[period, sub_universe, factor, premium]``; null until
    ``min_estimation_periods`` coefficients are available.
    """
    mode, min_periods, js = _est_cfg(cfg)
    wf = walk_forward_means(
        as_df(coef_series).filter(pl.col("factor") != "const"),
        "coef", ["sub_universe", "factor"], min_periods=min_periods,
        shift_periods=1, mode=mode,
    )
    if js:
        wf = james_stein(wf, by=("period", "sub_universe"))
        premium = pl.col("mean_js")
    else:
        premium = pl.col("mean")
    return wf.select("period", "sub_universe", "factor", premium.alias("premium"))


def walk_forward_ic(ic_series, cfg, lag=1):
    """``IC_k(t)`` per (period, factor) from the :func:`rank_ic` series.

    A lag-``L`` IC for period ``s`` uses the return over ``s+1 .. s+L``, so the
    availability shift equals the lag. James-Stein-shrunk across factors within
    each period. Returns ``[period, factor, ic]``.
    """
    mode, min_periods, js = _est_cfg(cfg)
    wf = walk_forward_means(
        as_df(ic_series).filter(pl.col("lag") == lag),
        "ic", ["factor"], min_periods=min_periods, shift_periods=lag, mode=mode,
    )
    if js:
        wf = james_stein(wf, by=("period",))
        ic = pl.col("mean_js")
    else:
        ic = pl.col("mean")
    return wf.select("period", "factor", ic.alias("ic"))


# ── cross-section assembly (the engine's entry point) ────────────────────────────


def expected_return_cross_section(neu_t, premia_t, ic_t, sigma, cfg, alpha=None,
                                  universe_col="industry"):
    """Integrated ``mu`` for one rebalance cross-section: ``[stock_id, mu]``.

    Parameters
    ----------
    neu_t : ``[stock_id, (industry,), <neutralized z cols...>]`` -- this period's rows.
    premia_t : ``[sub_universe, factor, premium]`` -- :func:`walk_forward_premia`
        sliced to the rebalance period (each stock uses its own sub-universe's λ).
    ic_t : ``[factor, ic]`` -- :func:`walk_forward_ic` sliced to the period.
    sigma : ``[stock_id, sigma]`` idio-vol frame (``RiskModel.sigma_idio``, in
        rebalance-period units). Required when a behavioural factor has an IC.
    alpha : factor -> strategic weight; defaults to equal ``1/K`` over the factor
        columns of ``neu_t``.

    A null z contributes 0 (a z-score is mean-zero within its sub-universe, so 0 is
    the neutral score); a factor with no usable premium/IC at ``t`` contributes 0
    everywhere (no view). Stocks whose factors are *all* null drop out.
    """
    neu_t = as_df(neu_t)
    fac_cols = factor_columns(neu_t)
    if not fac_cols:
        raise ValueError("expected_return_cross_section: no factor columns in neu_t.")
    if universe_col not in neu_t.columns:
        raise ValueError(f"expected_return_cross_section needs '{universe_col}'.")
    if alpha is None:
        alpha = {f: 1.0 / len(fac_cols) for f in fac_cols}

    lam = {
        (r["sub_universe"], r["factor"]): r["premium"]
        for r in as_df(premia_t).drop_nulls("premium").to_dicts()
    }
    ic = {
        r["factor"]: r["ic"] for r in as_df(ic_t).drop_nulls("ic").to_dicts()
    } if ic_t is not None else {}

    kinds = {f: registry()[f].kind for f in fac_cols}
    needs_sigma = any(
        kinds[f] is FactorKind.BEHAVIOURAL and f in ic for f in fac_cols
    )
    df = neu_t
    if needs_sigma:
        if sigma is None:
            raise ValueError("behavioural factors need the risk model's idio vol.")
        df = df.join(
            as_df(sigma).rename({"sigma": "_sigma"}), on="stock_id", how="left"
        )

    bank = subuniverse_mask("bank", universe_col)
    terms = []
    for f in fac_cols:
        a = float(alpha.get(f, 0.0))
        if kinds[f] is FactorKind.BEHAVIOURAL:
            if f not in ic:
                continue  # no usable IC yet: no view
            term = pl.lit(a * ic[f]) * pl.col("_sigma") * pl.col(f)
        else:
            # A pooled ("all") premium applies to every row; a sub-universe
            # premium overrides it on its own rows (pooled premia_series emits
            # exactly one of the two per factor -- legacy emits per-sub only).
            lam_all = lam.get(("all", f))
            lam_b = lam.get(("bank", f), lam_all)
            lam_i = lam.get(("insurance", f), lam_all)
            if lam_b is None and lam_i is None:
                continue  # no usable premium yet: no view
            lam_expr = (
                pl.when(bank)
                .then(pl.lit(lam_b, dtype=pl.Float64))
                .otherwise(pl.lit(lam_i, dtype=pl.Float64))
            )
            term = pl.lit(a) * lam_expr * pl.col(f)
        terms.append(term.fill_null(0.0))
    if not terms:
        raise ValueError("no factor has a usable premium/IC at this rebalance.")

    any_signal = pl.any_horizontal([pl.col(f).is_not_null() for f in fac_cols])
    mu = reduce(lambda x, y: x + y, terms)
    return df.filter(any_signal).select("stock_id", mu=mu)
