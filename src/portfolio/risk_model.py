"""Covariance estimation: structured factor risk model + shrinkage.

    Sigma = B F B' + D

with Ledoit-Wolf-style shrinkage on **both** estimated blocks, per
MVO_PARAMETER_ESTIMATION.md for the ``K < T < N`` regime (K ~ 20 factors,
T ~ 200-300 monthly observations, N ~ 1000 names, so ``K(K-1)/2 ~ T``):

* ``F`` (K x K): sample covariance of the monthly cross-sectional factor returns,
  LW-shrunk toward the **diagonal of sample factor variances**. Implemented in
  correlation space -- standardize the factor-return series, run sklearn's
  ``LedoitWolf`` (whose scaled-identity target on unit-variance data is the identity
  correlation), rescale by the sample vols -- so off-diagonals shrink exactly by
  ``(1 - delta)`` while the well-estimated per-factor variances are preserved.
* ``D`` (diagonal): per-name residual variances. Strict diagonality is structural
  (``N > T`` makes the full residual matrix pure noise and would break the cheap
  structured ``w' Sigma w``); the variances are Vasicek/LW-shrunk toward the
  cross-sectional mean -- the "volatility floor" that stops the optimizer from
  piling into a name whose sample variance randomly hovered near zero.

Both blocks are EMA-smoothed across rebalances (``portfolio.covariance.
ema_coefficient`` > 0.9); the engine threads the state through :func:`estimate` and
must slice the monthly histories to ``period <= t`` before calling (point-in-time
is the caller's contract, matching the precompute-then-slice pattern).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

_ID_COLS = ("stock_id", "date", "period", "industry")


def factor_returns(frame, exposure_cols, target_col, by="period"):
    """Per-``by`` cross-sectional OLS of ``target_col`` on ``exposure_cols``.

    ``frame`` holds one row per (stock, period) with the period-``m`` exposures and
    the period-``m+1`` return as ``target_col`` (build the target with the
    delisting-adjusted :func:`src.validation._common.forward_returns`, so a
    wipeout's terminal -100% enters the factor-return and residual history).
    Rows with a null target or exposure are dropped -- impute exposures first.

    Returns ``(factor_rets, residuals)``: a long ``[by, factor, f]`` frame (one row
    per period x factor, ``factor`` includes ``const``) and ``[stock_id, by,
    resid]``. Vectorised ``polars_ols`` ``.over(by)`` with the SVD solver, matching
    the Fama-MacBeth machinery.
    """
    import polars_ols  # noqa: F401 -- registers the `.least_squares` namespace

    df = frame.collect() if isinstance(frame, pl.LazyFrame) else frame
    df = df.drop_nulls([target_col, *exposure_cols])

    def _ols(mode):
        return (
            pl.col(target_col)
            .least_squares.ols(
                *exposure_cols, mode=mode, add_intercept=True,
                null_policy="drop", solve_method="svd",
            )
            .over(by)
        )

    residuals = df.select("stock_id", by, _ols("residuals").alias("resid"))
    rets = (
        df.select(by, _coef=_ols("coefficients"))
        .unique(subset=by, keep="first")
        .sort(by)
        .unnest("_coef")
        .unpivot(index=by, variable_name="factor", value_name="f")
    )
    return rets, residuals


def shrink_factor_cov(f_wide):
    """LW-shrink the K x K factor covariance toward the diagonal of sample variances.

    ``f_wide`` is the T x K factor-return history (numpy or DataFrame). Off-diagonals
    come out as exactly ``(1 - delta) * S_ij`` (sample covariance damped toward 0);
    the diagonal is the sample variances untouched. With ``K(K-1)/2 ~ T`` the LW
    formula picks a high ``delta`` -- expected and desired. Degenerate (zero-variance)
    columns keep a zero row/column. Returns ``(F, delta)``; ``F`` is PSD.
    """
    from sklearn.covariance import LedoitWolf

    X = f_wide.to_numpy() if isinstance(f_wide, pl.DataFrame) else np.asarray(f_wide, float)
    if X.ndim != 2:
        raise ValueError("shrink_factor_cov expects a T x K matrix.")
    sd = X.std(axis=0, ddof=0)
    if X.shape[0] < 2:
        return np.diag(sd**2), 1.0

    X = X - X.mean(axis=0, keepdims=True)
    ok = sd > 0
    Z = np.zeros_like(X)
    Z[:, ok] = X[:, ok] / sd[ok]
    lw = LedoitWolf(assume_centered=True).fit(Z)
    F = lw.covariance_ * np.outer(sd, sd)
    # The correlation-space diagonal is 1 by construction; pin the variances exactly.
    np.fill_diagonal(F, sd**2)
    return F, float(lw.shrinkage_)


def shrink_idio(s2, n_obs):
    """Vasicek/LW-style shrink of per-name idio variances toward the grand mean.

    ``delta* = mean_i(Var-hat(s2_i)) / mean_i((s2_i - s2_bar)^2)`` clipped to
    ``[0, 1]``, with the normal-theory ``Var-hat(s2_i) = 2 s2_i^2 / (n_i - 1)``.
    Zero cross-sectional dispersion means the mean *is* the estimate
    (``delta = 1``). Returns ``(s2_shrunk, delta)``; the cross-sectional mean is
    preserved exactly.
    """
    s2 = np.asarray(s2, dtype=float)
    n = np.maximum(np.asarray(n_obs, dtype=float), 2.0)
    s2_bar = s2.mean()
    dispersion = ((s2 - s2_bar) ** 2).mean()
    noise = (2.0 * s2**2 / (n - 1.0)).mean()
    delta = 1.0 if dispersion <= 0 else float(np.clip(noise / dispersion, 0.0, 1.0))
    return (1.0 - delta) * s2 + delta * s2_bar, delta


def ema_update(prev_cov, new_cov, coef):
    """EMA smoothing ``coef * prev + (1 - coef) * new``; ``prev=None`` seeds with new."""
    new_cov = np.asarray(new_cov, dtype=float)
    if prev_cov is None:
        return new_cov
    return float(coef) * np.asarray(prev_cov, dtype=float) + (1.0 - float(coef)) * new_cov


@dataclass
class RiskState:
    """EMA state threaded between rebalances, keyed by factor identity.

    ``factors`` labels the rows/columns of ``F`` so a later :func:`estimate`
    with a *different* factor set (dynamic re-selection) aligns by **name**:
    surviving factors keep their EMA history, entries involving a new factor
    seed at the fresh windowed estimate, and a factor swap at equal K never
    blends the covariances of two different factors. ``D`` is the per-name
    ``[stock_id, _d]`` idio frame (already a name-keyed join downstream).
    """

    factors: list
    F: np.ndarray
    D: pl.DataFrame


@dataclass
class RiskModel:
    """Structured covariance at one rebalance, in rebalance-period return units.

    The N x N Sigma is never materialized for optimization -- the optimizer consumes
    the parts (``w' Sigma w = ||F^0.5 B' w||^2 + sum_i D_i w_i^2``).
    """

    stock_ids: list
    factors: list
    B: np.ndarray  # N x K exposures (column order == `factors`)
    F: np.ndarray  # K x K shrunk + EMA-smoothed factor covariance
    D: np.ndarray  # (N,) shrunk + EMA-smoothed idio variances
    meta: dict = field(default_factory=dict)

    @property
    def sigma_idio(self) -> np.ndarray:
        """Per-name idio vol -- the Grinold sigma_i consumed by ``behavioural_mu``."""
        return np.sqrt(np.maximum(self.D, 0.0))

    @property
    def state(self) -> RiskState:
        """Factor-labelled EMA state to thread into the next :func:`estimate`."""
        return RiskState(
            factors=list(self.factors),
            F=self.F,
            D=pl.DataFrame({"stock_id": self.stock_ids, "_d": self.D}),
        )

    def covariance(self) -> np.ndarray:
        """Materialize the full N x N Sigma (tests / diagnostics only)."""
        return self.B @ self.F @ self.B.T + np.diag(self.D)


def _exposure_cols(exposures):
    return [c for c in exposures.columns if c not in _ID_COLS]


def estimate(exposures, factor_rets, residuals, cfg, state=None, by="period"):
    """Full risk model at one rebalance: window -> shrink both blocks -> EMA -> scale.

    Parameters
    ----------
    exposures : current cross-section ``[stock_id, (industry,), <factor cols...>]``.
        Null exposures are filled with 0 (a sector z is defined as neutral off its
        sub-universe; structural loadings should be imputed upstream). A ``const``
        column of ones is appended automatically. The optional ``industry`` column
        drives the sub-universe median fallback for names without residual history.
    factor_rets, residuals : the long monthly histories from :func:`factor_returns`,
        **already sliced to periods <= t by the caller** (point-in-time contract).
    cfg : reads ``portfolio.risk_model.*``, ``portfolio.covariance.ema_coefficient``
        and ``backtest.rebalancing_frequency_months`` (monthly -> quarterly scaling).
    state : ``RiskModel.state`` (a :class:`RiskState`) of the previous rebalance, or
        None to seed. When the factor set changed since, ``F`` is aligned by factor
        *name* -- surviving factors keep their EMA, new ones seed at the fresh
        windowed estimate. A legacy unlabelled ``(F, d_frame)`` tuple is still
        accepted (shape-checked reseed only).

    Returns a :class:`RiskModel`; shrinkage intensities and observation counts land
    in ``meta``.
    """
    pcfg = cfg.get("portfolio", {}) or {}
    rmcfg = pcfg.get("risk_model", {}) or {}
    cov_window = int(rmcfg.get("factor_cov_window_months", 60))
    idio_window = int(rmcfg.get("idio_window_months", 36))
    idio_min_obs = int(rmcfg.get("idio_min_obs", 12))
    freq = int(rmcfg.get("frequency_months", 1))
    rebal = int(cfg.get("backtest", {}).get("rebalancing_frequency_months", 3))
    ema = float(pcfg.get("covariance", {}).get("ema_coefficient", 0.94))
    scale = rebal / freq

    exposures = exposures.collect() if isinstance(exposures, pl.LazyFrame) else exposures
    factor_rets = factor_rets.collect() if isinstance(factor_rets, pl.LazyFrame) else factor_rets
    residuals = residuals.collect() if isinstance(residuals, pl.LazyFrame) else residuals

    cols = _exposure_cols(exposures)
    order = [*cols, "const"]

    # ── F: trailing window of factor returns -> LW-to-diagonal -> EMA -> scale ──
    recent = factor_rets.select(pl.col(by).unique().sort().tail(cov_window))[by].to_list()
    wide = (
        factor_rets.filter(pl.col(by).is_in(recent))
        .pivot(on="factor", index=by, values="f")
        .sort(by)
    )
    missing = [f for f in order if f not in wide.columns]
    if missing:
        raise ValueError(f"factor_rets lacks return series for exposures: {missing}")
    X = wide.select([pl.col(f).fill_null(0.0) for f in order]).to_numpy()
    f_hat, delta_f = shrink_factor_cov(X)

    # ── D: trailing residual variances -> impute -> Vasicek shrink -> EMA -> scale ──
    r_recent = residuals.select(pl.col(by).unique().sort().tail(idio_window))[by].to_list()
    stats = (
        residuals.filter(pl.col(by).is_in(r_recent))
        .group_by("stock_id")
        .agg(s2=pl.col("resid").var(), n=pl.col("resid").count())
        .filter(pl.col("n") >= idio_min_obs)
    )
    id_cols = ["stock_id"] + (["industry"] if "industry" in exposures.columns else [])
    d0 = exposures.select(id_cols).join(stats, on="stock_id", how="left")
    n_imputed = int(d0["s2"].null_count())
    if "industry" in d0.columns:
        d0 = d0.with_columns(pl.col("s2").fill_null(pl.col("s2").median().over("industry")))
    d0 = d0.with_columns(pl.col("s2").fill_null(pl.col("s2").median()))
    if d0["s2"].null_count():
        raise ValueError("estimate: no usable residual history to build the idio block.")
    # Imputed names carry the minimum-history sampling noise, not n=0.
    d0 = d0.with_columns(pl.col("n").fill_null(idio_min_obs))
    s2_hat, delta_d = shrink_idio(d0["s2"].to_numpy(), d0["n"].to_numpy())

    # ── EMA against the previous state, then monthly -> rebalance-period units ──
    if state is None:
        f_labels, f_prev, d_prev = None, None, None
    elif isinstance(state, RiskState):
        f_labels, f_prev, d_prev = state.factors, state.F, state.D
    else:  # legacy (F, d_frame) tuple: factor identities unknown
        f_prev, d_prev = state
        f_labels = None

    new = f_hat * scale
    if f_prev is None:
        F = new
    elif f_labels is not None:
        # Align the previous F by factor name: a factor-set change (dynamic
        # re-selection) must never blend the covariances of different factors,
        # even when K happens to match. Entries whose factors both survive keep
        # their EMA; entries involving a new factor seed at the fresh estimate.
        idx = {f: i for i, f in enumerate(f_labels)}
        pos = [(i, idx[f]) for i, f in enumerate(order) if f in idx]
        aligned = np.full_like(new, np.nan)
        if pos:
            pos_new, pos_old = zip(*pos)
            aligned[np.ix_(pos_new, pos_new)] = np.asarray(f_prev, dtype=float)[
                np.ix_(pos_old, pos_old)
            ]
        F = np.where(np.isnan(aligned), new, ema * aligned + (1.0 - ema) * new)
        F = (F + F.T) / 2.0  # exact symmetry after the mixed blend
    elif np.shape(f_prev) == new.shape:
        F = ema_update(f_prev, new, ema)
    else:
        F = new  # legacy unlabelled state with changed shape: reseed

    d_new = d0.with_columns(pl.Series("_d_new", s2_hat * scale)).select("stock_id", "_d_new")
    if d_prev is not None:
        d_new = d_new.join(d_prev, on="stock_id", how="left").with_columns(
            _d_new=pl.when(pl.col("_d").is_not_null())
            .then(ema * pl.col("_d") + (1.0 - ema) * pl.col("_d_new"))
            .otherwise(pl.col("_d_new"))  # name absent from state: seed at current
        )
    D = d_new["_d_new"].to_numpy()

    B = np.column_stack(
        [
            exposures.select([pl.col(c).fill_null(0.0) for c in cols]).to_numpy(),
            np.ones(exposures.height),
        ]
    )
    return RiskModel(
        stock_ids=exposures["stock_id"].to_list(),
        factors=order,
        B=B,
        F=F,
        D=D,
        meta={
            "delta_factor": delta_f,
            "delta_idio": delta_d,
            "t_obs": int(len(recent)),
            "n_idio_imputed": n_imputed,
            "scale": scale,
        },
    )
