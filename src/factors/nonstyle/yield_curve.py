"""Yield-curve (interest-rate risk) factors via Nelson-Siegel.

Summarise each sovereign curve by its Nelson-Siegel ``level`` / ``slope`` /
``curvature`` (:func:`fit_nelson_siegel`) and then measure how a security co-moves
with the curve.

1. **Fundamental sensitivity -- QUARTERLY.**
     * NIM sensitivity (banks)     -- metric = Net Interest Margin.
     * FIY sensitivity (insurance) -- metric = Float Investment Yield
       (investment income / reserves).

2. **Price sensitivity -- MONTHLY.**
"""

from __future__ import annotations

import numpy as np
import polars as pl
from nelson_siegel_svensson.calibrate import betas_ns_ols

from ..base import Applicability, Factor, FactorKind, register


def _tenor_to_years(col: pl.Expr) -> pl.Expr:
    """Parse a `tenor_description` like ``6M`` / ``120M`` into years.

    All retained tenors are quoted in months, so the numeric part divided by
    twelve gives the maturity in years (e.g. ``6M`` -> 0.5, ``120M`` -> 10.0).
    """
    return col.str.extract(r"(\d+)").cast(pl.Float64) / 12.0


def fit_nelson_siegel(zero_curve, cfg):
    """Estimate level/slope/curvature per date per sovereign curve.

    For every ``date`` x ``currency`` pair the Nelson-Siegel curve is fitted to
    ``zero_rate`` against tenor, with the decay ``tau`` fixed.

    Parameters
    ----------
    zero_curve : pl.LazyFrame | pl.DataFrame
        Zero-curve table with ``date``, ``currency``, ``tenor_description`` and
        ``zero_rate`` columns.
    cfg : Config
        Provides ``factors.nelson_siegel.decay_tau``.

    Returns
    -------
    pl.DataFrame
        One row per ``date`` x ``currency`` with ``level``, ``slope`` and
        ``curvature``, sorted by date then currency.
    """
    tau = float(cfg["factors"]["nelson_siegel"]["decay_tau"])

    lf = zero_curve.lazy() if isinstance(zero_curve, pl.DataFrame) else zero_curve

    # Collect the tenor (in years) and rate per (date, currency) into lists so
    # each curve can be fitted independently; curves need >= 3 points for the
    # three betas to be identified.
    grouped = (
        lf.select(
            "date",
            "currency",
            tenor=_tenor_to_years(pl.col("tenor_description")),
            zero_rate=pl.col("zero_rate"),
        )
        .drop_nulls(["tenor", "zero_rate"])
        .group_by("date", "currency")
        .agg(pl.col("tenor"), pl.col("zero_rate"))
        .collect()
    )

    dates: list = []
    currencies: list = []
    levels: list[float] = []
    slopes: list[float] = []
    curvatures: list[float] = []

    for row in grouped.iter_rows(named=True):
        t = np.asarray(row["tenor"], dtype=float)
        y = np.asarray(row["zero_rate"], dtype=float)
        if t.size < 3:
            continue
        curve, _ = betas_ns_ols(tau, t, y)
        dates.append(row["date"])
        currencies.append(row["currency"])
        levels.append(curve.beta0)
        slopes.append(curve.beta1)
        curvatures.append(curve.beta2)

    return pl.DataFrame(
        {
            "date": dates,
            "currency": currencies,
            "level": levels,
            "slope": slopes,
            "curvature": curvatures,
        }
    ).with_columns(
        pl.col("currency").cast(pl.Categorical)
    ).sort("date", "currency")


# ── Panel enrichment (build-time, so the factors stay IO-free) ────────────────


def attach_nelson_siegel(panel, ns_params):
    """Left-join the fitted NS ``level``/``slope``/``curvature`` onto the panel.
    """
    lf = panel.lazy() if isinstance(panel, pl.DataFrame) else panel
    nsp = ns_params.lazy() if isinstance(ns_params, pl.DataFrame) else ns_params

    out = lf.join(
        nsp.select("date", "currency", "level", "slope", "curvature"),
        left_on=["date", "currency_code"],
        right_on=["date", "currency"],
        how="left",
    )
    return out.collect() if isinstance(panel, pl.DataFrame) else out


# ── Shared rolling-OLS estimator ──────────────────────────────────────────────

_REGRESSORS = 2  # Delta level, Delta slope


def _window_periods(cfg, months_per_period: int) -> tuple[int, int]:
    """Convert the configured window (in months) to a period count + minimum.

    ``months_per_period`` is 1 for the monthly (price) framework and 3 for the
    quarterly (fundamental) one. ``min_periods`` (half the window,
    floored) allows partial windows once enough observations have accrued.
    """
    months = int(cfg["factors"]["yield_curve"]["sensitivity_window_months"])
    window = max(_REGRESSORS + 2, months // months_per_period)
    min_periods = max(_REGRESSORS + 2, window // 2)
    return window, min_periods


def _rolling_ols_betas(periodic, window, min_periods, *, difference):
    """Per-security rolling OLS of the dependent on (Δlevel, Δslope).

    ``periodic`` carries one row per (security, period) with ``stock_id``,
    ``_period``, ``_date`` (representative trading date), the dependent level
    ``_m`` and the curve ``level``/``slope`` sampled at that period. The curve
    params are differenced period-over-period; ``difference`` also differences
    ``_m`` (fundamental levels) or leaves it as-is (returns are already a change).
    """
    import polars_ols  # noqa: F401 -- registers the `.least_squares` expr namespace

    lf = periodic.lazy() if isinstance(periodic, pl.DataFrame) else periodic
    y = pl.col("_m").diff().over("stock_id") if difference else pl.col("_m")

    # Difference/sort must precede the rolling window; the upstream group_by does
    # not guarantee row order, so we sort by (stock_id, _period) first.
    prepared = (
        lf.sort("stock_id", "_period")
        .with_columns(
            _y=y,
            _x1=pl.col("level").diff().over("stock_id"),
            _x2=pl.col("slope").diff().over("stock_id"),
        )
        .drop_nulls(["_y", "_x1", "_x2"])
        # y ~ intercept + Δlevel + Δslope, rolling within each security. The
        # coefficient struct carries one field per feature name (_x1 / _x2).
        .with_columns(
            _coef=pl.col("_y")
            .least_squares.rolling_ols(
                "_x1",
                "_x2",
                window_size=window,
                min_periods=min_periods,
                add_intercept=True,
                mode="coefficients",
                null_policy="drop",
            )
            .over("stock_id")
        )
    )

    def finite(field: str, name: str) -> pl.Expr:
        beta = pl.col("_coef").struct.field(field)
        return pl.when(beta.is_finite()).then(beta).otherwise(None).alias(name)

    return prepared.select(
        "stock_id",
        pl.col("_date").alias("date"),
        finite("_x1", "level_beta"),
        finite("_x2", "slope_beta"),
    ).collect()


# ══════════════════════════════════════════════════════════════════════════════
# Framework 1 -- Fundamental sensitivity (QUARTERLY)
# ══════════════════════════════════════════════════════════════════════════════


def metric_yield_sensitivity(panel, metric: pl.Expr, cfg):
    """QUARTERLY rolling level/slope sensitivity of a fundamental metric.

    The metric is a point-in-time fundamental that only refreshes ~quarterly, so it
    is resampled to **calendar-quarter** frequency before differencing.

    The two sides of the regression are aggregated differently on purpose:

    Regresses ΔMetric on the change in average (Δlevel, Δslope) over a trailing
    window of ``sensitivity_window_months / 3`` quarters. Returns an eager
    ``[stock_id, date, level_beta, slope_beta]`` frame.
    """
    window, min_periods = _window_periods(cfg, months_per_period=3)
    lf = panel.lazy() if isinstance(panel, pl.DataFrame) else panel

    quarterly = (
        lf.select("stock_id", "date", "level", "slope", _m=metric)
        .drop_nulls(["_m", "level", "slope"])
        .filter(pl.col("_m").is_finite())
        # 3-month (calendar-quarter) buckets.
        .with_columns(_period=pl.col("date").dt.truncate("3mo"))
        .sort("stock_id", "date")
        .group_by("stock_id", "_period")
        .agg(
            _date=pl.col("date").last(),
            _m=pl.col("_m").last(),          # last reported metric in the quarter
            level=pl.col("level").mean(),    # quarter-average curve (flow-aligned)
            slope=pl.col("slope").mean(),
        )
    )

    return _rolling_ols_betas(quarterly, window, min_periods, difference=True)


class _MetricYieldSensitivity(Factor):
    """Base for the QUARTERLY fundamental-metric yield-curve sensitivities.

    Subclasses set the subsector ``metric`` expression and ``_beta`` (which
    :func:`metric_yield_sensitivity` column to expose). ``compute`` reads the NS
    ``level``/``slope`` off the panel and returns ``[stock_id, date, <shorthand>]``.
    """

    kind = FactorKind.SYSTEMATIC
    _beta: str = "level_beta"

    def metric(self, cfg) -> pl.Expr:
        raise NotImplementedError

    def compute(self, panel, cfg):
        out = metric_yield_sensitivity(panel, self.metric(cfg), cfg).select(
            "stock_id", "date", pl.col(self._beta).alias(self.shorthand)
        )
        return out.lazy() if isinstance(panel, pl.LazyFrame) else out


@register
class NIMYieldLevelSensitivity(_MetricYieldSensitivity):
    name = "NIM Sensitivity to Yield Level"
    shorthand = "NIM~Level"
    sleeve = "Yield Curve"
    applicability = Applicability.BANKS
    _beta = "level_beta"

    def metric(self, cfg) -> pl.Expr:
        return pl.col("net_interest_margin")


@register
class NIMYieldSlopeSensitivity(_MetricYieldSensitivity):
    name = "NIM Sensitivity to Yield Slope"
    shorthand = "NIM~Slope"
    sleeve = "Yield Curve"
    applicability = Applicability.BANKS
    _beta = "slope_beta"

    def metric(self, cfg) -> pl.Expr:
        return pl.col("net_interest_margin")


@register
class FIYYieldLevelSensitivity(_MetricYieldSensitivity):
    name = "FIY Sensitivity to Yield Level"
    shorthand = "FIY~Level"
    sleeve = "Yield Curve"
    applicability = Applicability.INSURANCE
    _beta = "level_beta"

    def metric(self, cfg) -> pl.Expr:
        return pl.col("insurance_investment_income_ltm") / pl.col("insurance_reserves")


@register
class FIYYieldSlopeSensitivity(_MetricYieldSensitivity):
    name = "FIY Sensitivity to Yield Slope"
    shorthand = "FIY~Slope"
    sleeve = "Yield Curve"
    applicability = Applicability.INSURANCE
    _beta = "slope_beta"

    def metric(self, cfg) -> pl.Expr:
        return pl.col("insurance_investment_income_ltm") / pl.col("insurance_reserves")


# ══════════════════════════════════════════════════════════════════════════════
# Framework 2 -- Price sensitivity (MONTHLY)
# ══════════════════════════════════════════════════════════════════════════════


def return_yield_sensitivity(panel, cfg):
    """MONTHLY rolling level/slope sensitivity of equity (excess) returns.

    Returns an eager ``[stock_id, date, level_beta, slope_beta]`` frame.
    """
    window, min_periods = _window_periods(cfg, months_per_period=1)
    lf = panel.lazy() if isinstance(panel, pl.DataFrame) else panel

    monthly = (
        lf.select("stock_id", "date", "level", "slope", "excess_return")
        .drop_nulls(["excess_return", "level", "slope"])
        .filter(pl.col("excess_return").is_finite())
        .with_columns(_period=pl.col("date").dt.truncate("1mo"))
        .sort("stock_id", "date")
        .group_by("stock_id", "_period")
        .agg(
            _date=pl.col("date").last(),
            level=pl.col("level").last(),
            slope=pl.col("slope").last(),
            # Monthly compounded excess return is itself the dependent variable.
            _m=(pl.col("excess_return") + 1.0).product() - 1.0,
        )
    )

    return _rolling_ols_betas(monthly, window, min_periods, difference=False)


class _ReturnYieldSensitivity(Factor):
    """Base for the MONTHLY price/return yield-curve sensitivities.

    Defined on the whole financial universe (returns are always observable).
    Subclasses set ``_beta``; ``compute`` returns ``[stock_id, date, <shorthand>]``.
    """

    kind = FactorKind.SYSTEMATIC
    applicability = Applicability.ALL_FINANCIALS
    _beta: str = "level_beta"

    def compute(self, panel, cfg):
        out = return_yield_sensitivity(panel, cfg).select(
            "stock_id", "date", pl.col(self._beta).alias(self.shorthand)
        )
        return out.lazy() if isinstance(panel, pl.LazyFrame) else out


@register
class ReturnYieldLevelSensitivity(_ReturnYieldSensitivity):
    name = "Return Sensitivity to Yield Level"
    shorthand = "Ret~Level"
    sleeve = "Yield Curve"
    _beta = "level_beta"


@register
class ReturnYieldSlopeSensitivity(_ReturnYieldSensitivity):
    name = "Return Sensitivity to Yield Slope"
    shorthand = "Ret~Slope"
    sleeve = "Yield Curve"
    _beta = "slope_beta"
