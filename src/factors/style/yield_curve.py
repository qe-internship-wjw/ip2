"""Yield-curve (interest-rate risk) style factors.

A security's yield-curve sensitivity is a **style** exposure: the rolling
time-series beta of one of its own series on the change in the sovereign curve's
Nelson-Siegel ``level`` / ``slope`` (attached to the panel at build time by
:func:`src.factors.nonstyle.yield_curve.attach_nelson_siegel`). Two frameworks,
split by sub-universe:

1. **Fundamental sensitivity -- QUARTERLY.**
     * NIM sensitivity (banks)     -- metric = Net Interest Margin.
     * FIY sensitivity (insurance) -- metric = Float Investment Yield
       (investment income / reserves).

2. **Price sensitivity -- MONTHLY, all-financials.**

Like every style factor each ``compute`` returns a long
``[stock_id, date, <shorthand>]`` frame of raw scores; regularization and
neutralization (against the structural MKT / CTRY / IND loadings) happen
downstream. These sensitivities are signals, never neutralization regressors.
"""

from __future__ import annotations

import polars as pl

from ..base import Applicability, Factor, FactorKind, register

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
