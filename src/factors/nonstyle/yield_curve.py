"""Yield-curve (interest-rate risk) factors via Nelson-Siegel.

Equity sensitivities to level/slope changes are estimated by rolling
regression (e.g. 36-month). The decay factor tau is a
hyperparameter fixed during monthly OLS term-structure estimation.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from nelson_siegel_svensson.calibrate import betas_ns_ols


def _tenor_to_years(col: pl.Expr) -> pl.Expr:
    """Parse a `tenor_description` like ``6M`` / ``120M`` into years.

    All retained tenors are quoted in months, so the numeric part divided by
    twelve gives the maturity in years (e.g. ``6M`` -> 0.5, ``120M`` -> 10.0).
    """
    return col.str.extract(r"(\d+)").cast(pl.Float64) / 12.0


def fit_nelson_siegel(zero_curve, cfg):
    """Estimate level/slope/curvature per date per sovereign curve.

    For every ``date`` x ``currency`` pair the Nelson-Siegel curve is fitted to
    ``zero_rate`` against the parsed tenor (in years) by OLS on the betas, with
    the decay ``tau`` held fixed at the configured hyperparameter. The three
    fitted betas map directly to the term-structure factors:

        level     = beta0   (long-run rate)
        slope     = beta1   (short-end vs long-end)
        curvature = beta2   (medium-term hump)

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
    ).sort("date", "currency")


def yield_level_sensitivity(panel, ns_params, cfg):
    """Rolling sensitivity of equity returns to sovereign level changes."""
    raise NotImplementedError


def yield_slope_sensitivity(panel, ns_params, cfg):
    """Rolling sensitivity of equity returns to sovereign slope changes."""
    raise NotImplementedError
