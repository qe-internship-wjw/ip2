"""Nelson-Siegel yield-curve enrichment (panel build-time).

Summarise each sovereign curve by its Nelson-Siegel ``level`` / ``slope`` /
``curvature`` (:func:`fit_nelson_siegel`) and attach those parameters onto the
panel (:func:`attach_nelson_siegel`), keyed by the security's trading currency.

This module produces shared *inputs* -- not factors. The yield-curve **style
factors** (how a security's returns / fundamentals co-move with the curve) live
in :mod:`src.factors.style.yield_curve` and read the ``level`` / ``slope``
columns this module attaches.
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
