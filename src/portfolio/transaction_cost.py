"""Transaction-cost model.

    TC(dw) = sum_i |dw_i| * c_i,    c_i = [ 3 * (11 / log10(M_ff,i))^6 + 3 ] * 1e-4

where ``M_ff,i = mcap_usd_i * free_float_fraction_i`` is the free-float market cap in
USD (``free_float_percentage`` from ``fundamental_master``, PIT via the fundamentals
as-of join) and the bracketed formula is in basis points -- hence the ``1e-4`` into
decimal return units. The same coefficients enter the optimizer objective (penalty on
turnover ``dw = w - w_prev``) and the engine's realized P&L accounting, so the
optimizer's view of costs and the books agree by construction.

Costs apply to **voluntary turnover only**: involuntary delisting exits are never
charged (a worthless equity is not sold at its last price -- see PORTFOLIO_PLAN.md §5
and DELISTING_HANDLING.md part 3).
"""

from __future__ import annotations

import numpy as np
import polars as pl

#: Floor on free-float mcap (USD). Coefficients are evaluated at max(M_ff, floor), so
#: this doubles as the cost cap: log10(1e7) = 7 -> 3 * (11/7)^6 + 3 ~ 48 bps.
MCAP_FLOOR = 1e7


def free_float_mcap(
    frame, mcap_col="mcap_usd", pct_col="free_float_percentage", by=None
):
    """Attach ``free_float_mcap = mcap * free-float fraction`` to a frame.

    The vendor scale of ``pct_col`` is auto-detected -- a median above 1 means
    percentage points (divide by 100), otherwise it is already a fraction -- and the
    fraction is clamped to ``[0, 1]``. A null fraction falls back to the median
    fraction of its ``by`` group (e.g. ``["period", "industry"]`` for the
    sub-universe median at each rebalance); a group that is entirely null defaults
    to 1.0 (full float: no information, no haircut). Accepts and returns eager or
    lazy, matching the input.
    """
    lazy_in = isinstance(frame, pl.LazyFrame)
    lf = frame.lazy()

    med = lf.select(pl.col(pct_col).median()).collect().item()
    scale = 0.01 if (med is not None and med > 1.0) else 1.0
    frac = (pl.col(pct_col) * scale).clip(0.0, 1.0)
    if by:
        keys = [by] if isinstance(by, str) else list(by)
        frac = frac.fill_null(frac.median().over(keys))
    else:
        frac = frac.fill_null(frac.median())

    out = lf.with_columns(
        (pl.col(mcap_col) * frac.fill_null(1.0)).alias("free_float_mcap")
    )
    return out if lazy_in else out.collect()


def linear_cost_coefficients(free_float_mcap):
    """Per-name linear cost coefficients ``c_i`` in decimal return units.

    ``M_ff`` is floored at :data:`MCAP_FLOOR` -- which also caps the coefficient --
    and a null/NaN ``M_ff`` gets the capped (most expensive) coefficient rather
    than a garbage value.
    """
    m = np.asarray(free_float_mcap, dtype=float)
    m = np.where(np.isfinite(m), m, MCAP_FLOOR)
    m = np.maximum(m, MCAP_FLOOR)
    bps = 3.0 * (11.0 / np.log10(m)) ** 6 + 3.0
    return bps * 1e-4


def cost(dw, free_float_mcap):
    """Per-rebalance transaction cost of turnover ``dw``, in decimal return units."""
    c = linear_cost_coefficients(free_float_mcap)
    return float(np.sum(c * np.abs(np.asarray(dw, dtype=float))))
