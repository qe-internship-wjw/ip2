"""Shared machinery for style factors.

Every style factor is a cross-sectional, per-security score. Unlike the non-style
structural factors (which collapse to one row per date), a style factor keeps
security identity, so ``compute`` returns a long frame

    [stock_id, date, <shorthand>]

carrying the raw (pre-neutralization) score in a column named after the factor's
shorthand. Standardization and neutralization happen downstream, so factors stay
declarative.

Two conveniences live here:

* :class:`RatioFactor` -- the common case, a purely cross-sectional expression of
  the current row. Subclasses just implement :meth:`RatioFactor.expr`.
* :func:`finalize` -- selects the identity columns plus the score, blanks out any
  non-finite value (``inf`` from division by zero, ``nan`` from ``0/0``) to null,
  and returns a frame matching the laziness of the input panel.
"""

from __future__ import annotations

import polars as pl

from ..base import Factor


def to_lazy(panel):
    """Normalise a panel to a LazyFrame regardless of what was passed in."""
    return panel.lazy() if isinstance(panel, pl.DataFrame) else panel


def clean(expr: pl.Expr) -> pl.Expr:
    """Map non-finite results (``inf`` / ``nan``) to null.

    This is the single guard for division by zero (``x/0`` -> ``inf``) and the
    indeterminate ``0/0`` (-> ``nan``); nulls in the inputs already propagate to
    null, so the ``.is_finite()`` check leaves them untouched.
    """
    return pl.when(expr.is_finite()).then(expr).otherwise(None)


def finalize(lf: pl.LazyFrame, score: pl.Expr, name: str, panel):
    """Project to ``[stock_id, date, name]``, matching the input's laziness."""
    out = lf.select("stock_id", "date", clean(score).alias(name))
    return out.collect() if isinstance(panel, pl.DataFrame) else out


class RatioFactor(Factor):
    """A style factor defined by a single cross-sectional expression.

    Subclasses implement :meth:`expr`; the raw score is that expression evaluated
    row-wise, with non-finite values nulled out.
    """

    def expr(self, cfg) -> pl.Expr:
        raise NotImplementedError

    def compute(self, panel, cfg):
        lf = to_lazy(panel)
        return finalize(lf, self.expr(cfg), self.shorthand, panel)
