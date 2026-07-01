"""Factor neutralization.

Ensure style factors are not implicitly betting on non-style risks. Run
cross-sectional OLS of raw factor scores on the non-style factors (market,
country/industry); the residuals are the neutralized factor scores.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ._common import ID_COLS, as_df, factor_columns


def _design_matrix(exposures: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    """Coerce the non-style exposures into a numeric design matrix.

    Numeric columns are used directly; any non-numeric membership columns (e.g. a
    ``region_code`` / ``industry`` label) are one-hot encoded so country/industry
    can be passed as raw labels. Returns the frame (with ``stock_id``/``date``
    retained) and the list of regressor columns.
    """
    cat_cols = [
        c
        for c, dtype in exposures.schema.items()
        if c not in ID_COLS and not dtype.is_numeric()
    ]
    if cat_cols:
        exposures = exposures.to_dummies(columns=cat_cols)
    return exposures, factor_columns(exposures)


def neutralize(raw_scores, nonstyle_exposures, by="date"):
    """Residuals of a per-date cross-sectional OLS of each style score on the
    non-style exposures.

    Parameters
    ----------
    raw_scores : frame with ``stock_id``, ``date`` and one or more style-score
        columns.
    nonstyle_exposures : frame with ``stock_id``, ``date`` and the non-style
        regressors (market / country / industry). Numeric loadings are used
        as-is; label columns are one-hot encoded (an intercept is always added,
        so it captures the market level).
    by : cross-section key (default ``"date"``).

    Returns
    -------
    pl.DataFrame
        ``stock_id``, ``by`` and the neutralized (residual) style columns. A stock
        whose regressors or score are null on a given date gets a null residual
        for that date.
    """
    scores = as_df(raw_scores)
    exposures, exp_cols = _design_matrix(as_df(nonstyle_exposures))
    style_cols = factor_columns(scores)

    joined = scores.join(
        exposures.select("stock_id", by, *exp_cols), on=["stock_id", by], how="left"
    )

    parts = []
    for _, sub in joined.group_by(by, maintain_order=True):
        n = sub.height
        # Design matrix with an intercept; mask rows with any non-finite regressor.
        X = np.column_stack([np.ones(n), sub.select(exp_cols).to_numpy()])
        x_ok = np.isfinite(X).all(axis=1)

        resid_cols = {}
        for c in style_cols:
            y = sub[c].to_numpy().astype(float)
            resid = np.full(n, np.nan)
            mask = x_ok & np.isfinite(y)
            if int(mask.sum()) > X.shape[1]:  # need more obs than regressors
                beta, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
                resid[mask] = y[mask] - X[mask] @ beta
            resid_cols[c] = resid

        parts.append(
            sub.select("stock_id", by).with_columns(
                [pl.Series(c, resid_cols[c]) for c in style_cols]
            )
        )

    return pl.concat(parts).sort("stock_id", by)
