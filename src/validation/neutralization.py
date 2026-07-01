"""Factor neutralization.

Ensure style factors are not implicitly betting on non-style risks. For each date
we run a cross-sectional OLS of the raw factor scores on the non-style factors
(market / country / industry) and keep the residuals as the neutralized scores.
"""

from __future__ import annotations

from ._common import as_df, cross_sectional_residuals, factor_columns


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
        ``stock_id``, ``by`` and the neutralized (residual) style columns, sorted
        by ``stock_id`` then ``by``. A stock whose regressors or score are null on
        a given date gets a null residual for that date.
    """
    scores = as_df(raw_scores)
    style_cols = factor_columns(scores)
    residuals = cross_sectional_residuals(scores, style_cols, nonstyle_exposures, by=by)
    return residuals.sort("stock_id", by)
