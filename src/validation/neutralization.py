"""Factor neutralization.

Ensure style factors are not implicitly betting on non-style risks. For each date
we run a cross-sectional OLS of the (regularized) factor scores on the non-style
design, keep the residuals, and **re-standardize** them. Missing structural
regressors are imputed cross-sectionally first (:func:`._common.impute_loadings`) so
a beta's warm-up null does not drop a newly listed stock's valid score.
"""

from __future__ import annotations

from ..data.preprocess import standardize
from ._common import as_df, cross_sectional_residuals, factor_columns


def neutralize(raw_scores, nonstyle_exposures, by="date", universe_col="industry"):
    """Re-standardized residuals of a per-``by`` cross-sectional OLS of each style
    score on the non-style exposures.

    Parameters
    ----------
    raw_scores : frame with ``stock_id``, ``date`` and one or more style-score
        columns, plus the ``universe_col`` sub-universe label.
    nonstyle_exposures : frame with ``stock_id``, ``date`` and the non-style
        design columns -- the market beta plus the expanded per-group country /
        industry betas (``beta_{group}``) and one-hot group dummies
        (``is_{group}``). Every non-identifier column is used as a regressor
        (:func:`._common.design_matrix`); an intercept is always added. Missing
        loadings are imputed cross-sectionally before the OLS
        (:func:`._common.impute_loadings`).
    by : cross-section key (default ``"date"``).
    universe_col : sub-universe label for the re-standardization (default
        ``"industry"``); a sector factor is re-standardized only within its own
        sub-universe.

    Returns
    -------
    pl.DataFrame
        ``stock_id``, ``by``, the ``universe_col`` label and the neutralized style
        columns -- each re-standardized to unit cross-sectional variance within its
        sub-universe -- sorted by ``stock_id`` then ``by``. A stock whose score is
        null (or whose regressors are still null after imputation) gets a null
        residual for that date.
    """
    scores = as_df(raw_scores)
    style_cols = factor_columns(scores)
    residuals = cross_sectional_residuals(scores, style_cols, nonstyle_exposures, by=by)
    # Re-standardize (within each factor's sub-universe) to restore unit cross-sectional variance.
    residuals = standardize(residuals, by=by, universe_col=universe_col)
    return residuals.sort("stock_id", by)
