"""Factor neutralization.

Ensure style factors are not implicitly betting on non-style risks. For each date
we run a cross-sectional OLS of the (regularized) factor scores on the non-style
design, keep the residuals, and **re-standardize** them. Missing structural
regressors are imputed cross-sectionally first (:func:`._common.impute_loadings`) so
a beta's warm-up null does not drop a newly listed stock's valid score.

Factors registered with ``neutralize = False`` (the structural-beta signals,
:mod:`src.factors.style.structural_beta`) are **exempt**: they are explicit bets on
the design and exactly collinear with their expanded ``beta_{g}`` block, so the OLS
would annihilate them. They pass through untouched (up to the re-standardization).
"""

from __future__ import annotations

from ..data.preprocess import regularize
from ..factors.base import registry
from ._common import as_df, cross_sectional_residuals, factor_columns


def neutralize(raw_scores, nonstyle_exposures, cfg, by="date", universe_col="industry"):
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
        residual for that date. Factor columns registered with ``neutralize =
        False`` ride through un-residualised (re-standardization only).
    """
    scores = as_df(raw_scores)
    known = registry()
    # Opt-out seam: `neutralize = False` factors (structural-beta signals) are kept
    # as-is -- cross_sectional_residuals preserves non-target columns.
    style_cols = [c for c in factor_columns(scores) if known[c].neutralize]
    residuals = cross_sectional_residuals(scores, style_cols, nonstyle_exposures, by=by)
    # Re-standardize (within each factor's sub-universe) to restore unit cross-sectional variance.
    residuals = regularize(residuals, cfg)
    return residuals.sort("stock_id", by)
