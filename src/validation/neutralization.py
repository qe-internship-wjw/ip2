"""Factor neutralization.

Ensure style factors are not implicitly betting on non-style risks. Run
cross-sectional OLS of raw factor scores on the non-style factors (market,
country/industry); the residuals are the neutralized factor scores.
"""

from __future__ import annotations


def neutralize(raw_scores, nonstyle_exposures, by="date"):
    """Return residuals of cross-sectional OLS of raw_scores on non-style factors."""
    raise NotImplementedError
