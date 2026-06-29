"""Non-style (structural, systematic) factors.

These harvest risk premia other than overall market movement and are the factors
we neutralize the style factors against (research plan, Non-Style Factors):

    market           global market movement
    country_industry country & (sub)industry exposures via factor-mimicking
                     portfolios with non-unit, dynamic loadings (Bekaert 2009)
    yield_curve      Nelson-Siegel level/slope/curvature IRR factors
    currency         sensitivity of subsector cash flows to a trade-weighted FX index

All non-style factors are systematic (carry an inherent risk premium).
"""
