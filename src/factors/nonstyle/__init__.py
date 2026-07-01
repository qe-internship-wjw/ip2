"""Non-style (structural, systematic) factors.

These harvest risk premia other than overall market movement and are the factors
we neutralize the style factors against. Each exposes a per-security loading:

    market           beta on the global market return
    country_industry beta on country & (sub)industry factor-mimicking returns
    yield_curve      Nelson-Siegel level/slope sensitivities

``currency`` (subsector cash-flow sensitivity to a trade-weighted FX index) is
planned but not yet implemented.
"""
