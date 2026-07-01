"""Factor validation (Experiment Plan).

Three distinct stages, one module each:

    neutralization  regress raw style scores on non-style factors; keep residuals
    single_factor   Rank IC / IC decay / IR, Fama-MacBeth
    redundancy      correlation flags, clustering, factor competition, lasso,
                    Schweinler-Wigner orthogonalization

Output is a shortlist of neutralized, non-redundant factors for the final model.
"""
