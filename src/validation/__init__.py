"""Factor validation (Experiment Plan).

Three distinct stages, one module each:

    neutralization  regress raw style scores on non-style loadings; keep residuals
    single_factor   Rank IC / IC decay / IR, Fama-MacBeth
    redundancy      correlation flags, clustering, lasso parsimony

Output is a shortlist of neutralized, non-redundant factors for the final model.
"""
