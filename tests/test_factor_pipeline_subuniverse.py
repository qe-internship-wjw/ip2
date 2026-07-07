"""Sub-universe awareness of the factor pipeline (preprocess + neutralization).

Banks and insurers are structurally disjoint: regularization must never let a
sector factor's median or z-score pool across the boundary, and the
structural-beta signals (``neutralize = False``) must ride through neutralization
untouched while true style factors are residualised against the design.

Column names are real registry shorthands ("NIM" is BANKS, "E/P" is
ALL_FINANCIALS, ...) so the tests exercise the registry-driven masks, not
duck-typing.
"""

import datetime as dt

import numpy as np
import polars as pl

from src.data import preprocess
from src.factors.base import Applicability, FactorKind, registry
from src.factors.style.structural_beta import center_by_group
from src.validation.neutralization import neutralize

D = dt.date(2024, 3, 29)

# limits [0, 1] make winsorization a no-op so median/z assertions stay exact.
CFG = {
    "preprocess": {
        "winsorize_limits": [0.0, 1.0],
        "group_by": "date",
        "universe_col": "industry",
    }
}


def test_structural_beta_registration():
    reg = registry()
    expected = {
        "Beta~MKT": Applicability.ALL_FINANCIALS,
        "Beta~CTRY": Applicability.ALL_FINANCIALS,
        "Beta~IND (B)": Applicability.BANKS,
        "Beta~IND (I)": Applicability.INSURANCE,
    }
    for shorthand, applicability in expected.items():
        assert reg[shorthand].applicability is applicability
        assert reg[shorthand].kind is FactorKind.SYSTEMATIC
        assert reg[shorthand].neutralize is False
    # Only the beta signals are exempt from neutralization.
    others = [cls for shorthand, cls in reg.items() if shorthand not in expected]
    assert others and all(cls.neutralize for cls in others)


def test_regularize_partitions_subuniverses():
    scores = pl.DataFrame(
        {
            "stock_id": [1, 2, 3, 4, 5, 6],
            "date": [D] * 6,
            "industry": ["bank"] * 4 + ["insurance_life", "insurance_pnc"],
            # Bank factor: a null to impute and a leaked insurer value (100.0)
            # that must be nulled, never pooled into the bank statistics.
            "NIM": [1.0, 2.0, 3.0, None, 100.0, None],
            # All-financials factor: standardized over the pooled cross-section.
            "E/P": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            # Bank-registered beta signal: same masking as any sector factor.
            "Beta~IND (B)": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )
    out = preprocess.regularize(scores, CFG).sort("stock_id")

    insurers = out.filter(pl.col("industry").str.starts_with("insurance"))
    assert insurers["NIM"].null_count() == 2
    assert insurers["Beta~IND (B)"].null_count() == 2

    # Bank NIM: null imputed with the *bank* median (2.0), z-scored over the
    # filled bank cross-section [1, 2, 3, 2]. Had the insurer 100.0 pooled in,
    # the median fill and the moments would differ wildly.
    z = 1.0 / np.sqrt(2.0 / 3.0)
    np.testing.assert_allclose(
        out.filter(pl.col("industry") == "bank")["NIM"].to_numpy(),
        [-z, 0.0, z, 0.0],
        atol=1e-12,
    )

    # Bank beta signal: z-scored strictly within banks.
    bank_beta = out.filter(pl.col("industry") == "bank")["Beta~IND (B)"].to_numpy()
    np.testing.assert_allclose(bank_beta.mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(bank_beta.std(ddof=1), 1.0, atol=1e-12)

    # All-financials factor: pooled moments over the whole cross-section.
    np.testing.assert_allclose(
        out["E/P"].to_numpy(),
        (np.arange(1.0, 7.0) - 3.5) / np.sqrt(3.5),
        atol=1e-12,
    )


def test_neutralize_residualises_styles_but_passes_beta_signals_through():
    rng = np.random.default_rng(7)
    n = 40
    mkt = rng.normal(0.0, 1.0, n)
    scores = pl.DataFrame(
        {
            "stock_id": np.arange(n),
            "date": [D] * n,
            "industry": ["bank"] * (n // 2) + ["insurance_life"] * (n - n // 2),
            # Style score contaminated by the market beta: must be residualised.
            "E/P": 2.0 * mkt + rng.normal(0.0, 0.3, n),
            # Structural-beta signal: exactly what the design would annihilate.
            "Beta~MKT": rng.normal(0.0, 1.0, n),
        }
    )
    loads = pl.DataFrame({"stock_id": np.arange(n), "date": [D] * n, "MKT": mkt})

    regd = preprocess.regularize(scores, CFG).sort("stock_id")
    neu = neutralize(regd, loads, CFG, by="date").sort("stock_id")

    # The exempt signal rides through un-residualised (re-standardization of an
    # already-standardized column is the identity).
    np.testing.assert_allclose(
        neu["Beta~MKT"].to_numpy(), regd["Beta~MKT"].to_numpy(), atol=1e-9
    )

    # The style factor is genuinely residualised: changed, and orthogonal to
    # the design regressor.
    ep = neu["E/P"].to_numpy()
    assert not np.allclose(ep, regd["E/P"].to_numpy(), atol=1e-3)
    assert abs(np.corrcoef(ep, mkt)[0, 1]) < 1e-6


def test_center_by_group_uses_calendar_month_cross_sections():
    # Staggered month-end trading days (17th vs 20th vs 31st) must land in one
    # calendar-month cross-section; each group's own median is removed.
    df = pl.DataFrame(
        {
            "stock_id": [1, 2, 3, 4, 5, 6],
            "date": [
                dt.date(2020, 1, 17),
                dt.date(2020, 1, 20),
                dt.date(2020, 1, 31),
                dt.date(2020, 1, 17),
                dt.date(2020, 1, 20),
                dt.date(2020, 1, 31),
            ],
            "grp": ["a", "a", "a", "b", "b", "b"],
            "X": [1.0, 2.0, 3.0, 101.0, 102.0, 103.0],
        }
    )
    out = center_by_group(df.lazy(), "X", "grp").collect()
    assert out.filter(pl.col("grp") == "a")["X"].to_list() == [-1.0, 0.0, 1.0]
    assert out.filter(pl.col("grp") == "b")["X"].to_list() == [-1.0, 0.0, 1.0]
