"""Transaction-cost model.

    TC(dw) = sum_i |dw_i| * [ 3 * (11 / log10(M_ff,i))^6 + 3 ]

where M_ff,i is the free-float market cap of security i. Enters the optimizer
objective as the penalty on turnover dw = w - w_prev.
"""

from __future__ import annotations


def cost(dw, free_float_mcap):
    """Per-rebalance transaction cost for turnover dw given free-float mcap."""
    raise NotImplementedError
