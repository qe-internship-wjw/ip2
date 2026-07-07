"""Run the dynamic-selection walk-forward backtest end-to-end.

Per regime (``backtest.schedule``): point-in-time feature selection at the
cutoff, per-regime FM-premia/IC/risk-history re-estimation on the selected
factors, and a segmented engine run with the book + risk-EMA state threaded
across regime boundaries (``src/backtest/dynamic.py``,
DYNAMIC_SELECTION_PLAN.md). Consumes the ``data/processed`` artifacts -- run
``scripts/build_processed.py`` first.

Outputs (default ``<data.cache>/backtest/``):

    results.feather            one row per traded rebalance (+ regime)
    weights.feather            [period, stock_id, weight, regime]
    selection_history.feather  stacked per-cutoff scorecards (+ regime)
    performance.feather        results + quarterly mkt/bench + hedged series
    run_meta.json              config hash, schedule, shortlists, summary,
                               diagnostics (written last -- completeness marker)

Run inside the ``ip2`` conda env, from the repo root:

    conda run -n ip2 python scripts/run_backtest.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.backtest.dynamic import performance_frame, run_dynamic  # noqa: E402
from src.backtest.inputs import load_artifacts  # noqa: E402
from src.backtest.metrics import (  # noqa: E402
    information_ratio,
    max_drawdown,
    regime_performance,
    sharpe_ratio,
)

# CLARABEL is blocked by this machine's Application Control policy; the
# optimizer's retry chain runs OSQP first -- give it tight tolerances so
# `optimal_inaccurate` iterates don't slip through loose.
OSQP_OPTS = {"eps_abs": 1e-7, "eps_rel": 1e-7, "max_iter": 100_000}


def _log(msg: str) -> None:
    print(f"[run_backtest] {msg}", flush=True)


def _write(df: pl.DataFrame, path: Path) -> None:
    """Atomic feather write (tmp + replace) so a crash never leaves a torn file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.write_ipc(tmp)
    tmp.replace(path)


def summarize(perf: pl.DataFrame) -> dict:
    """Headline metrics of the quarterly performance frame."""
    net = perf["net_ret"].to_numpy()
    n = len(net)
    return {
        "quarters_traded": n,
        "ann_return_net": float(np.prod(1.0 + net) ** (4.0 / n) - 1.0),
        "ann_vol_net": float(np.std(net, ddof=1) * 2.0),
        "sharpe_net": sharpe_ratio(net, periods_per_year=4),
        "sharpe_net_hedged": sharpe_ratio(perf["hedged"].to_numpy(), periods_per_year=4),
        "max_drawdown": max_drawdown(net),
        "ir_vs_benchmark": information_ratio(
            net, perf["bench"].fill_null(0.0).to_numpy(), periods_per_year=4
        ),
        "avg_turnover": float(perf["turnover"].mean()),
        "avg_tc_bps": float(perf["tc"].mean() * 1e4),
        "avg_gross_leverage": float(perf["gross_lev"].mean()),
        "avg_net_exposure": float(perf["net_exp"].mean()),
        "avg_mkt_beta": float(perf["mkt_beta"].mean()),
        "return_data_gaps": int(perf["n_return_gaps"].sum()),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="Dynamic-selection walk-forward backtest on the processed artifacts."
    )
    ap.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    ap.add_argument(
        "--out", default=None,
        help="output directory (default: <data.cache>/backtest)",
    )
    args = ap.parse_args(argv)

    # Categorical stock_id columns are joined across independently-read feathers.
    pl.enable_string_cache()

    cfg = config.load(args.config)
    out_dir = Path(args.out) if args.out else None
    if out_dir is None:
        cache = Path(cfg["data"].get("cache", "data/processed"))
        if not cache.is_absolute():
            cache = PROJECT_ROOT / cache
        out_dir = cache / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    _log("loading processed artifacts...")
    artifacts = load_artifacts(cfg, PROJECT_ROOT)

    _log("running the dynamic walk-forward (select -> build inputs -> engine per regime)...")
    res = run_dynamic(cfg, artifacts, solver_opts=OSQP_OPTS)
    for d in res.diagnostics:
        if "n_selected" in d:
            flags = " CARRIED" if d["carried_forward"] else ""
            flags += " BELOW-MIN" if d["below_min_warn"] else ""
            _log(
                f"regime {d['regime']} cutoff={d['cutoff']}: "
                f"{d['n_selected']} selected{flags}"
            )
    skipped = [d for d in res.diagnostics if "skipped" in d]
    wiped = [d for d in res.diagnostics if "book_wiped" in d]
    unsolved = [d for d in res.diagnostics if d.get("solved") is False]
    _log(
        f"traded {res.results.height} rebalances across {len(res.regimes)} regimes | "
        f"skipped {len(skipped)} | book-wipes {len(wiped)} | solver fallbacks {len(unsolved)}"
    )

    perf = performance_frame(res.results, artifacts.market_daily, cfg)
    summary = summarize(perf)
    for k, v in summary.items():
        _log(f"  {k}: {round(v, 4) if isinstance(v, float) else v}")
    _log("per-regime performance:")
    for row in regime_performance(res.results).to_dicts():
        _log(
            f"  regime {row['regime']} [{row['start']}..{row['end']}]: "
            f"sharpe {row['sharpe']:+.2f}, ann {row['ann_ret']:+.1%}, "
            f"maxDD {row['max_drawdown']:.1%}"
        )

    _write(res.results, out_dir / "results.feather")
    _write(res.weights, out_dir / "weights.feather")
    _write(res.selection_history, out_dir / "selection_history.feather")
    _write(perf, out_dir / "performance.feather")
    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": args.config,
        "config_sha256": hashlib.sha256(
            json.dumps(cfg.raw, sort_keys=True, default=str).encode()
        ).hexdigest(),
        "schedule": {k: str(v) for k, v in (cfg["backtest"].get("schedule") or {}).items()},
        "regimes": [
            {"index": r.index, "cutoff": str(r.cutoff),
             "formation_start": str(r.formation_start),
             "formation_end": str(r.formation_end)}
            for r in res.regimes
        ],
        "shortlists": res.shortlists,
        "summary": summary,
        "diagnostics": res.diagnostics,
    }
    # Written last: its presence marks a complete, consistent run.
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    _log(f"done in {time.perf_counter() - t0:,.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
