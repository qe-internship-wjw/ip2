"""Portfolio-construction ablation sweep (ABLATION_PLAN.md).

Two-level cached driver over six construction knobs:

    strategic_allocation, james_stein, gross_leverage, max_name_weight,
    risk_aversion  (engine-level; knobs 1-5)
    reselection_frequency_months                    (schedule-level; knob 6)

Cost structure (ABLATION_PLAN.md §5). ``select_features`` + ``build_inputs``
depend only on the regime schedule (i.e. ``reselection_frequency_months``) and
the fixed validation/risk-model config -- they are invariant to knobs 1-5, which
act only inside ``engine.run`` (optimizer + expected-return assembly). So:

  * Level 1 (expensive): per distinct reselection frequency, run PIT selection +
    ``build_inputs`` once per regime. Memoized by cutoff (selection) and by
    shortlist (inputs), so frequencies that share cutoffs share work.
  * Level 2 (cheap): re-run only the engine over knobs 1-5, reusing the cached
    per-regime inputs and threading ``EngineState`` across regime boundaries --
    exactly ``run_dynamic``'s inner loop with a patched config.

Outputs (default ``<data.cache>/ablation/``):

    results.feather     one row per config: the six knobs + summary metrics
    per_regime.feather  [config_id, regime, start, end, sharpe, ann_ret, maxDD]
    run_meta.json       grid spec, timings, cache stats (written last)

Run inside the ``ip2`` conda env, from the repo root:

    conda run -n ip2 python scripts/run_ablation.py --grid baseline   # Stage A
    conda run -n ip2 python scripts/run_ablation.py --grid ofat        # Stage B
    conda run -n ip2 python scripts/run_ablation.py --grid full        # 216 cells
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.backtest.dynamic import performance_frame  # noqa: E402
from src.backtest.engine import run as engine_run  # noqa: E402
from src.backtest.inputs import Artifacts, build_inputs, load_artifacts  # noqa: E402
from src.backtest.metrics import (  # noqa: E402
    information_ratio,
    max_drawdown,
    regime_performance,
    sharpe_ratio,
)
from src.backtest.schedule import regime_schedule  # noqa: E402
from src.validation.selection import select_features  # noqa: E402

# CLARABEL is blocked by this machine's Application Control policy; the retry
# chain runs OSQP first with tight tolerances (mirrors scripts/run_backtest.py).
OSQP_OPTS = {"eps_abs": 1e-7, "eps_rel": 1e-7, "max_iter": 100_000}

# The six knobs. Baseline is the current config.yaml.
KNOBS = (
    "strategic_allocation",
    "james_stein",
    "gross_leverage",
    "max_name_weight",
    "risk_aversion",
    "reselection_frequency_months",
)
BASELINE = {
    "strategic_allocation": "equal",
    "james_stein": True,
    "gross_leverage": 2.0,
    "max_name_weight": 0.02,
    "risk_aversion": 2.0,
    "reselection_frequency_months": 24,
}
GRID_VALUES = {
    "strategic_allocation": ["equal", "ir_weighted"],
    "james_stein": [True, False],
    "gross_leverage": [2.0, 3.0, 5.0],
    "max_name_weight": [0.02, 0.05],
    "risk_aversion": [1.0, 2.0, 5.0],
    "reselection_frequency_months": [24, 48, 240],
}


def _log(msg: str) -> None:
    print(f"[run_ablation] {msg}", flush=True)


def _write(df: pl.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.write_ipc(tmp)
    tmp.replace(path)


def config_id(params: dict) -> str:
    """Stable, human-readable id for one knob combination."""
    mnw = "none" if params["max_name_weight"] is None else params["max_name_weight"]
    return (
        f"alloc={params['strategic_allocation']}|js={int(params['james_stein'])}|"
        f"gl={params['gross_leverage']}|mnw={mnw}|ra={params['risk_aversion']}|"
        f"rf={params['reselection_frequency_months']}"
    )


def patch_cfg(base: config.Config, params: dict) -> config.Config:
    """Deep-copy the raw config and overwrite exactly the six ablation knobs."""
    raw = copy.deepcopy(base.raw)
    pf = raw.setdefault("portfolio", {})
    pf["strategic_allocation"] = params["strategic_allocation"]
    pf["risk_aversion"] = params["risk_aversion"]
    pf.setdefault("expected_returns", {})["james_stein"] = params["james_stein"]
    cons = pf.setdefault("constraints", {})
    cons["gross_leverage"] = params["gross_leverage"]
    cons["max_name_weight"] = params["max_name_weight"]
    sched = raw.setdefault("backtest", {}).setdefault("schedule", {})
    sched["reselection_frequency_months"] = params["reselection_frequency_months"]
    return config.Config(raw=raw)


# --- Level 1: selection + inputs, cached ------------------------------------


class Level1Cache:
    """Per-regime (shortlist, inputs), memoized across frequencies.

    Selection at a cutoff is deterministic given a fixed train_start + validation
    config, so it is memoized by cutoff; ``build_inputs`` is a pure function of
    the shortlist, so it is memoized by the shortlist tuple. Different reselection
    frequencies that share a cutoff therefore reuse both.
    """

    def __init__(self, base_cfg: config.Config, artifacts: Artifacts):
        self.base_cfg = base_cfg
        self.artifacts = artifacts
        self._sel: dict = {}      # cutoff.isoformat() -> raw shortlist
        self._inputs: dict = {}   # tuple(shortlist)   -> BacktestInputs
        self.sel_calls = 0
        self.input_builds = 0

    def _shortlist(self, cfg_f, reg, train_start):
        key = reg.cutoff.isoformat()
        if key not in self._sel:
            self.sel_calls += 1
            sel = select_features(
                self.artifacts.neu_q, self.artifacts.returns_q,
                self.artifacts.loadings_q, cfg_f,
                cutoff=reg.cutoff, train_start=train_start,
            )
            self._sel[key] = list(sel.shortlist)
        return self._sel[key]

    def _inputs_for(self, shortlist):
        key = tuple(shortlist)
        if key not in self._inputs:
            self.input_builds += 1
            self._inputs[key] = build_inputs(shortlist, self.base_cfg, self.artifacts)
        return self._inputs[key]

    def for_frequency(self, freq: int) -> list:
        """List of ``(regime, shortlist, inputs, carried)`` for one frequency.

        Mirrors ``run_dynamic``'s Level-1 loop, including the empty-shortlist
        carry-forward, so the engine sees identical inputs to the canonical run.
        """
        cfg_f = patch_cfg(self.base_cfg, {**BASELINE, "reselection_frequency_months": freq})
        regimes = regime_schedule(cfg_f)
        train_start = (cfg_f.get("backtest", {}).get("schedule") or {}).get("train_start")
        out, prev = [], None
        for reg in regimes:
            raw = self._shortlist(cfg_f, reg, train_start)
            carried = False
            shortlist = raw
            if not shortlist:
                if prev is None:
                    raise RuntimeError(
                        f"regime {reg.index} ({reg.cutoff}): empty shortlist, "
                        "nothing to carry forward."
                    )
                shortlist, carried = prev, True
            out.append((reg, shortlist, self._inputs_for(shortlist), carried))
            prev = shortlist
        return out


# --- Level 2: engine run over knobs 1-5, reusing cached inputs --------------


def run_engine(cfg_patched: config.Config, level1: list) -> pl.DataFrame:
    """Threaded segmented engine run; returns the concatenated results frame."""
    state, results = None, []
    for reg, _shortlist, inputs, _carried in level1:
        res = engine_run(
            inputs, cfg_patched, solver_opts=OSQP_OPTS,
            start_period=reg.formation_start, end_period=reg.formation_end,
            initial_state=state,
        )
        state = res.state
        if res.results.height:
            results.append(
                res.results.with_columns(regime=pl.lit(reg.index, dtype=pl.Int64))
            )
    if not results:
        raise RuntimeError("run_engine: no regime produced a traded rebalance.")
    return pl.concat(results)


def summarize(perf: pl.DataFrame) -> dict:
    """Headline metrics of the quarterly performance frame (matches run_backtest)."""
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


def run_config(base_cfg, artifacts, params, cache: Level1Cache) -> tuple[dict, pl.DataFrame]:
    """One ablation cell: summary row + per-regime frame."""
    level1 = cache.for_frequency(params["reselection_frequency_months"])
    cfg_p = patch_cfg(base_cfg, params)
    results = run_engine(cfg_p, level1)
    perf = performance_frame(results, artifacts.market_daily, cfg_p)
    row = {"config_id": config_id(params), **params, **summarize(perf)}
    per_regime = regime_performance(results).with_columns(
        config_id=pl.lit(config_id(params))
    )
    return row, per_regime


# --- grid enumeration -------------------------------------------------------


def ofat_configs() -> list[dict]:
    """Baseline + one-factor-at-a-time variations (Stage B)."""
    seen, out = set(), []
    for combo in [BASELINE, *(
        {**BASELINE, knob: v} for knob in KNOBS for v in GRID_VALUES[knob]
    )]:
        cid = config_id(combo)
        if cid not in seen:
            seen.add(cid)
            out.append(dict(combo))
    return out


def full_configs() -> list[dict]:
    """Full 2*2*3*2*3*3 = 216-cell factorial (Stage C, budget permitting)."""
    keys = list(GRID_VALUES)
    return [dict(zip(keys, vals)) for vals in product(*(GRID_VALUES[k] for k in keys))]


GRIDS = {
    "baseline": lambda: [dict(BASELINE)],
    "ofat": ofat_configs,
    "full": full_configs,
}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Portfolio-construction ablation sweep.")
    ap.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    ap.add_argument("--grid", choices=list(GRIDS), default="baseline")
    ap.add_argument("--out", default=None, help="output dir (default: <data.cache>/ablation)")
    args = ap.parse_args(argv)

    pl.enable_string_cache()
    base_cfg = config.load(args.config)

    out_dir = Path(args.out) if args.out else None
    if out_dir is None:
        cache_dir = Path(base_cfg["data"].get("cache", "data/processed"))
        if not cache_dir.is_absolute():
            cache_dir = PROJECT_ROOT / cache_dir
        out_dir = cache_dir / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    _log("loading processed artifacts...")
    artifacts = load_artifacts(base_cfg, PROJECT_ROOT)
    cache = Level1Cache(base_cfg, artifacts)

    configs = GRIDS[args.grid]()
    _log(f"grid '{args.grid}': {len(configs)} config(s)")

    rows, per_regime_frames, timings = [], [], []
    t_start = time.perf_counter()
    for i, params in enumerate(configs, 1):
        cid = config_id(params)
        # Time Level-1 (built/reused inside for_frequency) vs the engine run.
        t0 = time.perf_counter()
        level1 = cache.for_frequency(params["reselection_frequency_months"])
        t1 = time.perf_counter()
        cfg_p = patch_cfg(base_cfg, params)
        results = run_engine(cfg_p, level1)
        perf = performance_frame(results, artifacts.market_daily, cfg_p)
        t2 = time.perf_counter()

        summ = summarize(perf)
        rows.append({"config_id": cid, **params, **summ})
        per_regime_frames.append(
            regime_performance(results).with_columns(config_id=pl.lit(cid))
        )
        timings.append({"config_id": cid, "level1_s": t1 - t0, "engine_s": t2 - t1})
        _log(
            f"[{i}/{len(configs)}] {cid}: "
            f"sharpe_hedged={summ['sharpe_net_hedged']:+.3f} "
            f"sharpe_net={summ['sharpe_net']:+.3f} maxDD={summ['max_drawdown']:.1%} "
            f"tc={summ['avg_tc_bps']:.0f}bps gross={summ['avg_gross_leverage']:.2f} "
            f"| L1={t1 - t0:.1f}s engine={t2 - t1:.1f}s"
        )

    results_df = pl.DataFrame(rows)
    per_regime_df = pl.concat(per_regime_frames)
    _write(results_df, out_dir / "results.feather")
    _write(per_regime_df, out_dir / "per_regime.feather")

    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": args.config,
        "grid": args.grid,
        "n_configs": len(configs),
        "grid_values": GRID_VALUES,
        "baseline": BASELINE,
        "cache_stats": {
            "distinct_selection_cutoffs": cache.sel_calls,
            "distinct_input_builds": cache.input_builds,
        },
        "timings": timings,
        "wall_s": time.perf_counter() - t_start,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    _log(
        f"done in {meta['wall_s']:,.0f}s | "
        f"selection cutoffs={cache.sel_calls}, input builds={cache.input_builds} "
        f"-> {out_dir}"
    )


if __name__ == "__main__":
    main()
