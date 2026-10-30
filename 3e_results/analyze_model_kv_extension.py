"""Analysis of the follow-up campaign (model_kv_extension_v1): state-aware routing vs WLC.

Writes model_kv_extension_v1/results.csv and prints a summary.
"""

import csv
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE / "model_kv_extension_v1"
sys.path.insert(0, str(HERE / "analysis"))
import analyze_3e as A  # noqa: E402

CONFIGS = ("qwen_p4096", "lfm2_p4096", "openelm_p4096", "qwen_u4096")
CELLS = ("c12_sharegpt", "c8_bimodal")
SA, WLC = "state_aware", "weighted_least_connections"
METRICS = (  # name, getter (per run), scale
    ("e2el_p50", lambda r: r["lat"]["overall"]["e2el"]["p50"], 1e-3),
    ("e2el_p95", lambda r: r["lat"]["overall"]["e2el"]["p95"], 1e-3),
    ("ttft_p50", lambda r: r["lat"]["overall"]["ttft"]["p50"], 1e-3),
    ("completion_pp", lambda r: 100 * r["completion_rate"], 1.0),
)


def load_runs():
    runs = defaultdict(lambda: defaultdict(list))  # (config, cell) -> policy -> runs
    for mp in sorted(glob.glob(str(EXT / "*/*/rep*/run_meta.json"))):
        meta = json.load(open(mp))
        policy, config = meta["arm"].split("__")
        cell = f"c{meta['concurrency']}_{meta['workload']}"
        run = A.per_run(mp)
        bench = json.load(open(os.path.join(os.path.dirname(mp), meta["bench_result_file"])))
        warm = bench.get("num_warmup", 3)
        chosen = Counter(q.get("chosen_replica") for q in A.load_router(
            os.path.join(os.path.dirname(mp), "router_log.jsonl"))[0]
            if q.get("type") == "request" and isinstance(q.get("request_id"), int)
            and q["request_id"] >= warm)
        run.update(verdict=meta.get("verdict"),
                   pi5_share=100 * chosen.get("pi-node-1", 0) / sum(chosen.values()))
        runs[config, cell][policy].append(run)
    return runs


def bh(ps):
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj, prev = [0.0] * m, 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        prev = min(prev, ps[i] * m / rank)
        adj[i] = prev
    return adj


def main():
    runs = load_runs()
    pairs = [(c, cell) for c in CONFIGS for cell in CELLS if (c, cell) in runs]
    assert len(pairs) == 7, pairs
    rows = []
    for config, cell in pairs:
        wlc, sa = runs[config, cell][WLC], runs[config, cell][SA]
        assert len(wlc) == len(sa) == 6, (config, cell, len(wlc), len(sa))
        assert all(r["verdict"] == "clean" for r in wlc + sa), (config, cell)
        failed = (sum(r["n_fail"] for r in wlc), sum(r["n_fail"] for r in sa))
        share = (A.median([r["pi5_share"] for r in wlc]), A.median([r["pi5_share"] for r in sa]))
        for name, get, scale in METRICS:
            a = [get(r) * scale for r in wlc]
            b = [get(r) * scale for r in sa]
            diff, p, _, _ = A.perm_test(a, b)
            _, lo, hi = A.bootstrap_ci(a, b)
            rows.append(dict(config=config, cell=cell, metric=name,
                             wlc=round(A.median(a), 4), sa=round(A.median(b), 4),
                             diff=round(diff, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                             p=round(p, 5), wlc_failed=failed[0], sa_failed=failed[1],
                             wlc_pi5_share=round(share[0], 2), sa_pi5_share=round(share[1], 2)))
    for name, *_ in METRICS:
        sel = [r for r in rows if r["metric"] == name]
        for r, adj in zip(sel, bh([r["p"] for r in sel])):
            r["p_adj"] = round(adj, 5)

    print("State-aware minus WLC (s; completion in pp); BH across 7 comparisons per metric")
    for r in rows:
        sig = "*" if r["p_adj"] < 0.05 else ""
        print(f"  {r['config']:14} {r['cell']:13} {r['metric']:14} WLC {r['wlc']:8.2f} "
              f"SA {r['sa']:8.2f} diff {r['diff']:+7.2f} [{r['ci_lo']:+7.2f}, {r['ci_hi']:+7.2f}] "
              f"p={r['p']:.4f} p_adj={r['p_adj']:.4f}{sig}")
    print("\nFailed requests (WLC/SA) and median Pi 5 share (WLC/SA):")
    for config, cell in pairs:
        r = next(x for x in rows if x["config"] == config and x["cell"] == cell)
        print(f"  {config:14} {cell:13} failed {r['wlc_failed']}/{r['sa_failed']}  "
              f"Pi 5 share {r['wlc_pi5_share']:.1f}%/{r['sa_pi5_share']:.1f}%")

    with (EXT / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
