"""Analysis of the router-tracked load experiment (freshness_extension_v1), as pre-registered.

Writes freshness_extension_v1/results.csv and prints a summary.
"""

import csv
import itertools
import json
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE / "freshness_extension_v1"
sys.path.insert(0, str(HERE / "analysis"))
from analyze_3e import pct  # noqa: E402  (same percentile definition as the paper)

SA, WLC = "state_aware", "weighted_least_connections"
SAT, WLCT = "state_aware_tracked", "wlc_tracked"
ARMS = (SA, WLC, SAT, WLCT)
COMPARISONS = (  # (baseline, candidate); effect = candidate minus baseline
    (WLCT, SAT),  # primary
    (SA, SAT),
    (WLC, WLCT),
    (WLC, SA),
)
METRICS = ("e2el_p50", "e2el_p95", "ttft_p50", "completion")
DEADLINE_MS = 120_000.0
DEADLINE_FLOOR_MS = 119_000.0
BOOT_ITERS, BOOT_SEED = 20_000, 20261024
PI5 = "pi-node-1"


def ok_values(raw, metric):
    return [r[f"{metric}_ms"] for r in raw
            if r.get("success") and not r.get("truncated")
            and isinstance(r.get(f"{metric}_ms"), (int, float))]


def load_runs():
    runs = {arm: {} for arm in ARMS}
    for meta_path in sorted(ROOT.glob("*__fresh/c8_bimodal/rep*/run_meta.json")):
        meta = json.loads(meta_path.read_text())
        arm = meta["arm"].removesuffix("__fresh")
        rep = int(meta_path.parent.name[3:])
        bench = json.loads((meta_path.parent / meta["bench_result_file"]).read_text())
        raw = bench["raw"]
        warm = bench.get("num_warmup", 3)
        router = [json.loads(line) for line in (meta_path.parent / "router_log.jsonl").open()
                  if line.strip()]
        router = [r for r in router if r.get("type") == "request"
                  and isinstance(r.get("request_id"), int) and r["request_id"] >= warm]
        cut_router = sum(1 for r in router if r.get("outcome") != "success"
                         and (r.get("total_ms") or 0) >= DEADLINE_FLOOR_MS)
        cut_bench = sum(1 for r in raw if not r["success"] and r.get("outcome") == "connection_error")
        assert cut_router == cut_bench, (meta_path, cut_router, cut_bench)
        e2el = ok_values(raw, "e2el")
        n = len(raw)
        runs[arm][rep] = {
            "verdict": meta.get("verdict"),
            "e2el_p50": pct(e2el, 50) / 1e3,
            "e2el_p95": pct(e2el, 95) / 1e3,
            "ttft_p50": pct(ok_values(raw, "ttft"), 50) / 1e3,
            "completion": 100 * sum(1 for r in raw if r["success"]) / n,
            "deadline_cut": cut_bench,
            "other_failed": sum(1 for r in raw if not r["success"]) - cut_bench,
            "e2el_p50_bound": pct(e2el + [DEADLINE_MS] * cut_bench, 50) / 1e3,
            "e2el_p95_bound": pct(e2el + [DEADLINE_MS] * cut_bench, 95) / 1e3,
            "pi5_share": 100 * sum(1 for r in router if r.get("chosen_replica") == PI5) / len(router),
            "any_stale": 100 * sum(1 for r in router
                                   if any(s.get("stale") for s in r.get("state_vector", [])))
                         / len(router),
        }
    return runs


def stat(metric, base, cand):
    agg = statistics.mean if metric == "completion" else statistics.median
    return agg(cand) - agg(base)


def block_test(metric, base, cand):
    """Within-block swap permutation (exact) and block bootstrap CI."""
    obs = stat(metric, base, cand)
    n = len(base)
    extreme = total = 0
    for swaps in itertools.product((False, True), repeat=n):
        b = [c if s else a for a, c, s in zip(base, cand, swaps)]
        c = [a if s else c for a, c, s in zip(base, cand, swaps)]
        total += 1
        extreme += abs(stat(metric, b, c)) >= abs(obs) - 1e-12
    rng = random.Random(BOOT_SEED)
    boots = []
    for _ in range(BOOT_ITERS):
        idx = [rng.randrange(n) for _ in range(n)]
        boots.append(stat(metric, [base[i] for i in idx], [cand[i] for i in idx]))
    boots.sort()
    lo = boots[int(0.025 * BOOT_ITERS)]
    hi = boots[int(0.975 * BOOT_ITERS) - 1]
    return obs, lo, hi, extreme / total


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
    reps = sorted(runs[SA])
    for arm in ARMS:
        assert sorted(runs[arm]) == reps == list(range(1, 13)), (arm, sorted(runs[arm]))

    print("Per arm (12 runs each): median of run values; completion = mean")
    for arm in ARMS:
        v = list(runs[arm].values())
        med = lambda k: statistics.median(x[k] for x in v)  # noqa: E731
        print(f"  {arm:27} p50={med('e2el_p50'):6.2f}s p95={med('e2el_p95'):6.2f}s "
              f"ttft={med('ttft_p50'):5.2f}s completion={statistics.mean(x['completion'] for x in v):6.2f}% "
              f"deadline_cut={sum(x['deadline_cut'] for x in v)} other_failed={sum(x['other_failed'] for x in v)} "
              f"pi5={med('pi5_share'):5.1f}% any_stale={med('any_stale'):5.1f}% "
              f"verdicts={sorted({x['verdict'] for x in v})}")

    rows = []
    for metric in METRICS:
        results = []
        for base, cand in COMPARISONS:
            b = [runs[base][k][metric] for k in reps]
            c = [runs[cand][k][metric] for k in reps]
            results.append((base, cand, *block_test(metric, b, c)))
        for (base, cand, d, lo, hi, p), p_adj in zip(results, bh([r[-1] for r in results])):
            rows.append(dict(metric=metric, baseline=base, candidate=cand,
                             primary=(base, cand) == COMPARISONS[0] and metric == "e2el_p50",
                             diff=round(d, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                             p=round(p, 5), p_adj=round(p_adj, 5)))
    for metric in ("e2el_p50_bound", "e2el_p95_bound"):
        for base, cand in COMPARISONS:
            b = [runs[base][k][metric] for k in reps]
            c = [runs[cand][k][metric] for k in reps]
            d, lo, hi, p = block_test(metric, b, c)
            rows.append(dict(metric=metric, baseline=base, candidate=cand, primary=False,
                             diff=round(d, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                             p=round(p, 5), p_adj=""))

    print("\nComparisons (candidate minus baseline; s, completion in pp):")
    for r in rows:
        adjusted = r["p_adj"] != ""
        sig = "*" if adjusted and r["p_adj"] < 0.05 else ""
        tag = " PRIMARY" if r["primary"] else ""
        p_adj = f"{r['p_adj']:.4f}" if adjusted else "  -   "
        print(f"  {r['metric']:15} {r['candidate']:20} - {r['baseline']:27} "
              f"{r['diff']:+8.3f} [{r['ci_lo']:+8.3f}, {r['ci_hi']:+8.3f}] "
              f"p={r['p']:.4f} p_adj={p_adj}{sig}{tag}")

    with (ROOT / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (ROOT / "per_run.csv").open("w", newline="") as handle:
        keys = ["arm", "rep"] + list(next(iter(runs[SA].values())))
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for arm in ARMS:
            for rep in reps:
                writer.writerow({"arm": arm, "rep": rep, **runs[arm][rep]})


if __name__ == "__main__":
    main()
