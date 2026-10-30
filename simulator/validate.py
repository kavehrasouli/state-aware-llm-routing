import argparse
import glob
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import policies as P                                    # noqa: E402
import workload as W                                    # noqa: E402
from cluster import Cluster, load_service_model         # noqa: E402
from run_sim import pct, summarise, run_one             # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
REAL_POLICIES = [
    "random", "round_robin", "weighted_round_robin",
    "least_connections", "weighted_least_connections", "state_aware",
]
CELLS = [("uniform", 4), ("uniform", 8), ("uniform", 12),
         ("bimodal", 4), ("bimodal", 8), ("bimodal", 12),
         ("sharegpt", 8)]


def real_cell(policy, workload, conc, clean_only=True):
    """Per-rep (completion, e2el_p50, e2el_p95) from the logged runs."""
    pat = os.path.join(REPO, "3e_results", policy, f"c{conc}_{workload}", "rep*")
    out = []
    for d in sorted(glob.glob(pat)):
        meta_path = os.path.join(d, "run_meta.json")
        if not os.path.exists(meta_path):
            continue
        meta = json.loads(open(meta_path).read())
        if clean_only and meta.get("verdict") != "clean":
            continue
        bench = [f for f in glob.glob(os.path.join(d, "*.json"))
                 if "run_meta" not in f]
        if not bench:
            continue
        raw = json.loads(open(bench[0]).read()).get("raw", [])
        if not raw:
            continue
        ok = [r for r in raw if r.get("success") and not r.get("truncated")]
        lat = [r["e2el_ms"] for r in ok]
        out.append({
            "completion": len(ok) / len(raw),
            "e2el_p50": pct(lat, 50),
            "e2el_p95": pct(lat, 95),
        })
    return out


def med(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return pct(vals, 50) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--rate", type=float, default=2.0)
    ap.add_argument("--include-suspect", action="store_true",
                    help="use all real runs, not just clean ones")
    args = ap.parse_args()

    model = load_service_model()
    errs = {"completion": [], "e2el_p50": [], "e2el_p95": []}
    missing = []
    sim_cache = {}

    for workload, conc in CELLS:
        print(f"\n=== c{conc}_{workload} ===")
        print(f"{'policy':>28} | {'completion':>19} | {'e2el p50 (s)':>19} | "
              f"{'e2el p95 (s)':>19}")
        print(f"{'':>28} | {'real':>8} {'sim':>8} | {'real':>8} {'sim':>8} | "
              f"{'real':>8} {'sim':>8}")
        for policy in REAL_POLICIES:
            real = real_cell(policy, workload, conc,
                             clean_only=not args.include_suspect)
            if not real:
                missing.append(f"{policy} c{conc}_{workload}")
                continue
            sim = []
            for rep in range(args.reps):
                reqs, _ = run_one(model, policy, workload, conc, 1000 + rep,
                                  args.rate, 100, 120.0)
                sim.append(summarise(reqs))
            sim_cache.setdefault((workload, conc), {})[policy] = sim
            cells = []
            for key in ("completion", "e2el_p50", "e2el_p95"):
                r, s = med(real, key), med(sim, key)
                if r is not None and s is not None and r != 0:
                    errs[key].append(abs(s - r) / r)
                if key == "completion":
                    cells.append(f"{100 * r:7.1f}% {100 * s:7.1f}%")
                else:
                    cells.append(f"{r / 1000:8.1f} {s / 1000:8.1f}")
            print(f"{policy:>28} | {cells[0]} | {cells[1]} | {cells[2]}")

    print("\n" + "=" * 72)
    print("Median absolute relative error, simulator vs hardware "
          f"({len(errs['e2el_p50'])} policy-cells):")
    for key, vals in errs.items():
        if vals:
            print(f"  {key:>12}: median {100 * pct(vals, 50):5.1f}%  "
                  f"p90 {100 * pct(vals, 90):5.1f}%  max {100 * max(vals):5.1f}%")

    print("\nRank agreement (Kendall tau, real vs simulated policy ordering "
          "within each cell).\nThis is the property that matters for "
          "screening: does the simulator put the\npolicies in the right order, "
          "even when the absolute numbers drift?")
    for key in ("completion", "e2el_p50", "e2el_p95"):
        taus = []
        for workload, conc in CELLS:
            pairs = []
            for policy in REAL_POLICIES:
                real = real_cell(policy, workload, conc,
                                 clean_only=not args.include_suspect)
                if not real or policy not in sim_cache.get((workload, conc), {}):
                    continue
                r = med(real, key)
                s = med(sim_cache[(workload, conc)][policy], key)
                if r is not None and s is not None:
                    pairs.append((r, s))
            if len(pairs) < 3:
                continue
            conc_pairs = dis = 0
            for (r1, s1), (r2, s2) in itertools.combinations(pairs, 2):
                sign = (r1 - r2) * (s1 - s2)
                if sign > 0:
                    conc_pairs += 1
                elif sign < 0:
                    dis += 1
            if conc_pairs + dis:
                taus.append((conc_pairs - dis) / (conc_pairs + dis))
        if taus:
            print(f"  {key:>12}: median tau {pct(taus, 50):+.2f} over "
                  f"{len(taus)} cells  (values {[round(t, 2) for t in taus]})")
    if missing:
        print(f"\nno clean real runs for {len(missing)} policy-cells: "
              f"{', '.join(missing)}")
    print("\nRead these as screening tolerances. The simulator is for ranking "
          "policies and sizing experiments, not for quoting latencies.")


if __name__ == "__main__":
    main()
