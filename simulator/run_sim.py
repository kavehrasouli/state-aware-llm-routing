import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import policies as P                                    # noqa: E402
import workload as W                                    # noqa: E402
from cluster import Cluster, load_service_model         # noqa: E402

DEFAULT_POLICIES = [
    "random", "round_robin", "weighted_round_robin",
    "least_connections", "weighted_least_connections", "state_aware",
    "length_aware", "oracle_greedy",
]
GRID = [(w, c) for w in ("uniform", "bimodal") for c in (4, 8, 12)] + \
       [("sharegpt", 8)]


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    if len(xs) == 1:
        return float(xs[0])
    r = (q / 100.0) * (len(xs) - 1)
    lo = int(r)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (r - lo)


def summarise(requests, cls=None):
    sel = [r for r in requests if cls is None or r.cls == cls]
    if not sel:
        return None
    ok = [r for r in sel if r.outcome == "success"]
    lat = [r.e2el_ms() for r in ok]
    tpots = [r.tpot_ms() for r in ok if r.tpot_ms() is not None]
    return {
        "offered": len(sel),
        "success": len(ok),
        "completion": len(ok) / len(sel),
        "e2el_p50": pct(lat, 50),
        "e2el_p95": pct(lat, 95),
        "tpot_p50": pct(tpots, 50),
        "tpot_p95": pct(tpots, 95),
    }


def run_one(model, policy_name, wl, conc, seed, rate, n, deadline):
    reqs = W.build(wl, n=n, rate=rate, seed=seed)
    pol = P.build(policy_name, deadline_s=deadline)
    cluster = Cluster(model, pol, request_rate=rate, max_concurrency=conc,
                      upstream_timeout_s=deadline, seed=seed)
    cluster.run(reqs)
    share = {}
    for r in reqs:
        share[r.node] = share.get(r.node, 0) + 1
    return reqs, share


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default="bimodal", choices=sorted(W.BUILDERS))
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--all", action="store_true", help="run the whole 3e grid")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--rate", type=float, default=2.0)
    ap.add_argument("--n", type=int, default=100, help="requests per rep")
    ap.add_argument("--deadline", type=float, default=120.0)
    ap.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    ap.add_argument("--csv", help="write per-rep rows here")
    args = ap.parse_args()

    model = load_service_model()
    cells = GRID if args.all else [(args.workload, args.conc)]
    names = [p.strip() for p in args.policies.split(",") if p.strip()]
    rows = []

    for wl, conc in cells:
        print(f"\n=== c{conc}_{wl} ({args.reps} reps x {args.n} requests) ===")
        header = (f"{'policy':>28} {'compl':>7} {'e2el p50':>10} "
                  f"{'e2el p95':>10} {'tpot p50':>9}  node split")
        print(header)
        for name in names:
            per_rep = []
            shares = {}
            for rep in range(args.reps):
                reqs, share = run_one(model, name, wl, conc, 1000 + rep,
                                      args.rate, args.n, args.deadline)
                s = summarise(reqs)
                per_rep.append(s)
                for k, v in share.items():
                    shares[k] = shares.get(k, 0) + v
                row = {"policy": name, "workload": wl, "conc": conc,
                       "rep": rep, **s}
                for cls in sorted({r.cls for r in reqs}):
                    cs = summarise(reqs, cls)
                    row[f"{cls}_completion"] = cs["completion"]
                    row[f"{cls}_e2el_p95"] = cs["e2el_p95"]
                rows.append(row)
            med = lambda k: pct([r[k] for r in per_rep if r[k] is not None], 50)
            tot = sum(shares.values()) or 1
            split = "/".join(
                f"{100 * shares.get(n, 0) / tot:.0f}"
                for n in sorted(shares)
            )
            print(f"{name:>28} {100 * med('completion'):6.1f}% "
                  f"{med('e2el_p50') / 1000:9.1f}s {med('e2el_p95') / 1000:9.1f}s "
                  f"{med('tpot_p50'):8.1f}ms  {split}")

    if args.csv:
        keys = sorted({k for r in rows for k in r})
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
