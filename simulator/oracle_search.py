import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import policies as P                                    # noqa: E402
import workload as W                                    # noqa: E402
from cluster import Cluster, load_service_model         # noqa: E402
from run_sim import pct                                 # noqa: E402

import random                                           # noqa: E402


def evaluate(model, wl, conc, seed, assignment, rate, n, deadline):
    """Score an assignment: (failures, e2el_p95, e2el_p50). Lower is better."""
    reqs = W.build(wl, n=n, rate=rate, seed=seed)
    cluster = Cluster(model, P.FixedAssignment(assignment), request_rate=rate,
                      max_concurrency=conc, upstream_timeout_s=deadline,
                      seed=seed)
    cluster.run(reqs)
    ok = [r for r in reqs if r.outcome == "success"]
    lat = [r.e2el_ms() for r in ok]
    fails = len(reqs) - len(ok)
    return (fails, pct(lat, 95) or 0.0, pct(lat, 50) or 0.0), reqs


def seed_assignment(model, wl, conc, seed, policy, rate, n, deadline):
    reqs = W.build(wl, n=n, rate=rate, seed=seed)
    cluster = Cluster(model, P.build(policy, deadline_s=deadline),
                      request_rate=rate, max_concurrency=conc,
                      upstream_timeout_s=deadline, seed=seed)
    cluster.run(reqs)
    return [r.node for r in sorted(reqs, key=lambda r: r.rid)]


def hill_climb(model, wl, conc, seed, rate, n, deadline, iters, nodes, rng,
               start):
    cur = list(start)
    best_score, _ = evaluate(model, wl, conc, seed, cur, rate, n, deadline)
    stalls = 0
    for _ in range(iters):
        i = rng.randrange(len(cur))
        alt = rng.choice([x for x in nodes if x != cur[i]])
        cand = list(cur)
        cand[i] = alt
        score, _ = evaluate(model, wl, conc, seed, cand, rate, n, deadline)
        if score < best_score:
            cur, best_score = cand, score
            stalls = 0
        else:
            stalls += 1
    return cur, best_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default="bimodal", choices=sorted(W.BUILDERS))
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--iters", type=int, default=1500,
                    help="hill-climb steps per rep")
    ap.add_argument("--rate", type=float, default=2.0)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--deadline", type=float, default=120.0)
    ap.add_argument("--baselines", default="weighted_least_connections,"
                                           "length_aware,oracle_greedy")
    args = ap.parse_args()

    model = load_service_model()
    nodes = list(model["nodes"])
    baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]
    rows = {b: [] for b in baselines}
    rows["ORACLE (search)"] = []

    print(f"c{args.conc}_{args.workload}: {args.reps} reps, "
          f"{args.iters} search steps/rep\n")
    for rep in range(args.reps):
        seed = 1000 + rep
        best_seed_name, best_seed_score, best_seed_vec = None, None, None
        for b in baselines:
            vec = seed_assignment(model, args.workload, args.conc, seed, b,
                                  args.rate, args.n, args.deadline)
            score, _ = evaluate(model, args.workload, args.conc, seed, vec,
                                args.rate, args.n, args.deadline)
            rows[b].append(score)
            if best_seed_score is None or score < best_seed_score:
                best_seed_name, best_seed_score, best_seed_vec = b, score, vec
        rng = random.Random(seed)
        _, oracle_score = hill_climb(
            model, args.workload, args.conc, seed, args.rate, args.n,
            args.deadline, args.iters, nodes, rng, best_seed_vec
        )
        rows["ORACLE (search)"].append(oracle_score)
        print(f"  rep{rep}: seeded from {best_seed_name} "
              f"(fails={best_seed_score[0]}, p95={best_seed_score[1]/1000:.1f}s)"
              f" -> oracle fails={oracle_score[0]}, "
              f"p95={oracle_score[1]/1000:.1f}s")

    print(f"\n{'policy':>28} {'completion':>11} {'e2el p95':>10} {'e2el p50':>10}")
    for name, scores in rows.items():
        comp = 100 * (1 - pct([s[0] for s in scores], 50) / args.n)
        p95 = pct([s[1] for s in scores], 50) / 1000
        p50 = pct([s[2] for s in scores], 50) / 1000
        print(f"{name:>28} {comp:10.1f}% {p95:9.1f}s {p50:9.1f}s")

    base = rows.get("weighted_least_connections")
    orc = rows["ORACLE (search)"]
    if base:
        b95 = pct([s[1] for s in base], 50)
        o95 = pct([s[1] for s in orc], 50)
        bc = 1 - pct([s[0] for s in base], 50) / args.n
        oc = 1 - pct([s[0] for s in orc], 50) / args.n
        print(f"\nHeadroom over WLC in this cell: "
              f"completion {100*(oc-bc):+.1f} pp, "
              f"e2el p95 {100*(o95-b95)/b95:+.1f}%")
        print("That is the ceiling for ANY per-request routing policy here, "
              "learned or not.")


if __name__ == "__main__":
    main()
