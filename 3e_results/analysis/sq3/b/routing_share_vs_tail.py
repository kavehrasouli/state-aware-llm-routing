"""Relates each policy's Pi 5 dispatch share to its latency difference from WLC (descriptive).

Output: sq3b_routing_share_vs_tail.csv
"""
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHARES = HERE / "sq3b_dispatch_share.csv"
TESTS = HERE / "with_correction" / "sq3b_bh_corrected.csv"
OUT = HERE / "sq3b_routing_share_vs_tail.csv"

BASE = "weighted_least_connections"
ARMS = ("state_aware", "state_aware_v2", "cost_predict_v3", "wlc_conc")
FAST_NODE = "pi-node-1"
SHARE_MIN = 0.02   # share difference (fraction) below which direction is ignored
ALPHA = 0.05
# +1: difference expected to rise with Pi 5 share; -1: expected to fall.
EXPECT = {"e2el_p50": +1, "e2el_p95": -1}


def load_shares():
    with SHARES.open(newline="") as f:
        return {(r["arm"], r["cell"]): float(r["share_pooled"])
                for r in csv.DictReader(f)
                if r["node"] == FAST_NODE and r["share_pooled"]}


def load_tests():
    with TESTS.open(newline="") as f:
        return {(r["arm"], r["cell"], r["metric"]): (float(r["diff"]), float(r["p_adj_B"]))
                for r in csv.DictReader(f)
                if r["base_arm"] == BASE and r["metric"] in EXPECT
                and r["diff"] and r["p_adj_B"]}


def main():
    shares, tests = load_shares(), load_tests()
    rows = []
    for (arm, cell, metric), (diff, p_adj) in sorted(tests.items()):
        if arm not in ARMS or (arm, cell) not in shares or (BASE, cell) not in shares:
            continue
        d_share = shares[(arm, cell)] - shares[(BASE, cell)]
        counted = abs(d_share) >= SHARE_MIN
        rows.append({
            "arm": arm, "cell": cell, "metric": metric,
            "pi5_share_arm": round(shares[(arm, cell)], 6),
            "pi5_share_wlc": round(shares[(BASE, cell)], 6),
            "pi5_share_diff_pp": round(100 * d_share, 3),
            "latency_diff_s": round(diff / 1000, 4),
            "p_adj_B": p_adj,
            "significant": p_adj < ALPHA,
            "counted": counted,
            "agrees": (diff * d_share * EXPECT[metric]) > 0 if counted else "",
        })

    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {OUT.name}: {len(rows)} rows "
          f"(share threshold {100 * SHARE_MIN:.0f} pp, alpha {ALPHA}).")
    for metric in EXPECT:
        for label, keep in (("significant", lambda r: r["significant"]), ("all", lambda r: True)):
            sel = [r for r in rows if r["metric"] == metric and r["counted"] and keep(r)]
            agree = sum(r["agrees"] is True for r in sel)
            print(f"  {metric:9} {label:11}: {agree} of {len(sel)} agree")


if __name__ == "__main__":
    main()
