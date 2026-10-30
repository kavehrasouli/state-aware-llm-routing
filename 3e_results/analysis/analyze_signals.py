"""Compares the recorded routing signals: KV occupancy, engine request counts, and busy slots."""

import argparse
import csv
from scope import included_paths, CELLS
import glob
import json
import math
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # data root: 3e_results
OUT_HERE = os.path.dirname(os.path.abspath(__file__))  # outputs: everythingcombined
CONCS = [4, 8, 12]
WORKLOADS = ["uniform", "bimodal", "sharegpt"]
CELL_COLS = CELLS

_WLAB = {"uniform": "uni", "bimodal": "bim", "sharegpt": "sgpt"}


def col_label(workload, conc):
    return f"c{conc}_{_WLAB[workload]}"


def is_grid_cell(workload, conc):
    if workload in ("uniform", "bimodal"):
        return conc in CONCS
    return workload == "sharegpt" and conc in CONCS


# ───────────────────────── small stats, stdlib only ─────────────────────────
def mean(xs):
    return sum(xs) / len(xs) if xs else None


def pct(xs, p):
    if not xs:
        return None
    s = sorted(xs)
    if len(s) == 1:
        return float(s[0])
    i = (p / 100.0) * (len(s) - 1)
    lo = int(i)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


def _ranks(xs):
    """Average ranks, ties shared — required for a correct Spearman on data
    with many equal values (B is a small integer, so ties are everywhere)."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        shared = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def spearman(xs, ys):
    """Rank correlation. Returns None when either side is constant (no ranks to
    correlate), which happens in idle cells where every observation is zero."""
    if len(xs) < 3:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


# ───────────────────────── load ─────────────────────────
def load(clean_only):
    """Yield (cell, arm, decision) where decision is the list of per-replica
    observations recorded for one routing decision."""
    out = defaultdict(list)
    diag = {"logs": 0, "decisions": 0, "observations": 0, "skipped_runs": 0}
    for meta_path in [str(p) for p in included_paths()]:
        try:
            meta = json.loads(open(meta_path).read())
        except (OSError, json.JSONDecodeError):
            continue
        workload, conc = meta.get("workload"), meta.get("concurrency")
        if not is_grid_cell(workload, conc):
            continue
        if clean_only and meta.get("verdict") != "clean":
            diag["skipped_runs"] += 1
            continue
        arm = meta.get("arm") or meta.get("policy")
        log = os.path.join(os.path.dirname(meta_path), "router_log.jsonl")
        if not os.path.isfile(log):
            continue
        diag["logs"] += 1
        bench = json.loads(open(os.path.join(os.path.dirname(meta_path), meta["bench_result_file"])).read())
        num_warmup = bench.get("num_warmup", 3)
        for line in open(log):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") != "request":
                continue
            if not isinstance(rec.get("request_id"), int) or rec["request_id"] < num_warmup:
                continue
            sv = rec.get("state_vector") or []
            if not sv:
                continue
            diag["decisions"] += 1
            diag["observations"] += len(sv)
            out[(workload, conc, arm)].append(sv)
    return out, diag


# ───────────────────────── the three questions ─────────────────────────
def q1_signal_overlap(decisions):
    """u vs b_metrics: rank correlation over observations, plus how often the
    two argmin rules would choose the same replica from the same snapshot."""
    us, bs = [], []
    agree = total = 0
    for sv in decisions:
        elig = [c for c in sv if c.get("eligible")]
        for c in elig:
            if isinstance(c.get("u"), (int, float)) and isinstance(c.get("b_metrics"), (int, float)):
                us.append(c["u"])
                bs.append(c["b_metrics"])
        if len(elig) >= 2:
            # lowest-index tie-break on both sides, matching the router's argmin
            pick_u = min(range(len(elig)), key=lambda i: (elig[i].get("u", 0.0), i))
            pick_b = min(range(len(elig)), key=lambda i: (elig[i].get("b_metrics", 0.0), i))
            total += 1
            agree += (pick_u == pick_b)
    return {
        "n_obs": len(us),
        "rho": spearman(us, bs),
        "agree_frac": (agree / total) if total else None,
        "n_decisions": total,
    }


def q2_engine_vs_reality(decisions):
    """b_metrics (engine's 'processing') vs b_slots_active (slots truly busy)."""
    diffs, disagree, total = [], 0, 0
    both_zero = 0
    for sv in decisions:
        for c in sv:
            bm, ba = c.get("b_metrics"), c.get("b_slots_active")
            if not isinstance(bm, (int, float)) or not isinstance(ba, (int, float)):
                continue
            total += 1
            d = bm - ba
            diffs.append(d)
            if abs(d) > 1e-9:
                disagree += 1
            if bm == 0 and ba == 0:
                both_zero += 1
    return {
        "n": total,
        "disagree_frac": (disagree / total) if total else None,
        "mean_diff": mean(diffs),
        "p95_abs_diff": pct([abs(d) for d in diffs], 95),
        "max_abs_diff": max((abs(d) for d in diffs), default=None),
        "both_zero_frac": (both_zero / total) if total else None,
    }


def q3_imbalance(decisions):
    """Spread across eligible replicas at decision time. If the cluster is
    balanced there is nothing for any policy to exploit."""
    su, sb = [], []
    for sv in decisions:
        elig = [c for c in sv if c.get("eligible")]
        if len(elig) < 2:
            continue
        u = [c["u"] for c in elig if isinstance(c.get("u"), (int, float))]
        b = [c["b_metrics"] for c in elig if isinstance(c.get("b_metrics"), (int, float))]
        if len(u) >= 2:
            su.append(max(u) - min(u))
        if len(b) >= 2:
            sb.append(max(b) - min(b))
    return {
        "n": len(su),
        "u_p50": pct(su, 50), "u_p95": pct(su, 95),
        "u_zero_frac": (sum(1 for x in su if x == 0) / len(su)) if su else None,
        "b_p50": pct(sb, 50), "b_p95": pct(sb, 95),
        "b_zero_frac": (sum(1 for x in sb if x == 0) / len(sb)) if sb else None,
    }


# ───────────────────────── formatting ─────────────────────────
def f(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def pctf(x):
    return "n/a" if x is None else f"{x * 100:.1f}%"


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean-only", action="store_true",
                    help="restrict to runs that passed the validity gate "
                         "(default: all runs, since signal behaviour under "
                         "overload is itself of interest)")
    args = ap.parse_args()

    data, diag = load(args.clean_only)
    subset = "clean runs only" if args.clean_only else "all runs (clean and suspect)"

    L = ["# SQ1 — are the routed-on signals informative?\n",
         f"Source: {diag['logs']} router logs, **{diag['decisions']} routing decisions**, "
         f"**{diag['observations']} replica-observations** ({subset}). Read-only over the "
         "same files as `3e_report.md`; nothing here re-analyses latency.\n",
         "\nThese values were recorded on every decision but never analysed: the router "
         "logged, per replica, KV occupancy `u`, the engine's in-flight count "
         "`b_metrics`, and the genuinely-busy slot count `b_slots_active`.\n"]

    # ---- Q1 ----
    L.append("\n## 1. Do `u` and `B` carry the same information?\n")
    L.append("`state_aware` routes on argmin(u/w); `least_connections` on argmin(B/w). "
             "If the two signals move together, the policies were reading nearly the same "
             "thing and could not have diverged. `rho` is Spearman rank correlation over "
             "replica-observations; `same pick` is how often both rules select the same "
             "replica from the identical snapshot.\n")
    rows = []
    for workload, conc in CELL_COLS:
        for arm in sorted({a for (w, c, a) in data if (w, c) == (workload, conc)}):
            r = q1_signal_overlap(data[(workload, conc, arm)])
            rows.append([col_label(workload, conc), arm, r["n_obs"],
                         f(r["rho"], 3), pctf(r["agree_frac"]), r["n_decisions"]])
    L.append(table(["cell", "arm", "observations", "rho(u,B)", "same pick", "decisions"], rows))

    # ---- Q2 ----
    L.append("\n## 2. Does the engine's own count match reality?\n")
    L.append("`b_metrics` is what llama.cpp reports as processing; `b_slots_active` is how "
             "many slots were actually busy. A persistent gap means the concurrency signal "
             "the least-connections family routes on is not the concurrency the engine is "
             "carrying. `diff` is b_metrics - b_slots_active.\n")
    rows = []
    for workload, conc in CELL_COLS:
        for arm in sorted({a for (w, c, a) in data if (w, c) == (workload, conc)}):
            r = q2_engine_vs_reality(data[(workload, conc, arm)])
            rows.append([col_label(workload, conc), arm, r["n"],
                         pctf(r["disagree_frac"]), f(r["mean_diff"], 2),
                         f(r["p95_abs_diff"], 1), f(r["max_abs_diff"], 1),
                         pctf(r["both_zero_frac"])])
    L.append(table(["cell", "arm", "obs", "disagree", "mean diff", "p95 |diff|",
                    "max |diff|", "both zero"], rows))

    # ---- Q3 ----
    L.append("\n## 3. Was there enough imbalance to act on?\n")
    L.append("Spread (max-min) across eligible replicas at the moment of each decision. "
             "`zero` is the share of decisions where every replica looked identical on that "
             "signal -- decisions in which no state-aware policy could have differed from a "
             "blind one.\n")
    rows = []
    for workload, conc in CELL_COLS:
        for arm in sorted({a for (w, c, a) in data if (w, c) == (workload, conc)}):
            r = q3_imbalance(data[(workload, conc, arm)])
            rows.append([col_label(workload, conc), arm, r["n"],
                         f(r["u_p50"], 3), f(r["u_p95"], 3), pctf(r["u_zero_frac"]),
                         f(r["b_p50"], 1), f(r["b_p95"], 1), pctf(r["b_zero_frac"])])
    L.append(table(["cell", "arm", "decisions", "u spread p50", "u spread p95",
                    "u zero", "B spread p50", "B spread p95", "B zero"], rows))

    report = "\n".join(L) + "\n"
    open(os.path.join(OUT_HERE, "3e_signals.md"), "w").write(report)

    with open(os.path.join(OUT_HERE, "3e_signals.csv"), "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["workload", "concurrency", "arm", "metric", "value"])
        for (workload, conc, arm), decisions in sorted(data.items(), key=lambda kv: str(kv[0])):
            for name, res in (("q1", q1_signal_overlap(decisions)),
                              ("q2", q2_engine_vs_reality(decisions)),
                              ("q3", q3_imbalance(decisions))):
                for k, v in res.items():
                    w.writerow([workload, conc, arm, f"{name}_{k}", v])

    print(report)
    print(f"[wrote 3e_signals.md and 3e_signals.csv | {diag['decisions']} decisions, "
          f"{diag['observations']} observations, {diag['skipped_runs']} runs skipped]")


if __name__ == "__main__":
    main()
