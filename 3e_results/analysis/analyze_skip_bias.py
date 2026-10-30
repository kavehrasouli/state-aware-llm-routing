"""Explains which replay decisions were skipped (stale or ineligible replicas) and why."""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import analyze_divergence as AD

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # data root: 3e_results
OUT_HERE = os.path.dirname(os.path.abspath(__file__))  # outputs: everythingcombined
SOURCE_CSV = os.path.join(OUT_HERE, "divergence_analysis.csv")
DEFAULT_OUT = os.path.join(OUT_HERE, "skip_bias_analysis.csv")

# Verification tolerance: the same helpers are reused, so values must agree to float precision.
TOL = 1e-9

RULES = ("A_strict", "B_eligible_only", "C_eligible_subset")


# ── node predicates ──
# Same eligibility and staleness checks as analyze_divergence.py; missing fields count as failing.
def node_eligible(e):
    return e.get("eligible") is True


def node_stale(e):
    return e.get("stale") is not False


def classify(by_name):
    """Exclusive gate outcome for a well-formed, complete state vector."""
    any_inel = any(not node_eligible(by_name[n]) for n in AD.NODES)
    any_stale = any(node_stale(by_name[n]) for n in AD.NODES)
    if any_inel and any_stale:
        return "skip_both", any_inel, any_stale
    if any_inel:
        return "skip_ineligible_only", any_inel, any_stale
    if any_stale:
        return "skip_stale_only", any_inel, any_stale
    return None, any_inel, any_stale  # gate passes


# ───────────────────────── per-run pass ───────────────────────────────────
def scan_run(meta, router_log_path):
    """One pass over a run's log: Q1 classification, Q2 rates under all three
    rules, and stale-age samples."""
    weights = (meta.get("frozen_config") or {}).get("weights")
    if not weights or len(weights) != 3:
        return None
    weight_by_node = dict(zip(AD.NODES, weights))

    counts = defaultdict(int)
    # per-rule accumulators: 1/0 per included decision
    acc = {r: {"div_router": [], "div_unc": [], "tie_router": [], "tie_unc": []}
           for r in RULES}
    # Rule A magnitudes, mirroring analyze_divergence.py's own collection.
    mag_router, mag_unc = [], []
    stale_age = {n: [] for n in AD.NODES}   # age_ms of nodes flagged stale
    stale_age_missing = {n: 0 for n in AD.NODES}
    n_none_b_after_gate = 0
    thresholds = set()

    with open(router_log_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("type") != "request":
                continue

            if r.get("max_state_age_ms") is not None:
                thresholds.add(r.get("max_state_age_ms"))

            sv = r.get("state_vector")
            if not sv or not isinstance(sv, list):
                counts["skip_malformed_sv"] += 1
                continue
            by_name = {e.get("replica"): e for e in sv}
            if not all(n in by_name for n in AD.NODES):
                counts["skip_incomplete_sv"] += 1
                continue

            gate_fail, any_inel, any_stale = classify(by_name)

            if any_stale:
                for n in AD.NODES:
                    if node_stale(by_name[n]):
                        age = by_name[n].get("age_ms")
                        if isinstance(age, (int, float)):
                            stale_age[n].append(float(age))
                        else:
                            stale_age_missing[n] += 1

            # ---- scores, identical for every rule ----
            b_score = {n: by_name[n].get("b_metrics") for n in AD.NODES}
            deferred_score = {
                n: (by_name[n].get("b_metrics") or 0.0) + (by_name[n].get("requests_deferred") or 0.0)
                for n in AD.NODES
            }
            rw_score = {}
            for n in AD.NODES:
                u = by_name[n].get("u")
                w = weight_by_node.get(n)
                rw_score[n] = None if (u is None or w is None or w == 0) else u / w

            all_idx = list(range(3))
            elig_idx = [i for i in all_idx if node_eligible(by_name[AD.NODES[i]])]

            rule_candidates = {
                "A_strict": all_idx if gate_fail is None else None,
                "B_eligible_only": all_idx if len(elig_idx) == 3 else None,
                "C_eligible_subset": elig_idx if len(elig_idx) >= 2 else None,
            }

            for rule, cands in rule_candidates.items():
                if cands is None:
                    continue
                cand_b = AD.argmin_over(cands, lambda i: b_score[AD.NODES[i]])
                cand_bu = AD.argmin_over(cands, lambda i: deferred_score[AD.NODES[i]])
                cand_rw = AD.argmin_over(cands, lambda i: rw_score[AD.NODES[i]])
                if cand_b is None or cand_bu is None or cand_rw is None:
                    if rule == "A_strict":
                        counts["skip_undefined_argmin"] += 1
                    continue

                if rule == "A_strict":
                    counts["analysed"] += 1

                # b_tie: two or more candidates share the minimum B score.
                bs = [b_score[AD.NODES[i]] for i in cands if b_score[AD.NODES[i]] is not None]
                bus = [deferred_score[AD.NODES[i]] for i in cands
                       if deferred_score[AD.NODES[i]] is not None]
                if rule == "A_strict" and len(bs) != len(cands):
                    n_none_b_after_gate += 1
                acc[rule]["tie_router"].append(
                    1 if bs and sum(1 for s in bs if s == min(bs)) >= 2 else 0)
                acc[rule]["tie_unc"].append(
                    1 if bus and sum(1 for s in bus if s == min(bus)) >= 2 else 0)

                d_router = 1 if cand_b != cand_rw else 0
                d_unc = 1 if cand_bu != cand_rw else 0
                acc[rule]["div_router"].append(d_router)
                acc[rule]["div_unc"].append(d_unc)

                if rule == "A_strict":
                    if d_router:
                        a, b = rw_score[AD.NODES[cand_b]], rw_score[AD.NODES[cand_rw]]
                        if a is not None and b is not None and b > 0:
                            mag_router.append(a / b)
                    if d_unc:
                        a, b = rw_score[AD.NODES[cand_bu]], rw_score[AD.NODES[cand_rw]]
                        if a is not None and b is not None and b > 0:
                            mag_unc.append(a / b)

            if gate_fail is not None:
                counts[gate_fail] += 1

    total = sum(counts[k] for k in
                ("analysed", "skip_malformed_sv", "skip_incomplete_sv",
                 "skip_ineligible_only", "skip_stale_only", "skip_both",
                 "skip_undefined_argmin"))
    if total == 0:
        return None

    def rate(rule, key):
        v = acc[rule][key]
        return (sum(v) / len(v)) if v else None

    return {
        "counts": dict(counts),
        "total": total,
        "n_analysed_by_rule": {r: len(acc[r]["div_router"]) for r in RULES},
        "rates": {r: {k: rate(r, k) for k in
                      ("div_router", "div_unc", "tie_router", "tie_unc")} for r in RULES},
        "mag_median_router": AD.median(mag_router),
        "mag_median_unc": AD.median(mag_unc),
        "stale_age": stale_age,
        "stale_age_missing": stale_age_missing,
        "n_none_b_after_gate": n_none_b_after_gate,
        "thresholds": thresholds,
    }


# ───────────────────────── aggregation ────────────────────────────────────
COUNT_KEYS = ["analysed", "skip_malformed_sv", "skip_incomplete_sv",
              "skip_ineligible_only", "skip_stale_only", "skip_both",
              "skip_undefined_argmin"]


def build_cells(scans):
    """Group per-run scans into (clean, arm, cell), mirroring
    analyze_divergence.py's cell iteration so every cell -- including empty
    ones -- appears exactly once per block."""
    grouped = defaultdict(list)
    for s in scans:
        grouped[(s["clean"], s["arm"], s["cell"])].append(s)

    rows = []
    for clean in (True, False):
        arms = sorted({k[1] for k in grouped if k[0] == clean})
        for arm in arms:
            for w, c in AD.CELL_COLS:
                cell = AD.col_label(w, c)
                rows.append(aggregate_cell(clean, arm, cell, grouped.get((clean, arm, cell), [])))
    return rows


def aggregate_cell(clean, arm, cell, runs):
    row = {"clean": clean, "arm": arm, "cell": cell, "n_runs": len(runs)}

    for k in COUNT_KEYS:
        row[k] = sum(r["counts"].get(k, 0) for r in runs)
    row["skip_any_ineligible"] = row["skip_ineligible_only"] + row["skip_both"]
    row["skip_any_stale"] = row["skip_stale_only"] + row["skip_both"]
    row["skip_total"] = sum(row[k] for k in COUNT_KEYS if k != "analysed")
    total = sum(r["total"] for r in runs)
    row["total_decisions"] = total

    for k in (COUNT_KEYS + ["skip_any_ineligible", "skip_any_stale", "skip_total"]):
        row["share_" + k] = (row[k] / total) if total else None

    # Stale-age distribution per node, pooled over this cell's stale flags.
    for n in AD.NODES:
        ages = [a for r in runs for a in r["stale_age"][n]]
        tag = n.replace("pi-node-", "n")
        row[f"stale_flags_{tag}"] = len(ages)
        row[f"age_median_{tag}"] = AD.median(ages)
        row[f"age_p90_{tag}"] = AD.pct(ages, 90)
        row[f"age_max_{tag}"] = max(ages) if ages else None
        row[f"age_missing_{tag}"] = sum(r["stale_age_missing"][n] for r in runs)

    # Q2: per-rule cell stats. agg() is analyze_divergence.py's own
    # mean/sample-SD over RUN-level rates -- the run stays the unit.
    for rule in RULES:
        row[f"{rule}_n_analysed"] = sum(r["n_analysed_by_rule"][rule] for r in runs)
        for key, label in (("div_router", "div_router"), ("div_unc", "div_unc"),
                           ("tie_router", "tie_router"), ("tie_unc", "tie_unc")):
            a = AD.agg([r["rates"][rule][key] for r in runs
                        if r["rates"][rule][key] is not None])
            row[f"{rule}_{label}_mean"] = a["mean"]
            row[f"{rule}_{label}_std"] = a["std"] if a["n"] else None

    mr = [r["mag_median_router"] for r in runs if r["mag_median_router"] is not None]
    mu = [r["mag_median_unc"] for r in runs if r["mag_median_unc"] is not None]
    row["A_strict_mag_median_router"] = AD.median(mr)
    row["A_strict_mag_p90_router"] = AD.pct(mr, 90)
    row["A_strict_mag_median_unc"] = AD.median(mu)
    row["A_strict_mag_p90_unc"] = AD.pct(mu, 90)
    return row


# ───────────────────────── verification against the frozen CSV ────────────
# Rule A column -> divergence_analysis.csv column.
VERIFY_MAP = [
    ("n_runs", "n_runs"),
    ("analysed", "n_analyzed"),
    ("skip_total", "n_skipped"),
    ("A_strict_div_router_mean", "divergence_rate_router_mean"),
    ("A_strict_div_router_std", "divergence_rate_router_std"),
    ("A_strict_div_unc_mean", "divergence_rate_uncensored_mean"),
    ("A_strict_div_unc_std", "divergence_rate_uncensored_std"),
    ("A_strict_tie_router_mean", "b_tie_rate_router_mean"),
    ("A_strict_tie_router_std", "b_tie_rate_router_std"),
    ("A_strict_tie_unc_mean", "b_tie_rate_uncensored_mean"),
    ("A_strict_tie_unc_std", "b_tie_rate_uncensored_std"),
    ("A_strict_mag_median_router", "magnitude_median_router"),
    ("A_strict_mag_p90_router", "magnitude_p90_router"),
    ("A_strict_mag_median_unc", "magnitude_median_uncensored"),
    ("A_strict_mag_p90_unc", "magnitude_p90_uncensored"),
]


def verify(rows):
    if not os.path.exists(SOURCE_CSV):
        return None, [f"{SOURCE_CSV} not found -- cannot verify Rule A."]
    with open(SOURCE_CSV) as f:
        ref = {(r["clean"], r["arm"], r["cell"]): r for r in csv.DictReader(f)}
    mine = {(str(r["clean"]), r["arm"], r["cell"]): r for r in rows}

    problems = []
    only_ref = sorted(set(ref) - set(mine))
    only_mine = sorted(set(mine) - set(ref))
    for k in only_ref:
        problems.append(f"cell in divergence_analysis.csv but not recomputed: {k}")
    for k in only_mine:
        problems.append(f"cell recomputed but absent from divergence_analysis.csv: {k}")

    worst = (0.0, None)
    n_checked = 0
    for k in sorted(set(ref) & set(mine)):
        for my_col, ref_col in VERIFY_MAP:
            rv = (ref[k].get(ref_col) or "").strip()
            mv = mine[k][my_col]
            if rv == "" and mv is None:
                continue
            if rv == "" or mv is None:
                problems.append(f"{k} {ref_col}: csv={rv or '(empty)'} recomputed="
                                f"{'(empty)' if mv is None else mv}")
                continue
            d = abs(float(rv) - float(mv))
            n_checked += 1
            if d > worst[0]:
                worst = (d, f"{k} {ref_col}")
            if d > TOL:
                problems.append(f"{k} {ref_col}: csv={float(rv)!r} recomputed={float(mv)!r} "
                                f"absdiff={d:.3e}")
    return {"n_cells": len(set(ref) & set(mine)), "n_values": n_checked,
            "worst": worst}, problems


# ───────────────────────── output ─────────────────────────────────────────
def fmt(v, nd=3):
    if v is None:
        return "-"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def pooled(rows, clean):
    """Pool a block's rows by cell across arms, for the stdout summary."""
    out = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if r["clean"] is not clean:
            continue
        d = out[r["cell"]]
        for k in COUNT_KEYS + ["skip_any_ineligible", "skip_any_stale",
                               "skip_total", "total_decisions"]:
            d[k] += r[k]
        for rule in RULES:
            d[f"{rule}_n_analysed"] += r[f"{rule}_n_analysed"]
    return out


def main():
    ap = argparse.ArgumentParser(description="§3e skip-bias diagnostic (read-only).")
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"output CSV (default {DEFAULT_OUT})")
    args = ap.parse_args()
    if os.path.abspath(args.out) == os.path.abspath(SOURCE_CSV):
        sys.exit("FATAL: refusing to overwrite divergence_analysis.csv.")

    runs = AD.load_all_runs()          # canonical run selection, imported not copied
    usable = [r for r in runs if r.get("usable")]
    print("=" * 100)
    print("§3e SKIP-BIAS DIAGNOSTIC — read-only; analyze_divergence.py imported, not modified.")
    print("=" * 100)
    print(f"{len(runs)} run directories; {len(usable)} usable (selection inherited from "
          f"analyze_divergence.load_all_runs()).")

    scans, thresholds, none_b = [], set(), 0
    for r in usable:
        meta = json.loads(open(os.path.join(r["dir"], "run_meta.json")).read())
        path = os.path.join(r["dir"], meta["router_log_file"])
        s = scan_run(meta, path)
        if s is None:
            continue
        s.update({"clean": r.get("verdict") == "clean", "arm": r["arm"],
                  "cell": AD.col_label(r["workload"], r["conc"]),
                  "ref_n_analyzed": r["n_analyzed"], "ref_n_skipped": r["n_skipped"]})
        thresholds |= s["thresholds"]
        none_b += s["n_none_b_after_gate"]
        scans.append(s)

    # Per-run cross-check against the imported module's own counters.
    run_mismatch = [s for s in scans
                    if s["counts"].get("analysed", 0) != s["ref_n_analyzed"]
                    or s["total"] - s["counts"].get("analysed", 0) != s["ref_n_skipped"]]

    rows = build_cells(scans)
    stats, problems = verify(rows)

    print("\n" + "#" * 100)
    print("# VERIFICATION — Rule A vs divergence_analysis.csv (reported before anything else)")
    print("#" * 100)
    print(f"Per-run counters vs analyze_divergence.analyze_run(): "
          f"{len(scans) - len(run_mismatch)}/{len(scans)} runs agree on both n_analyzed and n_skipped.")
    for s in run_mismatch:
        print(f"  RUN MISMATCH {s['dir'] if 'dir' in s else s['arm']+'/'+s['cell']}: "
              f"analysed {s['counts'].get('analysed',0)} vs {s['ref_n_analyzed']}, "
              f"skipped {s['total']-s['counts'].get('analysed',0)} vs {s['ref_n_skipped']}")
    if stats:
        print(f"Per-cell values compared: {stats['n_values']} across {stats['n_cells']} cells "
              f"(tolerance {TOL:g}).")
        print(f"Largest absolute difference: {stats['worst'][0]:.3e}"
              + (f" at {stats['worst'][1]}" if stats['worst'][1] else ""))
    if problems:
        print(f"\n*** {len(problems)} MISMATCH(ES) — Rule A does NOT reproduce the CSV. "
              f"Treat every number below as suspect. ***")
        for p in problems:
            print("  " + p)
    else:
        print("\nRESULT: Rule A reproduces divergence_analysis.csv exactly "
              "(no cell differs by more than the tolerance).")
    if none_b:
        print(f"\nNOTE: {none_b} gate-passing decision(s) had a None b_metrics; b_tie there was "
              f"computed over the non-None scores.")

    thr = sorted(thresholds)
    print(f"\nmax_state_age_ms observed in the logs: {thr}")
    thr_val = thr[0] if len(thr) == 1 else None

    # ---- Q1 summary ----
    for clean in (True, False):
        p = pooled(rows, clean)
        if not p:
            continue
        print("\n" + "#" * 100)
        print(f"# Q1 — DECISION OUTCOMES, clean={clean} (pooled across arms; "
              f"per (clean, arm, cell) detail in the CSV)")
        print("#" * 100)
        print(f"{'cell':<9}{'total':>8}{'analysed':>10}{'malform':>9}{'incompl':>9}"
              f"{'inel_only':>11}{'stale_only':>12}{'both':>8}{'undef':>8}"
              f"{'|any_inel':>11}{'any_stale':>11}{'skip%':>8}")
        for cell in [AD.col_label(w, c) for w, c in AD.CELL_COLS]:
            d = p.get(cell)
            if not d:
                continue
            t = d["total_decisions"]
            print(f"{cell:<9}{t:>8}{d['analysed']:>10}{d['skip_malformed_sv']:>9}"
                  f"{d['skip_incomplete_sv']:>9}{d['skip_ineligible_only']:>11}"
                  f"{d['skip_stale_only']:>12}{d['skip_both']:>8}"
                  f"{d['skip_undefined_argmin']:>8}{d['skip_any_ineligible']:>11}"
                  f"{d['skip_any_stale']:>11}"
                  f"{(d['skip_total']/t if t else 0):>8.3f}")
        print("  (any_inel and any_stale OVERLAP the exclusive columns; do not add them in.)")

    # ---- stale age ----
    print("\n" + "#" * 100)
    print(f"# Q1 — age_ms OF NODES FLAGGED STALE, vs max_state_age_ms"
          + (f" = {thr_val}" if thr_val is not None else ""))
    print("#" * 100)
    print(f"{'block':<8}{'node':<11}{'stale flags':>12}{'median':>11}{'p90':>11}"
          f"{'max':>13}{'<=thresh':>10}{'age missing':>13}")
    for clean in (True, False):
        for n in AD.NODES:
            tag = n.replace("pi-node-", "n")
            ages = [a for s in scans if s["clean"] is clean for a in s["stale_age"][n]]
            miss = sum(s["stale_age_missing"][n] for s in scans if s["clean"] is clean)
            le = sum(1 for a in ages if thr_val is not None and a <= thr_val)
            print(f"{str(clean):<8}{n:<11}{len(ages):>12}{fmt(AD.median(ages), 1):>11}"
                  f"{fmt(AD.pct(ages, 90), 1):>11}{fmt(max(ages) if ages else None, 1):>13}"
                  f"{le:>10}{miss:>13}")

    # ---- Q2 summary ----
    for clean in (True, False):
        block = [r for r in rows if r["clean"] is clean and r["total_decisions"]]
        if not block:
            continue
        print("\n" + "#" * 100)
        print(f"# Q2 — RATES UNDER EACH INCLUSION RULE, clean={clean} "
              f"(cell means of run-level rates)")
        print("#" * 100)
        print(f"{'arm':<27}{'cell':<9}"
              f"{'A n':>7}{'A div':>8}{'A tie':>8}"
              f"{'B n':>7}{'B div':>8}{'B tie':>8}"
              f"{'C n':>7}{'C div':>8}{'C tie':>8}")
        for r in block:
            print(f"{r['arm']:<27}{r['cell']:<9}"
                  f"{r['A_strict_n_analysed']:>7}{fmt(r['A_strict_div_router_mean']):>8}"
                  f"{fmt(r['A_strict_tie_router_mean']):>8}"
                  f"{r['B_eligible_only_n_analysed']:>7}{fmt(r['B_eligible_only_div_router_mean']):>8}"
                  f"{fmt(r['B_eligible_only_tie_router_mean']):>8}"
                  f"{r['C_eligible_subset_n_analysed']:>7}{fmt(r['C_eligible_subset_div_router_mean']):>8}"
                  f"{fmt(r['C_eligible_subset_tie_router_mean']):>8}")
        print("  div/tie shown are the ROUTER variant; uncensored variants and all "
              "standard deviations are in the CSV.")

    fieldnames = list(rows[0].keys())
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV written: {args.out}  ({len(rows)} rows, {len(fieldnames)} columns)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
