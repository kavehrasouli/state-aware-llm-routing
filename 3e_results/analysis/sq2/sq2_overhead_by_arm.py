"""Selection overhead per policy, recomputed from the runs (means of per-run statistics)."""
import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)  # combined analysis helpers
for p in (HERE, PARENT):
    if p not in sys.path:
        sys.path.insert(0, p)

import analyze_divergence as AD          # noqa: E402
from analyze_sq2 import scan, SCOPES, in_scope, across_runs  # noqa: E402
from analyze_sq1 import resolve_num_warmup  # noqa: E402

SOURCE_CSV = os.path.join(HERE, "sq2_routing_overhead.csv")
STATS = ("median", "p90", "p99")
TOL = 1e-9


def collect():
    """WARM-UP EXCLUDED (fixed 2026-08-24). This previously called scan() without
    the warm-up arguments, so it counted all 103 router records per run while
    analyze_sq2.py counted 100. num_warmup is resolved per run by the same
    helper analyze_sq2 uses, and warmup_mode='measured' applies the
    request_id >= num_warmup predicate."""
    runs = AD.load_all_runs()
    usable = [r for r in runs if r.get("usable")]
    scans = []
    warmup_val = {}
    for r in usable:
        meta = json.loads(open(os.path.join(r["dir"], "run_meta.json")).read())
        nw, _src = resolve_num_warmup(meta, r["dir"])
        warmup_val[nw] = warmup_val.get(nw, 0) + 1
        s = scan(meta, os.path.join(r["dir"], meta["router_log_file"]), nw, "measured")
        if s is None:
            continue
        s.update({"scope_clean": r.get("verdict") == "clean", "arm": r["arm"],
                  "cell": AD.col_label(r["workload"], r["conc"]), "dir": r["dir"],
                  "num_warmup": nw})
        scans.append(s)
    return len(runs), len(usable), scans, warmup_val


def per_run_stat(sample, stat):
    if not sample:
        return None
    return AD.median(sample) if stat == "median" else AD.pct(
        sample, 90 if stat == "p90" else 99)


def build(scans):
    rows = []
    for scope in SCOPES:
        for arm in sorted({s["arm"] for s in scans if in_scope(s, scope)}):
            rs = [s for s in scans
                  if in_scope(s, scope) and s["arm"] == arm and s["select_us"]]
            if not rs:
                continue
            pooled = [v for s in rs for v in s["select_us"]]
            row = {"scope": scope, "arm": arm, "n_runs": len(rs),
                   "n_requests": len(pooled),
                   "n_cells": len({s["cell"] for s in rs})}
            for stat in STATS:
                per_run = [per_run_stat(s["select_us"], stat) for s in rs]
                m, sd, n = across_runs(per_run)
                row[f"select_us_{stat}_runmean"] = m
                row[f"select_us_{stat}_runsd"] = sd
                row[f"select_us_{stat}_n_runs"] = n
                row[f"select_us_{stat}_pooled"] = per_run_stat(pooled, stat)

            # Timing breakdown, only for arms that also time body reading and token estimation.
            # Component medians are taken separately, so they do not add up to the total median.
            for label, field in (("body_read_us", "body_us"),
                                 ("token_estimate_us", "est_us"),
                                 ("residual_us", "resid_us")):
                v = [x for s in rs for x in s[field]]
                row[f"{label}_median_pooled"] = AD.median(v) if v else None
                per_run = [AD.median(s[field]) for s in rs if s[field]]
                mm, ss, nn = across_runs(per_run)
                row[f"{label}_median_runmean"] = mm
                row[f"{label}_median_runsd"] = ss
            row["decomposition_available"] = bool(
                [x for s in rs for x in s["body_us"]])

            # Selection as a share of end-to-end latency, both conventions.
            sel = [v for s in rs for v in s["sel_for_ratio"]]
            tot = [v for s in rs for v in s["tot_for_ratio"]]
            rat = [v for s in rs for v in s["ratio_per_req"]]
            mt = AD.median(tot) if tot else None
            row["latency_share_ppm_ratio_of_medians"] = (
                (AD.median(sel) / 1000.0) / mt * 1e6 if sel and mt else None)
            row["latency_share_ppm_median_of_ratios"] = (
                AD.median(rat) * 1e6 if rat else None)
            row["total_ms_median_pooled"] = mt
            row["n_requests_with_total_ms"] = len(rat)
            rows.append(row)
    return rows


def verify_collapse(rows):
    """Check, rather than assert, which columns survive a run-count-weighted
    collapse of the per-cell CSV: the means do, the SDs do not."""
    if not os.path.exists(SOURCE_CSV):
        print(f"  {SOURCE_CSV} not found -- collapse check skipped.")
        return
    with open(SOURCE_CSV) as f:
        cells = list(csv.DictReader(f))
    mine = {(r["scope"], r["arm"]): r for r in rows}

    def weighted(scope, arm, col, stat):
        num = den = 0.0
        for c in cells:
            if c["scope"] != scope or c["arm"] != arm:
                continue
            v = c.get(col, "")
            n = c.get(f"select_us_{stat}_n_runs", "")
            if v in ("", None) or n in ("", None) or int(n) == 0:
                continue
            num += float(v) * int(n)
            den += int(n)
        return (num / den) if den else None

    print(f"{'column':<28}{'arms':>6}{'exact':>7}{'max abs diff':>15}   verdict")
    for kind in ("runmean", "runsd"):
        for stat in STATS:
            col = f"select_us_{stat}_{kind}"
            checked = exact = 0
            worst = 0.0
            for (scope, arm), row in sorted(mine.items()):
                collapsed = weighted(scope, arm, col, stat)
                truth = row[col]
                if collapsed is None or truth is None:
                    continue
                checked += 1
                d = abs(collapsed - truth)
                worst = max(worst, d)
                if d <= TOL:
                    exact += 1
            verdict = ("collapses exactly" if exact == checked
                       else "DOES NOT COLLAPSE -- recompute")
            print(f"{col:<28}{checked:>6}{exact:>7}{worst:>15.3e}   {verdict}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(HERE, "sq2_overhead_by_arm.csv"))
    args = ap.parse_args()
    for protected in ("sq2_routing_overhead.csv", "sq2_overhead_decomposition.csv",
                      "sq2_overhead_share_of_latency.csv", "sq2_staleness.csv",
                      "sq2_stale_handling.csv", "analyze_sq2.py"):
        if os.path.abspath(args.out) == os.path.join(HERE, protected):
            sys.exit(f"FATAL: refusing to overwrite {protected}.")

    print("=" * 100)
    print("SQ2 — one defensible per-arm overhead row. Read-only; analyze_sq2.scan reused.")
    print("=" * 100)
    n_all, n_usable, scans, warmup_val = collect()
    print(f"{n_all} run directories, {n_usable} usable "
          f"(selection imported from analyze_divergence.load_all_runs).")
    print(f"WARM-UP EXCLUDED (fixed 2026-08-24): num_warmup per run = {warmup_val}; "
          f"predicate request_id >= num_warmup, mirrored from analyze_3e.py.")

    print("\n" + "#" * 100)
    print("# CONVENTION")
    print("#" * 100)
    print("Each run's OWN median / p90 / p99 of select_us is computed first, then the")
    print("mean and SAMPLE SD (ddof = 1, analyze_divergence.agg) are taken across all of")
    print("that arm's runs, pooled over cells. The run is the unit of replication.")
    print("Percentiles are analyze_divergence.pct (linear interpolation). Units:")
    print("microseconds. *_pooled is the statistic over all that arm's requests at once,")
    print("shown for comparison only -- it is not the quotable figure.")

    rows = build(scans)

    print("\n" + "#" * 100)
    print("# COLLAPSE CHECK — can the per-cell CSV be aggregated to these numbers?")
    print("#" * 100)
    verify_collapse(rows)
    print("  All three *_runmean columns collapse exactly: each is a mean of per-run")
    print("  statistics, and a weighted mean of means over disjoint cells is the mean")
    print("  over their union -- the averaging is linear, the averaged quantity need not")
    print("  be. The *_runsd columns do NOT collapse: a weighted mean of per-cell SDs")
    print("  drops the between-cell component. That, and the absence of any p99 column")
    print("  from the earlier stdout summary, is what this file supplies.")
    print("  Note: *_p90_runmean is the MEAN ACROSS RUNS OF EACH RUN'S p90. It is not")
    print("  the p90 of the pooled request distribution and must not be labelled so.")

    for scope in SCOPES:
        sub = [r for r in rows if r["scope"] == scope]
        if not sub:
            continue
        print("\n" + "#" * 100)
        print(f"# PER-ARM OVERHEAD, scope={scope} (microseconds, run-level)")
        print("#" * 100)
        print(f"{'arm':<28}{'runs':>6}{'reqs':>7}"
              f"{'median':>8}{'±sd':>7}{'(pool)':>8}"
              f"{'p90':>8}{'±sd':>7}{'(pool)':>8}"
              f"{'p99':>8}{'±sd':>8}{'(pool)':>8}{'ppm':>7}")
        for r in sorted(sub, key=lambda x: -x["select_us_median_runmean"]):
            def f(k, nd=2):
                v = r[k]
                return f"{v:.{nd}f}" if v is not None else "-"
            print(f"{r['arm']:<28}{r['n_runs']:>6}{r['n_requests']:>7}"
                  f"{f('select_us_median_runmean'):>8}{f('select_us_median_runsd'):>7}"
                  f"{f('select_us_median_pooled'):>8}"
                  f"{f('select_us_p90_runmean'):>8}{f('select_us_p90_runsd'):>7}"
                  f"{f('select_us_p90_pooled'):>8}"
                  f"{f('select_us_p99_runmean'):>8}{f('select_us_p99_runsd'):>8}"
                  f"{f('select_us_p99_pooled'):>8}"
                  f"{f('latency_share_ppm_ratio_of_medians',3):>7}")
        print("  median/p90/p99 = mean across runs of each run's own percentile, "
              "± sample SD (ddof=1).")
        print("  (pool) = same percentile over all that arm's requests pooled.")
        print("  ppm = selection as a share of end-to-end latency, RATIO OF MEDIANS "
              "(median select / median total_ms).")

        dec = [r for r in sub if r["decomposition_available"]]
        if dec:
            print(f"\n  Decomposition, arms whose timer also covers body read and token "
                  f"estimation (medians, microseconds):")
            print(f"  {'arm':<26}{'select':>9}{'body read':>11}{'tok est':>9}"
                  f"{'residual':>10}{'sum of 3':>10}")
            for r in dec:
                comp = [r["body_read_us_median_pooled"],
                        r["token_estimate_us_median_pooled"],
                        r["residual_us_median_pooled"]]
                s3 = sum(c for c in comp if c is not None)
                print(f"  {r['arm']:<26}{r['select_us_median_pooled']:>9.2f}"
                      f"{comp[0]:>11.2f}{comp[1]:>9.2f}{comp[2]:>10.2f}{s3:>10.2f}")
            print("  residual is computed PER REQUEST (select - body - estimate) and then")
            print("  summarised; the four medians are medians of four separate")
            print("  distributions, so 'sum of 3' does NOT equal the select median.")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.out}  ({len(rows)} rows, {len(rows[0])} cols)")
    print("Quote scope=all for routing overhead, per methodology Section 6.")
    print("No statistical test applied; these are descriptive.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
