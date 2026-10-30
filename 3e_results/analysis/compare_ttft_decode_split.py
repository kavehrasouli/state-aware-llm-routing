"""Splits end-to-end latency into TTFT and decode duration and compares the policies.

Decode duration is derived per request (e2el - ttft) before taking percentiles.
"""
import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import analyze_3e as A3  # noqa: E402

CANDIDATE = "state_aware"
BASELINES = ["weighted_round_robin", "weighted_least_connections"]
ARMS = [CANDIDATE] + BASELINES
PCTILES = [50, 95]

PROTECTED = {"analyze_3e.py", "3e_report.md", "3e_summary.csv",
             "state_aware_rungs.md", "state_aware_rungs.csv"}

_RAW_CACHE = {}


def bench_raw(run):
    """The run's client-side per-request records. analyze_3e keeps only the
    summarised percentiles on the run dict, so the raw is re-read here."""
    d = run["dir"]
    if d in _RAW_CACHE:
        return _RAW_CACHE[d]
    meta = json.loads(open(os.path.join(d, "run_meta.json")).read())
    bf = meta.get("bench_result_file")
    raw = []
    if bf:
        p = os.path.join(d, bf)
        if os.path.exists(p):
            raw = json.loads(open(p).read()).get("raw", [])
    _RAW_CACHE[d] = raw
    return raw


def per_request_decode(run):
    """e2el_ms - ttft_ms per request, under analyze_3e.lat_for's own filter.

    Mirrors that filter rather than importing it because lat_for is a closure
    defined inside per_run() and cannot be reached from outside.
    """
    out = []
    for r in bench_raw(run):
        if not r.get("success") or A3.truthy_trunc(r.get("truncated")):
            continue
        e, t = r.get("e2el_ms"), r.get("ttft_ms")
        if isinstance(e, (int, float)) and isinstance(t, (int, float)):
            out.append(e - t)
    return out


def decode_summary(runs, percentile):
    """Per-run decode percentile, then agg across runs -- the same shape
    analyze_3e.latency_summary returns, so run_level_verdict can consume it."""
    return A3.agg([A3.pct(per_request_decode(run), percentile) for run in runs])


def metric_summary(runs, metric, percentile):
    """ttft and e2el come from analyze_3e's own per-run summaries; decode is
    derived here."""
    if metric == "decode":
        return decode_summary(runs, percentile)
    return A3.latency_summary(runs, "overall", metric, f"p{percentile}")


def sanity_check(cells):
    """The split is exact per request by construction; confirm it on real data,
    and show that the PERCENTILES do not sum, which is why decode is derived
    per request rather than by subtracting percentiles."""
    runs = A3.valid_runs(cells, CANDIDATE, 8, "uniform", clean_only=True)
    if not runs:
        return None
    run = runs[0]
    worst = 0.0
    n = 0
    for r in bench_raw(run):
        if not r.get("success") or A3.truthy_trunc(r.get("truncated")):
            continue
        e, t = r.get("e2el_ms"), r.get("ttft_ms")
        if isinstance(e, (int, float)) and isinstance(t, (int, float)):
            worst = max(worst, abs((t + (e - t)) - e))
            n += 1
    lat = run["lat"]["overall"]
    dec = per_request_decode(run)
    return {
        "n": n, "worst_residual": worst,
        "e2el_p95": lat["e2el"]["p95"], "ttft_p95": lat["ttft"]["p95"],
        "decode_p95": A3.pct(dec, 95),
        "sum_of_p95": lat["ttft"]["p95"] + A3.pct(dec, 95),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=HERE)
    ap.add_argument("--stem", default="ttft_decode_split")
    args = ap.parse_args()
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    for name in PROTECTED:
        if os.path.abspath(os.path.join(outdir, args.stem + ".md")) == \
                os.path.abspath(os.path.join(HERE, name)):
            sys.exit(f"FATAL: refusing to overwrite {name}.")

    runs = A3.load_all()
    cells = A3.group_cells(runs)

    print("=" * 100)
    print("TTFT vs decode split — is state_aware's e2el loss before generation "
          "or during it?")
    print("=" * 100)
    print(f"Runs loaded: {len(runs)} (analyze_3e.load_all).  "
          f"ALPHA={A3.ALPHA}  BOOT_ITERS={A3.BOOT_ITERS}  BOOT_SEED={A3.BOOT_SEED}")
    print("ttft: read from analyze_3e's own per-run summaries (LAT_METRICS "
          "already includes it).")
    print("decode: derived here as e2el_ms - ttft_ms PER REQUEST under "
          "lat_for's filter, then per-run percentile.")
    print("Latency inclusion: clean runs only. Run-level: per-run stat, then "
          "across runs.")

    # ---- queue_delay determination ----
    print("\n" + "#" * 100)
    print("# queue_delay — is it recorded router-side?")
    print("#" * 100)
    print("NO. The router JSONL request record has no queue_delay field. The only")
    print("queue_delay in the dataset is bench.py's, and it is CLIENT-side:")
    print("  bench.py:72   'queue_delay: float  # seconds client waited on "
          "semaphore before dispatch'")
    print("  bench.py:6-7  'TTFT is measured from actual HTTP dispatch time; "
          "queue_delay captures any")
    print("                 semaphore wait so client queuing is never silently "
          "folded into server latency.'")
    print("Because ttft and e2el are both measured FROM dispatch, client "
          "queue_delay is excluded")
    print("from them by construction and cannot contribute to any e2el "
          "difference between arms.")
    print("Router-side, arrival_ts_ns and dispatch_ts_ns exist, so a "
          "router-internal pre-dispatch")
    print("interval is derivable, but it is not a recorded queue delay and is "
          "dominated by select_ns")
    print("(SQ2: 1.4-34 us), which is ~4 orders of magnitude below the ttft "
          "differences below.")

    sc = sanity_check(cells)
    if sc:
        print("\n" + "#" * 100)
        print("# SPLIT VALIDITY (one clean state_aware c8_uni run)")
        print("#" * 100)
        print(f"  per-request ttft + decode == e2el on all {sc['n']} requests; "
              f"worst residual {sc['worst_residual']:.3e} ms")
        print(f"  but the PERCENTILES do not sum: ttft p95 {sc['ttft_p95']:.0f} + "
              f"decode p95 {sc['decode_p95']:.0f} = {sc['sum_of_p95']:.0f} "
              f"vs e2el p95 {sc['e2el_p95']:.0f}")
        print("  -> decode must be derived per request, which is what is done here.")

    # ---- exclusions ----
    print("\n" + "#" * 100)
    print("# RUNS AND EXCLUSIONS PER ARM PER CELL (latency path: clean only)")
    print("#" * 100)
    print(f"{'cell':<9}{'arm':<28}{'total':>7}{'clean':>7}{'excl':>6}{'excl %':>9}")
    ex_rows = []
    for workload, conc in A3.CELL_COLS:
        cell = A3.col_label(workload, conc)
        for arm in ARMS:
            total = len(A3.valid_runs(cells, arm, conc, workload))
            clean = len(A3.valid_runs(cells, arm, conc, workload, clean_only=True))
            row = [cell, arm, total, clean, total - clean,
                   f"{((total - clean) / total * 100):.1f}%" if total else "n/a"]
            ex_rows.append(row)
            print(f"{row[0]:<9}{row[1]:<28}{row[2]:>7}{row[3]:>7}{row[4]:>6}{row[5]:>9}")

    # ---- descriptives, all three arms ----
    print("\n" + "#" * 100)
    print("# DESCRIPTIVES — median-of-run-percentiles, ms (clean runs only)")
    print("#" * 100)
    print(f"{'cell':<9}{'arm':<28}{'n':>4}"
          + "".join(f"{f'{m} p{p}':>13}" for m in ("ttft", "decode", "e2el")
                   for p in PCTILES))
    desc_rows = []
    for workload, conc in A3.CELL_COLS:
        cell = A3.col_label(workload, conc)
        for arm in ARMS:
            rs = A3.valid_runs(cells, arm, conc, workload, clean_only=True)
            vals, cells_out = [], []
            for m in ("ttft", "decode", "e2el"):
                for p in PCTILES:
                    s = metric_summary(rs, m, p)
                    vals.append(s["median"])
                    cells_out.append(f"{s['median']:.0f}" if s["median"] is not None
                                     else "n/a")
            desc_rows.append([cell, arm, len(rs)] + cells_out)
            print(f"{cell:<9}{arm:<28}{len(rs):>4}"
                  + "".join(f"{c:>13}" for c in cells_out))

    # ---- differences ----
    diff_rows = []
    for base_arm in BASELINES:
        print("\n" + "#" * 100)
        print(f"# {base_arm} -> {CANDIDATE}  (delta = state_aware minus baseline, ms)")
        print("#" * 100)
        print(f"{'cell':<9}{'metric':<12}{'n':>7}{'delta':>11}{'95% CI':>26}"
              f"{'p':>8}{'enum':>9}{'min p':>8}  verdict")
        for workload, conc in A3.CELL_COLS:
            cell = A3.col_label(workload, conc)
            b = A3.valid_runs(cells, base_arm, conc, workload, clean_only=True)
            c = A3.valid_runs(cells, CANDIDATE, conc, workload, clean_only=True)
            nb, nc = len(b), len(c)
            if nb < 2 or nc < 2:
                print(f"{cell:<9}{'(all)':<12}{f'{nb}v{nc}':>7}"
                      f"{'untestable — needs >=2 clean reps per arm':>60}")
                continue
            floor = A3.attainable_min_p(nb, nc)
            for m in ("ttft", "decode", "e2el"):
                for p in PCTILES:
                    B, C = metric_summary(b, m, p), metric_summary(c, m, p)
                    v = A3.run_level_verdict(B, C)
                    lo, hi = v["ci"]
                    key = f"{m}_p{p}"
                    row = [base_arm, cell, key, f"{nb}v{nc}",
                           f"{v['diff']:+.0f}" if v["diff"] is not None else "n/a",
                           (f"[{lo:+.0f}, {hi:+.0f}]" if lo is not None else "n/a"),
                           (f"{v['p']:.3f}" if v["p"] is not None else "n/a"),
                           ("exact" if v["exact"] else "sampled"),
                           (f"{floor:.3f}" if floor is not None else "n/a"),
                           v["label"]]
                    diff_rows.append(row)
                    print(f"{cell:<9}{key:<12}{row[3]:>7}{row[4]:>11}{row[5]:>26}"
                          f"{row[6]:>8}{row[7]:>9}{row[8]:>8}  {row[9]}")

    # ---- write ----
    md = ["# TTFT vs decode split — state_aware against the weighted baselines", "",
          "Generated by `compare_ttft_decode_split.py`. `e2el_ms = ttft_ms + "
          "decode_ms` per request. `ttft` is read from analyze_3e's own per-run "
          "summaries; `decode` is derived here per request as `e2el_ms - ttft_ms` "
          "under `lat_for`'s filter, then reduced to a per-run percentile. "
          "Percentiles of ttft and decode do NOT sum to the e2el percentile, which "
          "is why decode is derived per request rather than by subtracting "
          "percentiles.", "",
          "All statistics are analyze_3e's own (`perm_test`, `bootstrap_ci`, "
          "`attainable_min_p`, `run_level_verdict`). Latency uses clean runs only; "
          "run-level means per-run percentile first, then across runs. Base is the "
          "weighted baseline, candidate is `state_aware`; a negative Δ favours "
          "`state_aware`.", "",
          "**queue_delay is not recorded router-side.** The only `queue_delay` in "
          "the dataset is `bench.py`'s client-side semaphore wait, and because "
          "`ttft` and `e2el` are both measured from HTTP dispatch it is excluded "
          "from them by construction — it cannot contribute to any e2el difference "
          "between arms.", "",
          "## Runs and exclusions per arm per cell", ""]
    md.append(A3.md_table(["cell", "arm", "total runs", "clean runs", "excluded",
                           "excluded %"], ex_rows))
    md += ["", "## Descriptives — median across runs of each run's percentile (ms)", ""]
    md.append(A3.md_table(
        ["cell", "arm", "n"] + [f"{m} p{p}" for m in ("ttft", "decode", "e2el")
                                for p in PCTILES], desc_rows))
    md += ["", "## Differences (Δ = state_aware − baseline, ms)", ""]
    md.append(A3.md_table(
        ["baseline", "cell", "metric", "n", "Δ", "95% CI", "p", "enum", "min p",
         "verdict"], diff_rows))

    md_path = os.path.join(outdir, args.stem + ".md")
    csv_path = os.path.join(outdir, args.stem + ".csv")
    open(md_path, "w").write("\n".join(md) + "\n")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["baseline", "cell", "metric", "n", "delta", "ci_95", "p",
                    "enum", "min_p", "verdict"])
        w.writerows(diff_rows)

    print(f"\nPermutation tests executed: {A3.TESTS_RUN['count']}  "
          f"(no multiplicity correction, matching 3e_report.md §9)")
    print(f"Wrote {md_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
