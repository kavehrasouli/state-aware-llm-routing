"""Compares state-aware routing with WRR and WLC per cell, using analyze_3e's statistics.

A negative difference favors state-aware routing.
"""
import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import analyze_3e as A3  # noqa: E402

CANDIDATE = "state_aware"
BASELINES = [
    ("weighted_round_robin",
     "capacity-weighted cyclic, no live state -- absent from 3e_report.md"),
    ("weighted_least_connections",
     "capacity-weighted live in-flight count -- §2's baseline, reported there "
     "without p/CI in five of seven cells"),
]

PROTECTED = {"3e_report.md", "3e_summary.csv", "analyze_3e.py"}


def fmt(v, nd=0, signed=True):
    if v is None:
        return "n/a"
    return f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"


def series(vals, nd=0):
    """Per-run values, in run order, for the columns §8 never prints."""
    if not vals:
        return "n/a"
    return ", ".join(f"{v:.{nd}f}" for v in vals)


def latency_block(cells, base_arm, cand_arm):
    """Mirrored from analyze_3e.decision_section_ladder's latency table: same
    columns, same order, same helpers. Extended with the per-run series, which
    that function does not emit."""
    rows, per_run, untestable = [], [], []
    for workload, conc in A3.CELL_COLS:
        cell = A3.col_label(workload, conc)
        b_runs = A3.valid_runs(cells, base_arm, conc, workload, clean_only=True)
        c_runs = A3.valid_runs(cells, cand_arm, conc, workload, clean_only=True)
        nb, nc = len(b_runs), len(c_runs)
        if nb < 2 or nc < 2:
            why = []
            if nb < 2:
                why.append(f"{base_arm} has {nb} clean rep{'s' if nb != 1 else ''}")
            if nc < 2:
                why.append(f"{cand_arm} has {nc} clean rep{'s' if nc != 1 else ''}")
            untestable.append(f"- `{cell}`: **untestable** — " + " and ".join(why)
                              + " (a permutation test needs >=2 per arm)")
            continue
        floor = A3.attainable_min_p(nb, nc)
        for metric, percentile in A3.DECISION_METRICS:
            key = f"{metric}_{percentile}"
            B = A3.latency_summary(b_runs, "overall", metric, percentile)
            C = A3.latency_summary(c_runs, "overall", metric, percentile)
            v = A3.run_level_verdict(B, C)
            lo, hi = v["ci"]
            nd = 1 if metric == "tpot" else 0
            rows.append([
                cell, key, f"{nb}v{nc}",
                fmt(v["diff"], nd),
                (f"[{lo:+.{nd}f}, {hi:+.{nd}f}]" if lo is not None else "n/a"),
                (f"{v['p']:.3f}" if v["p"] is not None else "n/a"),
                ("exact" if v["exact"] else "sampled"),
                (f"{floor:.3f}" if floor is not None else "n/a"),
                v["label"],
            ])
            per_run.append([
                cell, key,
                f"{B['median']:.{nd}f}" if B["median"] is not None else "n/a",
                series(B["values"], nd),
                f"{C['median']:.{nd}f}" if C["median"] is not None else "n/a",
                series(C["values"], nd),
            ])
    return rows, per_run, untestable


def completion_block(cells, base_arm, cand_arm):
    """Mirrored from the same function's completion table. ALL runs."""
    rows, per_run = [], []
    for workload, conc in A3.CELL_COLS:
        cell = A3.col_label(workload, conc)
        b_runs = A3.valid_runs(cells, base_arm, conc, workload)
        c_runs = A3.valid_runs(cells, cand_arm, conc, workload)
        nb, nc = len(b_runs), len(c_runs)
        B = A3.scaled(A3.completion_summary(b_runs), 100.0)
        C = A3.scaled(A3.completion_summary(c_runs), 100.0)
        if nb < 2 or nc < 2:
            rows.append([cell, f"{nb}v{nc}", "n/a", "n/a", "n/a", "n/a", "n/a",
                         "n/a", "untestable (<2 runs)"])
            continue
        v = A3.run_level_verdict(B, C, lower_is_better=False)
        lo, hi = v["ci"]
        floor = A3.attainable_min_p(nb, nc)
        rows.append([
            cell, f"{nb}v{nc}",
            f"{B['median']:.1f}%", f"{C['median']:.1f}%",
            fmt(v["diff"], 1),
            (f"[{lo:+.1f}, {hi:+.1f}]" if lo is not None else "n/a"),
            (f"{v['p']:.3f}" if v["p"] is not None else "n/a"),
            (f"{floor:.3f}" if floor is not None else "n/a"),
            v["label"],
        ])
        per_run.append([cell, f"{B['median']:.1f}", series(B["values"], 1),
                        f"{C['median']:.1f}", series(C["values"], 1)])
    return rows, per_run


def exclusion_block(cells, arms):
    """Per ARM per CELL exclusions. Appendix A reports only repo-wide per-arm
    totals, so this is computed here from the same valid_runs filter: total is
    every run with class_counts, clean is the subset the latency path keeps."""
    rows = []
    for workload, conc in A3.CELL_COLS:
        cell = A3.col_label(workload, conc)
        for arm in arms:
            total = len(A3.valid_runs(cells, arm, conc, workload))
            clean = len(A3.valid_runs(cells, arm, conc, workload, clean_only=True))
            dropped = total - clean
            rows.append([cell, arm, total, clean, dropped,
                         f"{(dropped / total * 100):.1f}%" if total else "n/a"])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=HERE)
    ap.add_argument("--stem", default="state_aware_rungs")
    args = ap.parse_args()
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    for name in PROTECTED:
        if os.path.abspath(os.path.join(outdir, args.stem + ".md")) == \
                os.path.abspath(os.path.join(HERE, name)):
            sys.exit(f"FATAL: refusing to overwrite {name}.")

    print("=" * 96)
    print("state_aware vs the two weighted baselines — §8 ladder format")
    print("Machinery imported from analyze_3e: perm_test, bootstrap_ci,")
    print("attainable_min_p, run_level_verdict, valid_runs, latency_summary,")
    print("completion_summary. Nothing existing is modified.")
    print("=" * 96)

    runs = A3.load_all()
    cells = A3.group_cells(runs)
    print(f"Runs loaded: {len(runs)}  (analyze_3e.load_all)")
    print(f"ALPHA={A3.ALPHA}  BOOT_ITERS={A3.BOOT_ITERS}  BOOT_SEED={A3.BOOT_SEED}  "
          f"PERM_EXACT_LIMIT={A3.PERM_EXACT_LIMIT}")
    print(f"Metrics: {[f'{m}_{p}' for m, p in A3.DECISION_METRICS]} + completion rate")
    print("Direction: base = weighted baseline, candidate = state_aware; negative")
    print("latency delta favours state_aware, positive completion delta favours it.")
    print("Inclusion: latency = clean runs only; completion = ALL runs "
          "(analyze_3e.valid_runs).")

    md = ["# state_aware against the two weighted baselines",
          "",
          "Generated by `compare_state_aware_rungs.py`. Every statistic is produced "
          "by machinery imported from `analyze_3e.py` — the same permutation test, "
          "bootstrap interval, attainable-minimum-p calculation and verdict "
          "labelling that produced `3e_report.md`. Neither comparison is available "
          "in quotable form in that report: the weighted_round_robin pair is absent "
          "entirely, and the weighted_least_connections pair appears in §2 without "
          "p, CI, `enum` or `min p` in five of seven cells.",
          "",
          f"- `ALPHA` = {A3.ALPHA}, `BOOT_ITERS` = {A3.BOOT_ITERS}, "
          f"`BOOT_SEED` = {A3.BOOT_SEED}, `PERM_EXACT_LIMIT` = {A3.PERM_EXACT_LIMIT}",
          "- Base is the weighted baseline, candidate is `state_aware`, matching §2. "
          "A negative latency Δ favours `state_aware`; a positive completion Δ "
          "favours it.",
          "- Latency uses clean runs only; completion uses ALL runs. Both filters "
          "are `analyze_3e.valid_runs`.",
          "- `min p` is the smallest p this pair of run counts can produce under "
          "perfect separation. Where it exceeds ALPHA the cell cannot return a "
          "significant result at any effect size, so a non-significant outcome "
          "there is a resolution limit, not evidence of equivalence.",
          "",
          "## Runs and exclusions per arm per cell",
          "",
          "`total` is every run with class counts (the completion denominator); "
          "`clean` is the subset the latency path keeps.",
          ""]
    arms = [CANDIDATE] + [b for b, _ in BASELINES]
    ex_rows = exclusion_block(cells, arms)
    md.append(A3.md_table(["cell", "arm", "total runs", "clean runs",
                           "excluded", "excluded %"], ex_rows))

    print("\n" + "#" * 96)
    print("# RUNS AND EXCLUSIONS PER ARM PER CELL")
    print("#" * 96)
    print(f"{'cell':<9}{'arm':<28}{'total':>7}{'clean':>7}{'excl':>6}{'excl %':>9}")
    for r in ex_rows:
        print(f"{r[0]:<9}{r[1]:<28}{r[2]:>7}{r[3]:>7}{r[4]:>6}{r[5]:>9}")

    csv_rows = []
    for base_arm, note in BASELINES:
        md += ["", f"## {base_arm} -> {CANDIDATE}", "", f"_{note}_", ""]
        lat_rows, lat_per_run, untestable = latency_block(cells, base_arm, CANDIDATE)
        comp_rows, comp_per_run = completion_block(cells, base_arm, CANDIDATE)

        print("\n" + "#" * 96)
        print(f"# {base_arm} -> {CANDIDATE}")
        print("#" * 96)
        print("\nLATENCY (clean runs only). delta = state_aware minus baseline.")
        print(f"{'cell':<9}{'metric':<10}{'n':>7}{'delta':>12}{'95% CI':>26}"
              f"{'p':>8}{'enum':>9}{'min p':>8}  verdict")
        for r in lat_rows:
            print(f"{r[0]:<9}{r[1]:<10}{r[2]:>7}{r[3]:>12}{r[4]:>26}"
                  f"{r[5]:>8}{r[6]:>9}{r[7]:>8}  {r[8]}")
        for u in untestable:
            print("  " + u.replace("- ", ""))

        if lat_rows:
            md += ["### Latency (clean runs only)", "",
                   "Δ is median-of-reps of `state_aware` minus the baseline, in ms; "
                   "negative favours `state_aware`.", ""]
            md.append(A3.md_table(
                ["cell", "metric", "n", "Δ median", "95% CI", "p", "enum",
                 "min p", "verdict"], lat_rows))
        if untestable:
            md += ["", "**Cells where latency cannot be tested:**", ""] + untestable

        md += ["", "### Per-run medians (clean runs)", "",
               "The per-run values behind each Δ. `3e_report.md` prints "
               "median-of-reps with `[min–max]` but never the series itself.", ""]
        md.append(A3.md_table(
            ["cell", "metric", "base median", "base per-run",
             "state_aware median", "state_aware per-run"], lat_per_run))

        print("\nCOMPLETION (ALL runs). delta in percentage points; positive favours "
              "state_aware.")
        print(f"{'cell':<9}{'n':>7}{'base':>8}{'cand':>8}{'delta pp':>10}"
              f"{'95% CI':>20}{'p':>8}{'min p':>8}  verdict")
        for r in comp_rows:
            print(f"{r[0]:<9}{r[1]:>7}{r[2]:>8}{r[3]:>8}{r[4]:>10}"
                  f"{r[5]:>20}{r[6]:>8}{r[7]:>8}  {r[8]}")

        md += ["", "### Completion rate (ALL runs — clean and suspect)", "",
               "Δ in percentage points; positive favours `state_aware`.", ""]
        md.append(A3.md_table(
            ["cell", "n", "base", "state_aware", "Δ pp", "95% CI", "p",
             "min p", "verdict"], comp_rows))
        md += ["", "### Per-run completion (ALL runs)", ""]
        md.append(A3.md_table(
            ["cell", "base median %", "base per-run %",
             "state_aware median %", "state_aware per-run %"], comp_per_run))

        for r in lat_rows:
            csv_rows.append({"baseline": base_arm, "candidate": CANDIDATE,
                             "family": "latency_clean_only", "cell": r[0],
                             "metric": r[1], "n": r[2], "delta": r[3],
                             "ci_95": r[4], "p": r[5], "enum": r[6],
                             "min_p": r[7], "verdict": r[8]})
        for r in comp_rows:
            csv_rows.append({"baseline": base_arm, "candidate": CANDIDATE,
                             "family": "completion_all_runs", "cell": r[0],
                             "metric": "completion_rate", "n": r[1],
                             "delta": r[4], "ci_95": r[5], "p": r[6],
                             "enum": "", "min_p": r[7], "verdict": r[8]})

    md_path = os.path.join(outdir, args.stem + ".md")
    csv_path = os.path.join(outdir, args.stem + ".csv")
    open(md_path, "w").write("\n".join(md) + "\n")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)

    print(f"\nPermutation tests executed by this script: {A3.TESTS_RUN['count']}")
    print("No correction for multiple comparisons is applied, matching "
          "3e_report.md §9.")
    print(f"Wrote {md_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
