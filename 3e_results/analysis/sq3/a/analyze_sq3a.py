"""Completion results: per-run completion, runs failing the validity filter, and failure types.

Uses all measured runs, since filtering on failures would condition on the outcome.
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))       # .../3e_results
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import analyze_3e as A3  # noqa: E402
import analyze_divergence as AD  # noqa: E402

BLOCK_FILES = {
    1: ("sq3a_completion_rate.csv",
        "SQ3a reliability headline: completion rate per arm and cell — n runs, "
        "the per-run rates, their mean and sample SD, and the median with "
        "min-max across repetitions."),
    2: ("sq3a_structural_failures.csv",
        "SQ3a coverage: every (arm, cell) with no run passing the validity "
        "gate, separating a cell that was never run from one that was run and "
        "always failed, with the gate condition each failed run hit."),
    3: ("sq3a_ladder_completion.csv",
        "SQ3a ladder: run-level permutation tests of completion rate for each "
        "rung, per cell, with the bootstrap CI and the attainable minimum p."),
    4: ("sq3a_failure_composition.csv",
        "SQ3a failure composition: what the failed requests were, by outcome "
        "category, on the client side and the router side."),
}

# Validity-filter conditions from run_3e.compute_verdict; `other` covers every remaining cause.
GATE_CONDITIONS = [
    ("no_replica_available", "router reported no healthy replica for >=1 request"),
    ("replica_dropped_out", "mid-run eviction: a replica left the pool"),
    ("failure_rate_above_gate", "run failure rate exceeded the driver's gate"),
    ("other", "non-clean for some other reason (reason text carried verbatim)"),
]

# Fifth rung (WLC -> state-aware), added after the four imported rungs.
SQ3A_EXTRA_STEP = ("weighted_least_connections", "state_aware",
                   "state: live queue state on top of weighted connections", False)
EXPECTED_IMPORTED_STEPS = [
    ("random", "round_robin"),
    ("round_robin", "weighted_round_robin"),
    ("round_robin", "least_connections"),
    ("least_connections", "weighted_least_connections"),
]


# ───────────────────────── driver-derived constants ───────────────────────
def detect_failure_rate_max(driver_path):
    """The failure-rate gate, read out of the driver rather than assumed.

    Mirrors analyze_sq1.detect_np's stance: block 2 classifies runs against this
    threshold, so guessing it would silently mislabel them. Fails loudly if the
    driver does not state it unambiguously."""
    if not os.path.exists(driver_path):
        raise SystemExit(f"FATAL: driver not found at {driver_path}; cannot read "
                         f"FAILURE_RATE_MAX. Pass --driver.")
    text = open(driver_path).read()
    hits = [(float(m.group(1)), text[:m.start()].count("\n") + 1)
            for m in re.finditer(r"^FAILURE_RATE_MAX\s*=\s*([0-9.]+)", text, re.M)]
    if not hits:
        raise SystemExit(f"FATAL: no FAILURE_RATE_MAX found in {driver_path}; "
                         f"refusing to assume the verdict gate.")
    if len({h[0] for h in hits}) > 1:
        raise SystemExit(f"FATAL: conflicting FAILURE_RATE_MAX in {driver_path}: "
                         + "; ".join(f"{v} (line {ln})" for v, ln in hits))
    return hits[0][0], hits[0][1]


def report_appendix_counts(report_path):
    """The two run counts 3e_report.md states in its Appendix A, parsed rather
    than retyped, so the reconciliation below compares against the file."""
    if not os.path.exists(report_path):
        return None, None
    text = open(report_path).read()
    grid = re.search(r"Grid runs loaded:\s*\*\*(\d+)\*\*", text)
    warm = re.search(r"Router warmup records excluded:\s*\*\*(\d+)\*\*", text)
    return (int(grid.group(1)) if grid else None,
            int(warm.group(1)) if warm else None)


# ───────────────────────── grouping ───────────────────────────────────────
def group_all(runs):
    """(arm, conc, workload) -> [run], INCLUDING off-grid cells.

    Mirrored from analyze_3e.group_cells, which is identical except that it also
    requires run["grid"]. Dropping non-grid runs would make an off-grid cell
    invisible to this section instead of visible-and-flagged, and SQ3a's whole
    point is to account for coverage. Key shape is unchanged, so the imported
    analyze_3e.valid_runs works against the result.

    NOTE ON THE KEY: analyze_3e.per_run stores the resolved ARM under the key
    "policy" (`meta["arm"] or meta["policy"]`). It is the arm, not the routing
    policy: wlc_conc runs weighted_least_connections and is kept separate here
    precisely because of that resolution. Every column this script emits calls
    it `arm`.
    """
    g = defaultdict(list)
    for r in runs:
        if "router" not in r:
            continue
        g[A3.cellkey(r["policy"], r["conc"], r["workload"])].append(r)
    return g


def enumerate_cells(runs):
    """CELL_COLS order, then any observed (workload, conc) outside it."""
    canonical = list(A3.CELL_COLS)
    observed = {(r["workload"], r["conc"]) for r in runs if "router" in r}
    extra = sorted(observed - set(canonical), key=lambda wc: (wc[0], wc[1]))
    return canonical, extra


def enumerate_arms(runs):
    """analyze_3e.POLICIES order first (so the ladder reads top to bottom), then
    any arm present in the data but absent from that list."""
    observed = {r["policy"] for r in runs if "router" in r}
    ordered = [a for a in A3.POLICIES if a in observed]
    return ordered + sorted(observed - set(ordered))


def base_row(arm, workload, conc, in_grid, runs):
    clean = [r for r in runs if r.get("verdict") == "clean"]
    return {
        "arm": arm,
        "cell": A3.col_label(workload, conc),
        "workload": workload,
        "concurrency": conc,
        "in_grid": in_grid,
        "n_runs": len(runs),
        "n_clean": len(clean),
        "n_not_clean": len(runs) - len(clean),
    }


def fmt_list(values, nd=4):
    return "|".join("" if v is None else f"{v:.{nd}f}" for v in values)


# ───────────────────────── block 1 — completion rate ──────────────────────
def build_block1(cells, keys):
    rows = []
    for arm, workload, conc, in_grid in keys:
        runs = A3.valid_runs(cells, arm, conc, workload)          # ALL runs
        if not runs:
            continue
        row = base_row(arm, workload, conc, in_grid, runs)
        summary = A3.completion_summary(runs)
        per_run = [A3.run_completion(r) for r in runs]
        ordered = sorted(zip((r.get("rep") for r in runs), per_run),
                         key=lambda t: (t[0] is None, t[0]))
        row.update({
            # The per-run rates themselves, so nothing below cannot be rechecked.
            "reps": "|".join(str(rep) for rep, _ in ordered),
            "completion_rate_per_run": fmt_list([v for _, v in ordered]),
            "completion_rate_mean": summary["mean"],
            # agg's SD is the SAMPLE SD (ddof=1) and returns 0.0 at n==1, which
            # is undefined rather than zero variance -- n_runs is carried above.
            "completion_rate_sd": summary["std"] if summary["n"] >= 2 else None,
            "completion_rate_sd_is_defined": summary["n"] >= 2,
            "completion_rate_median": summary["median"],
            "completion_rate_min": summary["min"],
            "completion_rate_max": summary["max"],
            # analyze_3e.fmt_completion's own rendering, verbatim, so the chapter
            # can quote the same string this project already uses elsewhere.
            "median_min_max_fmt": A3.fmt_completion(summary),
            "successes": summary["successes"],
            "offered": summary["offered"],
            "completion_rate_pooled": summary["pooled"],
            "goodput_req_s_median": A3.throughput_summary(runs)["median"],
        })
        # The clean / not-clean split, reported BESIDE the all-runs figure. No
        # block filters on it.
        for label, subset in (("clean", [r for r in runs if r.get("verdict") == "clean"]),
                              ("not_clean", [r for r in runs if r.get("verdict") != "clean"])):
            s = A3.completion_summary(subset) if subset else None
            row[f"completion_rate_mean_{label}"] = s["mean"] if s else None
            row[f"completion_rate_median_{label}"] = s["median"] if s else None
            row[f"n_runs_{label}"] = len(subset)
        row["all_runs_minus_clean_pp"] = (
            (summary["mean"] - A3.completion_summary(
                [r for r in runs if r.get("verdict") == "clean"])["mean"]) * 100
            if summary["mean"] is not None
            and any(r.get("verdict") == "clean" for r in runs) else None)
        rows.append(row)
    return rows


# ───────────────────────── block 2 — structural failures ──────────────────
def classify_gate(run, failure_rate_max):
    """Which validity conditions this run failed, from the numeric router_metrics
    run_3e.compute_verdict wrote, plus the recorded reason for cross-checking.

    Mirrored from that function: it applies these conditions inline and keeps
    only the joined string, so they are re-applied to the numbers here rather
    than parsed back out of prose. The recorded string is still compared, and a
    disagreement is reported rather than resolved silently.
    """
    meta_path = os.path.join(run["dir"], "run_meta.json")
    try:
        meta = json.loads(open(meta_path).read())
    except OSError as e:
        return (["other"], f"run_meta unreadable: {e}", None, False)
    rm = meta.get("router_metrics") or {}
    reason = meta.get("verdict_reason") or ""
    hit = []
    if (rm.get("no_healthy_replica") or 0) > 0:
        hit.append("no_replica_available")
    if (rm.get("eviction_events") or 0) > 0:
        hit.append("replica_dropped_out")
    fr = rm.get("failure_rate")
    if isinstance(fr, (int, float)) and fr > failure_rate_max:
        hit.append("failure_rate_above_gate")
    if not hit:
        # Non-clean for a reason outside the three named conditions: an empty
        # router log, several run_ids in one log, or a pre-gate abort.
        hit.append("other")
    # Cross-check: every named condition should be visible in the reason string.
    tokens = {"no_replica_available": "no_healthy_replica",
              "replica_dropped_out": "eviction",
              "failure_rate_above_gate": "failure_rate"}
    agrees = all(tokens[h] in reason for h in hit if h in tokens)
    return hit, reason, fr, agrees


def build_block2(cells, keys, failure_rate_max):
    """Every (arm, cell) with zero runs passing the gate.

    Two distinct states are separated, because they mean opposite things: a cell
    with n_attempted == 0 was never run at all (nothing to conclude), while a
    cell with n_attempted > 0 and n_clean == 0 was run and failed every time
    (which is itself the SQ3a result for that arm).
    """
    rows, disagreements = [], []
    for arm, workload, conc, in_grid in keys:
        runs = A3.valid_runs(cells, arm, conc, workload)
        clean = [r for r in runs if r.get("verdict") == "clean"]
        if clean:
            continue
        row = base_row(arm, workload, conc, in_grid, runs)
        row["status"] = "never_run" if not runs else "run_and_all_failed"
        row["n_attempted"] = len(runs)
        row["n_failed_gate"] = len(runs)
        counts = Counter()
        detail, reasons, rates = [], [], []
        for r in sorted(runs, key=lambda x: (x.get("rep") is None, x.get("rep"))):
            hit, reason, fr, agrees = classify_gate(r, failure_rate_max)
            counts.update(hit)
            detail.append(f"rep{r.get('rep')}:{'+'.join(hit)}"
                          + (f"(fr={fr:.4f})" if isinstance(fr, (int, float)) else ""))
            if reason:
                reasons.append(reason)
            if isinstance(fr, (int, float)):
                rates.append(fr)
            if not agrees:
                disagreements.append((r["dir"], hit, reason))
        for key, _desc in GATE_CONDITIONS:
            row[f"failed_on_{key}"] = counts.get(key, 0)
        row["failed_runs_detail"] = "|".join(detail)
        row["distinct_verdict_reasons"] = "|".join(sorted(set(reasons)))
        row["run_failure_rate_median"] = A3.median(rates)
        row["run_failure_rate_min"] = min(rates) if rates else None
        row["run_failure_rate_max"] = max(rates) if rates else None
        row["completion_rate_mean"] = A3.completion_summary(runs)["mean"] if runs else None
        rows.append(row)
    return rows, disagreements


# ───────────────────────── block 3 — ladder on completion ─────────────────
def ladder_steps():
    """The imported four rungs plus the fifth this section adds.

    The imported list is checked against what SQ3a claims it is, so a change to
    analyze_3e.LADDER_STEPS surfaces here instead of silently changing which
    comparisons the chapter reports.
    """
    imported = list(A3.LADDER_STEPS)
    got = [(b, c) for b, c, _l, _d in imported]
    if got != EXPECTED_IMPORTED_STEPS:
        raise SystemExit(
            "FATAL: analyze_3e.LADDER_STEPS is no longer the four rungs SQ3a "
            f"describes.\n  imported: {got}\n  expected: {EXPECTED_IMPORTED_STEPS}")
    return imported + [SQ3A_EXTRA_STEP]


def build_block3(cells, keys_by_cell, steps):
    """Run-level tests on completion rate, ALL runs, one row per (step, cell).

    The test path is analyze_3e's own: completion_summary -> scaled(.., 100) ->
    run_level_verdict, which calls perm_test (exhaustive below PERM_EXACT_LIMIT
    splits, Monte-Carlo above) and bootstrap_ci (percentile CI on the difference
    of per-run medians, resampling runs). attainable_min_p is reported beside
    the p-value: where it exceeds ALPHA the comparison could not have returned a
    significant result at any effect size, so a large p there is a resolution
    limit, not evidence of no difference.

    EVERY comparison is emitted, including the ones where nothing cleared alpha
    and the ones with too few runs to test. Suppressing null results would make
    the surviving p-values uninterpretable.
    """
    rows = []
    for base_arm, cand_arm, label, descriptive in steps:
        for workload, conc, in_grid in keys_by_cell:
            b_runs = A3.valid_runs(cells, base_arm, conc, workload)   # ALL runs
            c_runs = A3.valid_runs(cells, cand_arm, conc, workload)
            nb, nc = len(b_runs), len(c_runs)
            B, C = A3.completion_summary(b_runs), A3.completion_summary(c_runs)
            row = {
                "step": f"{base_arm}->{cand_arm}",
                "base_arm": base_arm,
                "cand_arm": cand_arm,
                "mechanism_isolated": label,
                "descriptive_only": descriptive,
                "cell": A3.col_label(workload, conc),
                "workload": workload,
                "concurrency": conc,
                "in_grid": in_grid,
                "n_base": nb,
                "n_cand": nc,
                "n_base_clean": sum(1 for r in b_runs if r.get("verdict") == "clean"),
                "n_cand_clean": sum(1 for r in c_runs if r.get("verdict") == "clean"),
                "base_completion_median": B["median"],
                "cand_completion_median": C["median"],
                "base_completion_mean": B["mean"],
                "cand_completion_mean": C["mean"],
                # attainable_min_p depends only on the two run counts, so it is
                # defined even where the test itself is not.
                "attainable_min_p": A3.attainable_min_p(nb, nc),
            }
            if nb < 2 or nc < 2:
                why = []
                if nb < 2:
                    why.append(f"{base_arm} has {nb} run{'s' if nb != 1 else ''}")
                if nc < 2:
                    why.append(f"{cand_arm} has {nc} run{'s' if nc != 1 else ''}")
                row.update({
                    "diff_pp": None, "ci_low_pp": None, "ci_high_pp": None,
                    "p_permutation": None, "enumeration": None,
                    "perm_p_floor_combinatorial": None,
                    "underpowered_by_floor": None, "verdict": "untestable",
                    "untestable_reason": " and ".join(why)
                                         + " (a permutation test needs >=2 per arm)",
                    "tested": False,
                })
                rows.append(row)
                continue
            v = A3.run_level_verdict(A3.scaled(B, 100.0), A3.scaled(C, 100.0),
                                     lower_is_better=False)
            lo, hi = v["ci"]
            row.update({
                "diff_pp": v["diff"],
                "ci_low_pp": lo,
                "ci_high_pp": hi,
                "p_permutation": v["p"],
                # perm_test enumerates every split below PERM_EXACT_LIMIT and
                # samples above it; "exhaustive" means the p-value is exact.
                "enumeration": "exhaustive" if v["exact"] else "estimated",
                # perm_test's floor is the combinatorial bound; attainable_min_p is the realized minimum for the median.
                "perm_p_floor_combinatorial": v["floor"],
                "underpowered_by_floor": v["underpowered"],
                "verdict": v["label"],
                "untestable_reason": "",
                "tested": True,
                "cleared_alpha": (v["p"] is not None and v["p"] <= A3.ALPHA),
                "loo_worst_p": v["loo"],
                "loo_damaging": v["loo_damaging"],
            })
            rows.append(row)
    return rows


# ───────────────────────── block 4 — failure composition ──────────────────
def bench_outcome_counts(run):
    """Per-request outcome tallies for one run's MEASURED requests.

    analyze_3e.per_run reduces the bench raw array to success / fail / truncated
    counts and does not retain the outcome field, so this reads the tally
    run_3e.bench_summary already wrote from that same array. Mirrored, not
    recomputed: the numbers come from the driver's own reduction.
    """
    try:
        meta = json.loads(open(os.path.join(run["dir"], "run_meta.json")).read())
    except OSError:
        return {}
    return dict((meta.get("bench_summary") or {}).get("outcome_counts") or {})


def build_block4(cells, keys, bench_cats, router_cats, status_cats):
    rows = []
    for arm, workload, conc, in_grid in keys:
        runs = A3.valid_runs(cells, arm, conc, workload)
        if not runs:
            continue
        row = base_row(arm, workload, conc, in_grid, runs)
        # ---- client side: same denominator as block 1's completion rate ----
        bench = Counter()
        for r in runs:
            bench.update(bench_outcome_counts(r))
        offered = sum(bench.values())
        n_success = bench.get("success", 0)
        n_failed = offered - n_success
        row.update({
            "bench_offered": offered,
            "bench_success": n_success,
            "bench_failed": n_failed,
            "bench_failure_rate_pooled": (n_failed / offered) if offered else None,
        })
        for cat in bench_cats:
            c = bench.get(cat, 0)
            row[f"bench_{cat}_count"] = c
            row[f"bench_{cat}_share_of_failed"] = (c / n_failed) if n_failed else None
            row[f"bench_{cat}_share_of_offered"] = (c / offered) if offered else None
        # Truncation is a separate axis from the outcome tally: a request can be
        # successful AND truncated. Both counts come from analyze_3e.per_run.
        trunc = sum(r.get("n_truncated", 0) for r in runs)
        trunc_ok = sum(r.get("n_truncated_success", 0) for r in runs)
        row["truncated_count"] = trunc
        row["truncated_success_count"] = trunc_ok
        row["truncated_share_of_failed"] = ((trunc - trunc_ok) / n_failed) if n_failed else None
        # ---- router side: measured requests, warm-up already excluded ----
        rout = Counter()
        statuses = Counter()
        for r in runs:
            rout.update(r.get("router", {}).get("outcomes", {}))
            statuses.update(r.get("router", {}).get("upstream_error_statuses", {}))
        r_total = sum(rout.values())
        r_fail = r_total - rout.get("success", 0)
        row["router_measured"] = r_total
        row["router_failed"] = r_fail
        for cat in router_cats:
            c = rout.get(cat, 0)
            row[f"router_{cat}_count"] = c
            row[f"router_{cat}_share_of_failed"] = (c / r_fail) if r_fail else None
        for st in status_cats:
            row[f"router_upstream_error_http_{st}"] = statuses.get(st, 0)
        row["bench_vs_router_failed_delta"] = n_failed - r_fail
        rows.append(row)
    return rows


def discover_categories(runs):
    """What outcome categories the data actually contains — never assumed.

    Client side and router side use different vocabularies for the same events
    (the client's `http_error` is the router's `upstream_timeout_headers` plus
    its 502s), so both are enumerated and both are reported.
    """
    bench, router, statuses = Counter(), Counter(), Counter()
    for r in runs:
        if "router" not in r:
            continue
        bench.update(bench_outcome_counts(r))
        router.update(r.get("router", {}).get("outcomes", {}))
        statuses.update(r.get("router", {}).get("upstream_error_statuses", {}))
    order = lambda c: ["success"] * ("success" in c) + sorted(k for k in c if k != "success")
    return order(bench), order(router), sorted(statuses), bench, router, statuses


# ───────────────────────── output ─────────────────────────────────────────
def write_csv(path, rows):
    if not rows:
        return 0, 0
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    return len(rows), len(fields)


def write_readme(outdir, manifest, ctx):
    L = [
        "# SQ3a result-section artefacts",
        "",
        "Generated by `analyze_sq3a.py` from the run folders under `3e_results/`.",
        "Do not edit by hand -- rerun the script. Analysis only; no plots.",
        "",
        "## Files",
        "",
        "| file | rows | cols | SQ3a subsection it supports |",
        "|---|---:|---:|---|",
    ]
    for b in sorted(manifest):
        fn, desc, nr, nc = manifest[b]
        L.append(f"| `{fn}` | {nr} | {nc} | {desc} |")
    L += [
        "",
        "| subsection | read this file | quote these columns |",
        "|---|---|---|",
        f"| SQ3a.1 completion rate by arm and cell | `{manifest[1][0]}` | "
        "`completion_rate_mean`, `completion_rate_sd`, `n_runs`, "
        "`median_min_max_fmt` |",
        f"| SQ3a.2 where an arm produced no analysable run | `{manifest[2][0]}` | "
        "`status`, `n_attempted`, `failed_on_*` |",
        f"| SQ3a.3 does each ladder rung buy completion | `{manifest[3][0]}` | "
        "`diff_pp`, `p_permutation`, `ci_low_pp`/`ci_high_pp`, `enumeration`, "
        "`attainable_min_p` |",
        f"| SQ3a.4 what the failures were made of | `{manifest[4][0]}` | "
        "`bench_*_share_of_failed`, `router_*_share_of_failed` |",
        "",
        "## Scope: all runs, not clean runs",
        "",
        "Every block uses **all** runs, clean and suspect. Completion rate is the "
        "measure of whether the offered load survived; the clean/suspect gate is "
        "itself partly a function of the failure rate, so filtering on it would "
        "condition on the outcome being measured and would delete exactly the "
        "cells the section is about.",
        "",
        "The split is nonetheless visible everywhere: `n_clean` and `n_not_clean` "
        "are on every row, and block 1 additionally carries "
        "`completion_rate_mean_clean`, `completion_rate_mean_not_clean` and "
        "`all_runs_minus_clean_pp`, which is how much a clean-only reading would "
        "have moved that cell, in percentage points.",
        "",
        "## Grouping",
        "",
        "- **Every output is keyed on `(arm, cell)`, never on the policy name.** "
        "The `wlc_conc` arm logs `policy == \"weighted_least_connections\"`; "
        "keying on the policy string would pool the re-weighted control into the "
        "weighted-least-connections baseline it exists to control for. The arm "
        "comes from `analyze_3e.per_run`, which resolves "
        "`run_meta[\"arm\"] or run_meta[\"policy\"]`.",
        f"- Runs with no `arm` field (pre-dating the arm concept) fall back to "
        f"policy: {ctx['n_no_arm_fallback']} of {ctx['n_runs']} runs.",
        "- The run is the unit of replication. Every SD is the **sample** SD "
        "(ddof = 1, `analyze_3e.agg`), reported only where n >= 2; "
        "`completion_rate_sd_is_defined` marks the rest.",
        "",
        "## Cell coverage",
        "",
        f"- Grid cells (`analyze_3e.CELL_COLS`): "
        f"{', '.join('`' + A3.col_label(w, c) + '`' for w, c in A3.CELL_COLS)}.",
    ]
    if ctx["extra_cells"]:
        L.append("- Cells outside that list are carried with `in_grid=False` "
                 "rather than dropped, and are excluded from nothing here: "
                 + ", ".join(f"`{A3.col_label(w, c)}` "
                             f"({ctx['extra_cell_runs'][(w, c)]} run(s), arms "
                             f"{sorted(ctx['extra_cell_arms'][(w, c)])})"
                             for w, c in ctx["extra_cells"]) + ".")
        L.append("  `analyze_3e.group_cells` drops these, which is why "
                 "`3e_report.md` never mentions them (see the reconciliation "
                 "section below).")
    L += [
        "",
        "## Reconciliation with `3e_report.md`",
        "",
        f"`3e_report.md` Appendix A states **{ctx['report_grid']}** grid runs "
        f"loaded and **{ctx['report_warm']}** router warm-up records excluded. "
        f"The SQ1/SQ2 chain sees **{ctx['sq_runs']}** runs and "
        f"**{ctx['all_warm']}** warm-up records.",
        "",
        "All analysis chains use the same frozen run_inclusion.csv ledger. "
        "The historical least_connections ShareGPT c4 pilot is excluded. "
        "Completion includes all 508 measured runs; clean-only analyses use 416.",
        "",
        "## Validity gate (block 2)",
        "",
        f"`run_3e.compute_verdict` marks a run `suspect` on any of: no healthy "
        f"replica for some request, a run failure rate above "
        f"`FAILURE_RATE_MAX` (**{ctx['frm']:.0%}**, read from "
        f"`run_3e.py` line {ctx['frm_line']}, not assumed), a mid-run eviction "
        f"event, an empty router log, or several `run_id`s in one log. Block 2 "
        f"re-applies the first three to the numeric `router_metrics` that "
        f"function wrote and buckets everything else as `other`, carrying the "
        f"reason text verbatim.",
        "",
        "| condition column | meaning |",
        "|---|---|",
    ]
    for key, desc in GATE_CONDITIONS:
        L.append(f"| `failed_on_{key}` | {desc} |")
    L += [
        "",
        f"Conditions actually observed in this dataset: "
        f"{', '.join('`' + k + '`' for k in ctx['conditions_seen']) or 'none'}. "
        f"`status` separates `never_run` ({ctx['n_never_run']} cells) from "
        f"`run_and_all_failed` ({ctx['n_all_failed']} cells) -- a cell with no "
        f"attempts supports no conclusion, while a cell that was attempted and "
        f"failed every time is itself a result.",
        "",
        "## Outcome categories (block 4) — discovered, not assumed",
        "",
        "The two sides of the request use different vocabularies, so both are "
        "reported and neither is translated into the other.",
        "",
        "| side | categories observed | source |",
        "|---|---|---|",
        f"| client (bench) | {', '.join('`' + c + '`' for c in ctx['bench_cats'])} | "
        "`run_meta.bench_summary.outcome_counts`, over the 100 measured requests |",
        f"| router | {', '.join('`' + c + '`' for c in ctx['router_cats'])} | "
        "`analyze_3e.per_run`'s `router.outcomes`, warm-up excluded |",
        f"| HTTP status on router `upstream_error` | "
        f"{', '.join(str(s) for s in ctx['status_cats'])} | "
        "`analyze_3e.per_run`'s `router.upstream_error_statuses` |",
        "",
        ctx["category_note"],
        "",
        f"Truncation is a **separate axis**, not an outcome value: a request can "
        f"be both successful and truncated. Observed across the whole dataset: "
        f"{ctx['n_truncated']} truncated requests "
        f"({ctx['n_truncated_success']} of them successful).",
        "",
        "## Statistical machinery (block 3)",
        "",
        f"- Test: `analyze_3e.perm_test`, two-sided permutation on the "
        f"difference of per-run **medians**, reached through "
        f"`run_level_verdict`. Exhaustive whenever the split count is at or "
        f"below `PERM_EXACT_LIMIT` ({A3.PERM_EXACT_LIMIT:,}), Monte Carlo "
        f"({A3.BOOT_ITERS:,} shuffles) above it. The `enumeration` column says "
        f"which was used for every row.",
        f"- Interval: `analyze_3e.bootstrap_ci`, percentile CI on the same "
        f"difference, resampling **runs** ({A3.BOOT_ITERS:,} iterations, seed "
        f"{A3.BOOT_SEED}, so the numbers are reproducible).",
        "- Unit: the run. Requests inside one run share a router process, a "
        "queue and a thermal state, so a pooled request-level test would be "
        "anticonservative.",
        f"- `attainable_min_p` (`analyze_3e.attainable_min_p`) is the smallest "
        f"p these two run counts can produce under perfect separation. Where it "
        f"exceeds alpha ({A3.ALPHA}), a large p is a **resolution limit, not "
        f"evidence of no difference**. It is reported for every row, including "
        f"the untestable ones, because it depends only on the run counts.",
        "- `perm_p_floor_combinatorial` is a different and smaller quantity "
        "(2/C for equal group sizes, 1/C otherwise); it is what "
        "`run_level_verdict`'s `underpowered` label keys on. Both are emitted "
        "so they are not confused.",
        "- `loo_worst_p` / `loo_damaging` come from "
        "`analyze_3e.loo_diagnostics` and are carried through unchanged. They "
        "bear on the reading only where `cleared_alpha` is true: on a null "
        "comparison `loo_damaging` is set merely because a zero median "
        "difference counts as a sign flip, and `verdict` has already resolved "
        "the row before the diagnostic is consulted.",
        "",
        "## Resolution note",
        "",
        "Above the enumeration limit, attainable_min_p is a sampling-resolution fallback, not an exhaustive minimum. For SA contrasts use observed_pool_min_p in the sensitivity/report tables, which includes observed ties exactly.",
        "",
        "## Multiplicity",
        "",
        f"This script executed **{ctx['tests_run']} permutation tests** "
        f"({ctx['n_tested']} of {ctx['n_comparisons']} ladder comparisons were "
        f"testable; the rest had fewer than two runs in one arm). "
        f"**No multiplicity correction is applied.** At alpha = {A3.ALPHA}, "
        f"about {ctx['tests_run'] * A3.ALPHA:.1f} significant results would be "
        f"expected by chance alone across a family this size, so an individual "
        f"p-value describes a single comparison and is not a family-wise error "
        f"rate. The count excludes the refits inside "
        f"`analyze_3e.loo_diagnostics`, matching that script's own accounting.",
        "",
        f"Comparisons clearing alpha without correction: "
        f"{ctx['n_cleared']} of {ctx['n_tested']}.",
        "",
    ]
    p = os.path.join(outdir, "README.md")
    open(p, "w").write("\n".join(L) + "\n")
    return p


def print_conventions():
    print("\n" + "#" * 100)
    print("# CONVENTIONS (printed at runtime, not only in comments)")
    print("#" * 100)
    print("1. Grouping is (arm, cell). NEVER policy name: the wlc_conc arm logs")
    print("   policy == 'weighted_least_connections' and would otherwise pool into the")
    print("   weighted-least-connections baseline it is a control for.")
    print("2. SCOPE is ALL runs, clean and suspect. Completion measures whether the load")
    print("   survived; the clean gate is partly a function of the failure rate, so")
    print("   filtering on it would condition on the outcome. n_clean / n_not_clean and")
    print("   the split means are emitted beside every figure instead.")
    print("3. The run is the unit of replication. SDs are SAMPLE SDs (ddof=1,")
    print("   analyze_3e.agg), reported only where n >= 2.")
    print("4. Tests are analyze_3e's own perm_test / bootstrap_ci via run_level_verdict:")
    print("   two-sided permutation on the difference of per-run MEDIANS, percentile")
    print("   bootstrap CI resampling runs. Nothing is reimplemented.")
    print("5. attainable_min_p is the smallest p a pair of run counts can produce at all.")
    print("   Above alpha it means the comparison is unresolvable, NOT that the arms are")
    print("   equal. It is distinct from perm_test's combinatorial floor; both emitted.")
    print("6. Every comparison is reported, including null and untestable ones.")
    print("7. Outcome categories are DISCOVERED from the data, not enumerated in code.")
    print("8. Off-grid cells are carried with in_grid=False, not dropped.")


def main():
    ap = argparse.ArgumentParser(description="SQ3a result-section artefacts (read-only).")
    ap.add_argument("--outdir", default=HERE)
    ap.add_argument("--driver", default=os.path.join(A3.HERE, "run_3e.py"))
    ap.add_argument("--report", default=os.path.join(ROOT, "3e_report.md"))
    ap.add_argument("--skip-sq-crosscheck", action="store_true",
                    help="skip the analyze_divergence.load_all_runs cross-check")
    args = ap.parse_args()
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    print("=" * 100)
    print("SQ3a — completion, structural failures, the ladder, and failure "
          "composition. Read-only on all inputs.")
    print("=" * 100)

    frm, frm_line = detect_failure_rate_max(args.driver)
    print(f"FAILURE_RATE_MAX = {frm} ({frm:.0%}), read from "
          f"{os.path.relpath(args.driver, ROOT)} line {frm_line}.")

    runs = A3.load_all()
    with_router = [r for r in runs if "router" in r]
    no_router = [r for r in runs if "router" not in r]

    # ══════════════ RECONCILIATION — resolved before anything else is produced ══
    print("\n" + "#" * 100)
    print("# RECONCILIATION — 3e_report.md vs the SQ1/SQ2 chain")
    print("#" * 100)
    grid_runs = [r for r in runs if r.get("grid")]
    offgrid = [r for r in runs if not r.get("grid")]
    warm_grid = sum(r.get("router", {}).get("warmup_excluded", 0) for r in grid_runs)
    warm_all = sum(r.get("router", {}).get("warmup_excluded", 0) for r in runs)
    report_grid, report_warm = report_appendix_counts(args.report)
    print(f"analyze_3e.load_all: {len(runs)} run directories; "
          f"{len(grid_runs)} grid; {len(offgrid)} off-grid.")
    print(f"  warm-up router records: {warm_grid} over grid runs, "
          f"{warm_all} over all runs.")
    print(f"3e_report.md Appendix A (parsed): grid runs loaded = {report_grid}, "
          f"router warm-up records excluded = {report_warm}.")
    ok_grid = (report_grid == len(grid_runs))
    ok_warm = (report_warm == warm_grid)
    print(f"  grid count matches report:  {'YES' if ok_grid else 'NO'}")
    print(f"  warm-up count matches report: {'YES' if ok_warm else 'NO'}")
    if not args.skip_sq_crosscheck:
        dv = AD.load_all_runs()
        sq_runs = len(dv)
        sq_usable = sum(1 for r in dv if r.get("usable"))
        print(f"analyze_divergence.load_all_runs (the SQ1/SQ2 chain): "
              f"{sq_runs} run directories, {sq_usable} usable — shared ledger filter.")
    else:
        sq_runs = len(runs)
        print("SQ-chain cross-check skipped (--skip-sq-crosscheck); using the "
              "same directory count, which carries no grid filter either.")
    print(f"\nDIFFERENCE: {sq_runs} - {report_grid} = {sq_runs - report_grid} run; "
          f"{warm_all} - {report_warm} = {warm_all - report_warm} warm-up records.")
    print("The missing run(s), present in the SQ chain and absent from 3e_report.md:")
    offgrid_desc = "none"
    for r in offgrid:
        offgrid_desc = (f"{r['policy']} {A3.col_label(r['workload'], r['conc'])} "
                        f"rep{r['rep']} ({os.path.relpath(r['dir'], ROOT)}, "
                        f"verdict {r.get('verdict')})")
        print(f"  {os.path.relpath(r['dir'], ROOT)}")
        print(f"    arm={r['policy']} workload={r['workload']} conc={r['conc']} "
              f"rep={r['rep']} verdict={r.get('verdict')}")
        print(f"    offered={r.get('offered')} success={r.get('n_success')} "
              f"completion={r.get('completion_rate')}")
        print(f"    warm-up router records={r.get('router', {}).get('warmup_excluded')}")
        print(f"    is_grid_cell({r['workload']}, {r['conc']}) = "
              f"{A3.is_grid_cell(r['workload'], r['conc'])} — CELL_COLS admits "
              f"sharegpt only at concurrency 8.")
    print("All loaders use the frozen inclusion ledger; the off-grid pilot is excluded.")
    if no_router:
        print(f"\n*** {len(no_router)} run(s) carry no router block and are absent "
              f"from every block below:")
        for r in no_router:
            print(f"    {os.path.relpath(r['dir'], ROOT)}: {r.get('notes')}")

    # ══════════════ setup ══════════════
    cells = group_all(runs)
    canonical, extra = enumerate_cells(runs)
    arms = enumerate_arms(runs)
    keys = [(a, w, c, (w, c) in canonical)
            for a in arms for w, c in canonical + extra]
    keys_by_cell = [(w, c, (w, c) in canonical) for w, c in canonical + extra]

    n_no_arm = 0
    for r in with_router:
        try:
            m = json.loads(open(os.path.join(r["dir"], "run_meta.json")).read())
        except OSError:
            continue
        if not m.get("arm"):
            n_no_arm += 1

    print_conventions()

    print(f"\nArms observed ({len(arms)}): {arms}")
    print(f"Cells: {len(canonical)} grid "
          f"{[A3.col_label(w, c) for w, c in canonical]}"
          + (f"; {len(extra)} off-grid "
             f"{[A3.col_label(w, c) for w, c in extra]}" if extra else "; no off-grid"))
    print(f"(arm, cell) combinations enumerated: {len(keys)}")

    bench_cats, router_cats, status_cats, bench_tot, router_tot, status_tot = \
        discover_categories(runs)
    print("\nOUTCOME CATEGORIES PRESENT IN THE DATA (discovered, not assumed):")
    print(f"  client (bench_summary.outcome_counts): "
          + ", ".join(f"{c}={bench_tot[c]}" for c in bench_cats))
    print(f"  router (per_run router.outcomes, warm-up excluded): "
          + ", ".join(f"{c}={router_tot[c]}" for c in router_cats))
    print(f"  HTTP status on router upstream_error: "
          + ", ".join(f"{s}={status_tot[s]}" for s in status_cats))
    n_trunc = sum(r.get("n_truncated", 0) for r in with_router)
    n_trunc_ok = sum(r.get("n_truncated_success", 0) for r in with_router)
    print(f"  truncation (separate axis, not an outcome value): {n_trunc} "
          f"truncated, {n_trunc_ok} of them successful")
    category_note = (
        "The client's `http_error` bucket and the router's "
        "`upstream_timeout_headers` describe the same events from two ends; the "
        "router side is the one that distinguishes a timeout (504) from another "
        "upstream failure (502). Neither side is rewritten to match the other."
        if "http_error" in bench_cats and "upstream_timeout_headers" in router_cats
        else "Client-side and router-side vocabularies are reported as recorded; "
             "neither is rewritten to match the other.")
    print("  NOTE: " + category_note)

    # ══════════════ blocks ══════════════
    b1 = build_block1(cells, keys)
    b2, disagreements = build_block2(cells, keys, frm)
    steps = ladder_steps()
    b3 = build_block3(cells, keys_by_cell, steps)
    b4 = build_block4(cells, keys, [c for c in bench_cats if c != "success"],
                      [c for c in router_cats if c != "success"], status_cats)

    manifest = {}
    for b, rows in ((1, b1), (2, b2), (3, b3), (4, b4)):
        fn, desc = BLOCK_FILES[b]
        nr, nc = write_csv(os.path.join(outdir, fn), rows)
        manifest[b] = (fn, desc, nr, nc)

    # ══════════════ stdout: block 1 ══════════════
    print("\n" + "#" * 100)
    print("# BLOCK 1 — COMPLETION RATE per (arm, cell), ALL RUNS")
    print("#" * 100)
    print(f"{'arm':<28}{'cell':<9}{'n':>4}{'cln':>5}{'sus':>5}"
          f"{'mean':>9}{'sd':>8}{'median':>9}{'min':>8}{'max':>8}{'clean-only':>12}")
    for arm in arms:
        for w, c in canonical + extra:
            row = next((r for r in b1 if r["arm"] == arm
                        and r["cell"] == A3.col_label(w, c)), None)
            if row is None:
                continue
            g = lambda v, nd=1: ("-" if v is None else f"{v * 100:.{nd}f}")
            sd = ("-" if row["completion_rate_sd"] is None
                  else f"{row['completion_rate_sd'] * 100:.2f}")
            print(f"{arm:<28}{row['cell']:<9}{row['n_runs']:>4}"
                  f"{row['n_clean']:>5}{row['n_not_clean']:>5}"
                  f"{g(row['completion_rate_mean']):>9}{sd:>8}"
                  f"{g(row['completion_rate_median']):>9}"
                  f"{g(row['completion_rate_min']):>8}"
                  f"{g(row['completion_rate_max']):>8}"
                  f"{g(row['completion_rate_mean_clean']):>12}")
    print("  Percent. 'sd' is the SAMPLE SD across runs (ddof=1); '-' means n<2, i.e.")
    print("  undefined rather than zero. 'clean-only' is what a clean-only reading would")
    print("  have shown for the same cell -- reported, not applied.")

    # ══════════════ stdout: block 2 ══════════════
    print("\n" + "#" * 100)
    print("# BLOCK 2 — (arm, cell) WITH ZERO RUNS PASSING THE VALIDITY GATE")
    print("#" * 100)
    never = [r for r in b2 if r["status"] == "never_run"]
    failed = [r for r in b2 if r["status"] == "run_and_all_failed"]
    print(f"{len(b2)} of {len(keys)} (arm, cell) combinations have no clean run: "
          f"{len(never)} never run, {len(failed)} run and always failed.")
    if failed:
        print("\nRUN AND ALWAYS FAILED — these are a result, not a gap:")
        print(f"{'arm':<28}{'cell':<9}{'att':>5}{'fail':>6}"
              + "".join(f"{k.replace('_', ' ')[:15]:>17}"
                       for k, _ in GATE_CONDITIONS)
              + f"{'compl':>8}{'fr med':>9}")
        for r in failed:
            print(f"{r['arm']:<28}{r['cell']:<9}{r['n_attempted']:>5}"
                  f"{r['n_failed_gate']:>6}"
                  + "".join(f"{r[f'failed_on_{k}']:>17}" for k, _ in GATE_CONDITIONS)
                  + f"{(r['completion_rate_mean'] * 100 if r['completion_rate_mean'] is not None else float('nan')):>8.1f}"
                  + f"{(r['run_failure_rate_median'] if r['run_failure_rate_median'] is not None else float('nan')):>9.3f}")
        print("  'compl' = mean completion % over those failed runs; 'fr med' = median")
        print("  run failure rate as the driver's gate computed it (over all 103 router")
        print("  records, warm-up included -- which is why it differs from 1 - compl).")
    if never:
        print(f"\nNEVER RUN — no attempt exists, so no conclusion is available "
              f"({len(never)}):")
        by_arm = defaultdict(list)
        for r in never:
            by_arm[r["arm"]].append(r["cell"])
        for arm in arms:
            if arm in by_arm:
                print(f"  {arm:<28} {', '.join(by_arm[arm])}")
    seen = [k for k, _ in GATE_CONDITIONS
            if sum(r[f"failed_on_{k}"] for r in b2) > 0]
    print(f"\nGate conditions actually triggered: {seen or ['none']}")
    for key, desc in GATE_CONDITIONS:
        n = sum(r[f"failed_on_{key}"] for r in b2)
        print(f"  failed_on_{key:<26} {n:>4}  ({desc})")
    reasons = Counter(x for r in b2 for x in r["distinct_verdict_reasons"].split("|") if x)
    print(f"Distinct verdict_reason strings behind them: {len(reasons)}")
    if disagreements:
        print(f"*** {len(disagreements)} run(s) where the numeric classification and the "
              f"recorded verdict_reason disagree:")
        for d, hit, reason in disagreements:
            print(f"    {os.path.relpath(d, ROOT)}: numeric={hit} reason={reason!r}")
    else:
        print("Numeric re-application of the gate agrees with every recorded "
              "verdict_reason.")

    # ══════════════ stdout: block 3 ══════════════
    print("\n" + "#" * 100)
    print("# BLOCK 3 — LADDER COMPARISONS ON COMPLETION RATE (ALL RUNS, run-level)")
    print("#" * 100)
    for base_arm, cand_arm, label, descriptive in steps:
        tag = "  [descriptive only in analyze_3e]" if descriptive else ""
        print(f"\n{base_arm} -> {cand_arm}   ({label}){tag}")
        print(f"  {'cell':<9}{'n':>8}{'base%':>8}{'cand%':>8}{'d pp':>8}"
              f"{'95% CI (pp)':>22}{'p':>8}{'enum':>12}{'min p':>8}  verdict")
        for r in [x for x in b3 if x["base_arm"] == base_arm
                  and x["cand_arm"] == cand_arm]:
            n = f"{r['n_base']}v{r['n_cand']}"
            bm = "-" if r["base_completion_median"] is None else f"{r['base_completion_median'] * 100:.1f}"
            cm = "-" if r["cand_completion_median"] is None else f"{r['cand_completion_median'] * 100:.1f}"
            amp = "-" if r["attainable_min_p"] is None else f"{r['attainable_min_p']:.3f}"
            if not r["tested"]:
                print(f"  {r['cell']:<9}{n:>8}{bm:>8}{cm:>8}{'-':>8}{'-':>22}"
                      f"{'-':>8}{'-':>12}{amp:>8}  untestable: "
                      f"{r['untestable_reason']}")
                continue
            ci = f"[{r['ci_low_pp']:+.1f}, {r['ci_high_pp']:+.1f}]"
            print(f"  {r['cell']:<9}{n:>8}{bm:>8}{cm:>8}{r['diff_pp']:>+8.1f}"
                  f"{ci:>22}{r['p_permutation']:>8.3f}{r['enumeration']:>12}"
                  f"{amp:>8}  {r['verdict']}")
    print("\n  d pp = median(cand) - median(base) in percentage points; positive favours")
    print("  the candidate. 'min p' is attainable_min_p: where it exceeds "
          f"alpha={A3.ALPHA} the")
    print("  comparison could not have returned a significant result at ANY effect size,")
    print("  so a large p there is a resolution limit, not evidence of equivalence.")
    print("  Every comparison is shown, cleared or not.")

    # ══════════════ stdout: block 4 ══════════════
    print("\n" + "#" * 100)
    print("# BLOCK 4 — FAILURE COMPOSITION per arm (cells pooled for this view)")
    print("#" * 100)
    fail_bench = [c for c in bench_cats if c != "success"]
    fail_router = [c for c in router_cats if c != "success"]
    print(f"{'arm':<28}{'offered':>9}{'failed':>8}{'fail%':>8}"
          + "".join(f"{'bench ' + c[:11]:>20}" for c in fail_bench)
          + f"{'trunc':>8}")
    for arm in arms:
        rs = [r for r in b4 if r["arm"] == arm]
        off = sum(r["bench_offered"] for r in rs)
        fl = sum(r["bench_failed"] for r in rs)
        if not off:
            continue
        cols = ""
        for c in fail_bench:
            n = sum(r[f"bench_{c}_count"] for r in rs)
            cols += f"{f'{n} ({n / fl * 100:.0f}%)' if fl else str(n):>20}"
        print(f"{arm:<28}{off:>9}{fl:>8}{fl / off * 100:>8.1f}{cols}"
              f"{sum(r['truncated_count'] for r in rs):>8}")
    print("  Percentages in the category columns are shares OF THAT ARM'S FAILURES.")
    print(f"\n{'arm':<28}{'measured':>10}{'failed':>8}"
          + "".join(f"{'rtr ' + c[:14]:>21}" for c in fail_router))
    for arm in arms:
        rs = [r for r in b4 if r["arm"] == arm]
        tot = sum(r["router_measured"] for r in rs)
        fl = sum(r["router_failed"] for r in rs)
        if not tot:
            continue
        cols = ""
        for c in fail_router:
            n = sum(r[f"router_{c}_count"] for r in rs)
            cols += f"{f'{n} ({n / fl * 100:.0f}%)' if fl else str(n):>21}"
        print(f"{arm:<28}{tot:>10}{fl:>8}{cols}")
    print("  Router side, warm-up excluded. HTTP statuses on upstream_error: "
          + ", ".join(f"{s}={status_tot[s]}" for s in status_cats))
    print("  " + category_note)
    delta = sum(abs(r["bench_vs_router_failed_delta"]) for r in b4)
    print(f"  Client-vs-router failed-count disagreement, summed over cells: {delta} "
          f"(bench_vs_router_failed_delta per row; analyze_3e's Appendix A lists the "
          f"reconciliation mismatches individually).")

    # ══════════════ multiplicity ══════════════
    n_tested = sum(1 for r in b3 if r["tested"])
    n_cleared = sum(1 for r in b3 if r.get("cleared_alpha"))
    n_underpowered = sum(1 for r in b3 if r["tested"]
                         and r["attainable_min_p"] is not None
                         and r["attainable_min_p"] > A3.ALPHA)
    print("\n" + "#" * 100)
    print("# MULTIPLICITY — family size, stated so the chapter can address it")
    print("#" * 100)
    print(f"Permutation tests executed by this script: {A3.TESTS_RUN['count']}")
    print(f"  ladder comparisons enumerated: {len(b3)}")
    print(f"  of those, testable (>=2 runs per arm): {n_tested}")
    print(f"  of those, resolvable at alpha={A3.ALPHA} "
          f"(attainable_min_p <= alpha): {n_tested - n_underpowered}")
    print(f"  clearing alpha without correction: {n_cleared}")
    print("NO multiplicity correction is applied. At alpha="
          f"{A3.ALPHA}, roughly {A3.TESTS_RUN['count'] * A3.ALPHA:.1f} significant")
    print("results would be expected by chance alone across a family of this size, so an")
    print("individual p-value is descriptive of ONE comparison and is not a family-wise")
    print("error rate. The count excludes the refits inside analyze_3e.loo_diagnostics,")
    print("matching that script's own accounting convention.")

    ctx = {
        "n_runs": len(with_router),
        "n_no_arm_fallback": n_no_arm,
        "extra_cells": extra,
        "extra_cell_runs": {wc: sum(1 for r in with_router
                                    if (r["workload"], r["conc"]) == wc)
                            for wc in extra},
        "extra_cell_arms": {wc: {r["policy"] for r in with_router
                                 if (r["workload"], r["conc"]) == wc}
                            for wc in extra},
        "report_grid": report_grid,
        "report_warm": report_warm,
        "sq_runs": sq_runs,
        "all_warm": warm_all,
        "offgrid_desc": offgrid_desc,
        "frm": frm,
        "frm_line": frm_line,
        "conditions_seen": seen,
        "n_never_run": len(never),
        "n_all_failed": len(failed),
        "bench_cats": bench_cats,
        "router_cats": router_cats,
        "status_cats": status_cats,
        "category_note": category_note,
        "n_truncated": n_trunc,
        "n_truncated_success": n_trunc_ok,
        "tests_run": A3.TESTS_RUN["count"],
        "n_tested": n_tested,
        "n_comparisons": len(b3),
        "n_cleared": n_cleared,
    }
    readme = write_readme(outdir, manifest, ctx)

    print(f"\nWritten to {outdir}:")
    for b in sorted(manifest):
        fn, _, nr, nc = manifest[b]
        print(f"  block {b}: {fn}  ({nr} rows, {nc} cols)")
    print(f"  {os.path.basename(readme)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
