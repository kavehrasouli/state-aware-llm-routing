import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)  # combined analysis helpers
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

import analyze_divergence as AD  # noqa: E402
# resolve_num_warmup is defined once, in analyze_sq1, and imported here rather
# than duplicated so both SQ sections resolve the warm-up count identically.
from analyze_sq1 import resolve_num_warmup  # noqa: E402

SCOPES = ("all", "clean", "not_clean")

BLOCK_FILES = {
    1: ("sq2_routing_overhead.csv",
        "SQ2 routing overhead: per-request selection time by arm and cell."),
    2: ("sq2_overhead_decomposition.csv",
        "SQ2 overhead decomposition: what the extended timer is actually spending, for the arms whose timer covers body read and token estimation."),
    3: ("sq2_overhead_share_of_latency.csv",
        "SQ2 overhead in context: selection time as a share of end-to-end request latency."),
    4: ("sq2_staleness.csv",
        "SQ2 metric freshness: per-node stale rates and the age of stale state at decision time."),
    5: ("sq2_stale_handling.csv",
        "SQ2 staleness response: what the router did when state was stale."),
}

STALE_MODES_SEEN = set()   # filled at scan time; never hardcoded


# ───────────────────────── small stat helpers ─────────────────────────────
def stats_of(values):
    """median / p90 / p99 of a raw sample, via analyze_divergence.pct so the
    percentile definition matches every other artefact in this thesis."""
    if not values:
        return {"median": None, "p90": None, "p99": None, "n": 0}
    return {"median": AD.median(values), "p90": AD.pct(values, 90),
            "p99": AD.pct(values, 99), "n": len(values)}


def across_runs(per_run):
    """mean and SAMPLE SD (ddof=1, via analyze_divergence.agg) of a per-run
    statistic. agg returns std=0.0 when only one run contributes, which is
    undefined rather than zero variance -- the n is carried so that is visible."""
    a = AD.agg([v for v in per_run if v is not None])
    return a["mean"], (a["std"] if a["n"] else None), a["n"]


def emit_stat_columns(row, prefix, per_run_lists, pooled_values):
    """Both conventions, side by side and named so they cannot be confused:
    *_runmean/_runsd = statistic computed per run then averaged across runs
    (the run is the unit of replication); *_pooled = the same statistic over
    all requests pooled, ignoring run boundaries."""
    for stat in ("median", "p90", "p99"):
        per_run = [stats_of(v)[stat] for v in per_run_lists if v]
        m, s, n = across_runs(per_run)
        row[f"{prefix}_{stat}_runmean"] = m
        row[f"{prefix}_{stat}_runsd"] = s
        row[f"{prefix}_{stat}_n_runs"] = n
    pooled = stats_of(pooled_values)
    for stat in ("median", "p90", "p99"):
        row[f"{prefix}_{stat}_pooled"] = pooled[stat]
    row[f"{prefix}_n_requests"] = pooled["n"]


# ───────────────────────── one pass per run ───────────────────────────────
def scan(meta, log_path, num_warmup=0, warmup_mode="all"):
    """One pass over one run's log, collecting every block's raw material.

    WARM-UP EXCLUSION (added 2026-08-24). Methodology: "Each run issues three
    warm-up requests before the 100 measured requests; warm-up is excluded from
    every statistic, on both the client and the router side." Until this date
    every SQ2 block counted all 103 router records per run.

    The predicate is analyze_3e.py's, at its lines 224-225 --
        isinstance(request.get("request_id"), int) and request["request_id"] >= num_warmup
    -- mirrored rather than imported because it sits inline in that script's own
    per-run loop.

    warmup_mode: "measured" keeps request_id >= num_warmup (what gets written);
                 "warmup"   keeps request_id <  num_warmup (so the excluded
                            population can be characterised, not just subtracted);
                 "all"      applies no filter (reproduces the pre-2026-08-24
                            numbers for the before/after comparison).
    Named warmup_mode, not mode: the record loop below already binds `mode` to
    the record's stale_mode.
    """
    select_us, body_us, est_us, resid_us = [], [], [], []
    ratio_per_req, sel_for_ratio, tot_for_ratio = [], [], []
    n_records = 0
    node_stale = {n: 0 for n in AD.NODES}
    node_age = {n: [] for n in AD.NODES}
    n_complete = 0
    any_stale = all_stale = 0
    modes = Counter()
    on_stale = 0
    logged_max_age = set()

    with open(log_path) as fh:
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
            # Warm-up filter, before any counter is touched.
            if warmup_mode != "all":
                rid = r.get("request_id")
                is_measured = isinstance(rid, int) and rid >= num_warmup
                if (warmup_mode == "measured") != is_measured:
                    continue
            n_records += 1

            if r.get("max_state_age_ms") is not None:
                logged_max_age.add(r["max_state_age_ms"])
            mode = r.get("stale_mode")
            modes[mode] += 1
            STALE_MODES_SEEN.add(mode)
            if r.get("decision_on_stale_state") is True:
                on_stale += 1

            sel_ns = r.get("select_ns")
            if sel_ns is not None:
                select_us.append(sel_ns / 1000.0)
                b, e = r.get("request_body_read_ns"), r.get("request_token_estimate_ns")
                if b is not None and e is not None:
                    body_us.append(b / 1000.0)
                    est_us.append(e / 1000.0)
                    resid_us.append((sel_ns - b - e) / 1000.0)
                tot_ms = r.get("total_ms")
                if tot_ms is not None and tot_ms > 0:
                    sel_for_ratio.append(sel_ns / 1000.0)
                    tot_for_ratio.append(tot_ms)
                    # select_us is microseconds, total_ms milliseconds
                    ratio_per_req.append((sel_ns / 1000.0) / 1000.0 / tot_ms)

            sv = r.get("state_vector")
            if sv and isinstance(sv, list):
                by = {e.get("replica"): e for e in sv}
                if all(n in by for n in AD.NODES):
                    n_complete += 1
                    flags = []
                    for n in AD.NODES:
                        st = by[n].get("stale") is not False
                        flags.append(st)
                        if st:
                            node_stale[n] += 1
                            age = by[n].get("age_ms")
                            if isinstance(age, (int, float)):
                                node_age[n].append(float(age))
                    if any(flags):
                        any_stale += 1
                    if all(flags):
                        all_stale += 1

    if n_records == 0:
        return None
    return {
        "n_records": n_records, "select_us": select_us,
        "body_us": body_us, "est_us": est_us, "resid_us": resid_us,
        "ratio_per_req": ratio_per_req, "sel_for_ratio": sel_for_ratio,
        "tot_for_ratio": tot_for_ratio,
        "node_stale": node_stale, "node_age": node_age,
        "n_complete": n_complete, "any_stale": any_stale, "all_stale": all_stale,
        "modes": modes, "on_stale": on_stale, "logged_max_age": logged_max_age,
    }


# ───────────────────────── cells and keys ─────────────────────────────────
def enumerate_cells(scans):
    """CELL_COLS order, then any observed cell outside it. Mirrored from
    analyze_divergence.all_cell_rows, which iterates CELL_COLS inline and so
    drops such cells entirely."""
    canonical = [AD.col_label(w, c) for w, c in AD.CELL_COLS]
    extra = sorted({s["cell"] for s in scans} - set(canonical))
    return canonical, extra


def group_keys(scans, canonical, extra):
    present = {(s["scope_clean"], s["arm"], s["cell"]) for s in scans}
    keys = []
    for scope in SCOPES:
        arms = sorted({s["arm"] for s in scans if in_scope(s, scope)})
        for arm in arms:
            for cell in canonical + extra:
                if cell in canonical or any(
                        (c, arm, cell) in present for c in (True, False)):
                    keys.append((scope, arm, cell, cell in canonical))
    return keys


def in_scope(s, scope):
    return scope == "all" or (scope == "clean") == s["scope_clean"]


def runs_for(scans, scope, arm, cell):
    return [s for s in scans if in_scope(s, scope)
            and s["arm"] == arm and s["cell"] == cell]


def base_row(scope, arm, cell, in_cc, runs):
    return {"scope": scope, "arm": arm, "cell": cell, "n_runs": len(runs),
            "in_reference_cell_list": in_cc}


# ───────────────────────── blocks ─────────────────────────────────────────
def build_block1(scans, keys):
    rows = []
    for scope, arm, cell, in_cc in keys:
        runs = runs_for(scans, scope, arm, cell)
        row = base_row(scope, arm, cell, in_cc, runs)
        pooled = [v for s in runs for v in s["select_us"]]
        emit_stat_columns(row, "select_us", [s["select_us"] for s in runs], pooled)
        rows.append(row)
    return rows


def build_block2(scans, keys, decomposed_arms):
    rows = []
    for scope, arm, cell, in_cc in keys:
        if arm not in decomposed_arms:
            continue
        runs = runs_for(scans, scope, arm, cell)
        row = base_row(scope, arm, cell, in_cc, runs)
        for prefix, field in (("select_us", "select_us"), ("body_read_us", "body_us"),
                              ("token_estimate_us", "est_us"), ("residual_us", "resid_us")):
            pooled = [v for s in runs for v in s[field]]
            emit_stat_columns(row, prefix, [s[field] for s in runs], pooled)
        rows.append(row)
    return rows


def build_block3(scans, keys):
    rows = []
    for scope, arm, cell, in_cc in keys:
        runs = runs_for(scans, scope, arm, cell)
        row = base_row(scope, arm, cell, in_cc, runs)

        # Convention A: ratio of medians -- median(select) / median(total).
        per_run_A = []
        for s in runs:
            if s["sel_for_ratio"] and s["tot_for_ratio"]:
                ms = AD.median(s["sel_for_ratio"]) / 1000.0
                mt = AD.median(s["tot_for_ratio"])
                per_run_A.append(ms / mt if mt else None)
        m, sd, n = across_runs(per_run_A)
        row["ratio_of_medians_runmean"] = m
        row["ratio_of_medians_runsd"] = sd
        row["ratio_of_medians_n_runs"] = n
        sel_all = [v for s in runs for v in s["sel_for_ratio"]]
        tot_all = [v for s in runs for v in s["tot_for_ratio"]]
        row["ratio_of_medians_pooled"] = (
            (AD.median(sel_all) / 1000.0) / AD.median(tot_all)
            if sel_all and tot_all and AD.median(tot_all) else None)

        # Convention B: median of per-request ratios.
        per_run_B = [AD.median(s["ratio_per_req"]) for s in runs if s["ratio_per_req"]]
        m, sd, n = across_runs(per_run_B)
        row["median_per_request_ratio_runmean"] = m
        row["median_per_request_ratio_runsd"] = sd
        row["median_per_request_ratio_n_runs"] = n
        pooled_ratios = [v for s in runs for v in s["ratio_per_req"]]
        row["median_per_request_ratio_pooled"] = AD.median(pooled_ratios)
        row["p90_per_request_ratio_pooled"] = AD.pct(pooled_ratios, 90)

        row["select_us_median_pooled"] = AD.median(sel_all)
        row["total_ms_median_pooled"] = AD.median(tot_all)
        row["n_requests_with_total_ms"] = len(pooled_ratios)
        rows.append(row)
    return rows


def build_block4(scans, keys):
    rows = []
    for scope, arm, cell, in_cc in keys:
        runs = runs_for(scans, scope, arm, cell)
        n_complete = sum(s["n_complete"] for s in runs)
        any_s = sum(s["any_stale"] for s in runs)
        all_s = sum(s["all_stale"] for s in runs)
        per_run_any = [s["any_stale"] / s["n_complete"] for s in runs if s["n_complete"]]
        per_run_all = [s["all_stale"] / s["n_complete"] for s in runs if s["n_complete"]]
        am, asd, an = across_runs(per_run_any)
        lm, lsd, ln = across_runs(per_run_all)
        for node in AD.NODES:
            row = base_row(scope, arm, cell, in_cc, runs)
            st = sum(s["node_stale"][node] for s in runs)
            per_run_rate = [s["node_stale"][node] / s["n_complete"]
                            for s in runs if s["n_complete"]]
            rm, rsd, rn = across_runs(per_run_rate)
            ages = [a for s in runs for a in s["node_age"][node]]
            row.update({
                "node": node,
                "decisions_complete_sv": n_complete,
                "stale_count": st,
                "stale_rate_pooled": (st / n_complete) if n_complete else None,
                "stale_rate_runmean": rm, "stale_rate_runsd": rsd,
                "stale_rate_n_runs": rn,
                # age_ms is pooled over decisions on purpose: a median of
                # per-run medians is not the median of the union.
                "age_ms_n": len(ages),
                "age_ms_median_pooled": AD.median(ages),
                "age_ms_p90_pooled": AD.pct(ages, 90),
                "age_ms_max_pooled": max(ages) if ages else None,
                "any_node_stale_count": any_s,
                "any_node_stale_share_pooled": (any_s / n_complete) if n_complete else None,
                "any_node_stale_share_runmean": am, "any_node_stale_share_runsd": asd,
                "all_nodes_stale_count": all_s,
                "all_nodes_stale_share_pooled": (all_s / n_complete) if n_complete else None,
                "all_nodes_stale_share_runmean": lm, "all_nodes_stale_share_runsd": lsd,
                "max_state_age_ms_observed": "|".join(
                    str(v) for v in sorted({v for s in runs for v in s["logged_max_age"]})),
            })
            rows.append(row)
    return rows


def build_block5(scans, keys, modes_sorted):
    rows = []
    for scope, arm, cell, in_cc in keys:
        runs = runs_for(scans, scope, arm, cell)
        row = base_row(scope, arm, cell, in_cc, runs)
        total = sum(s["n_records"] for s in runs)
        row["n_requests"] = total
        for mode in modes_sorted:
            c = sum(s["modes"].get(mode, 0) for s in runs)
            row[f"mode_{mode}_count"] = c
            row[f"mode_{mode}_share_pooled"] = (c / total) if total else None
            per_run = [s["modes"].get(mode, 0) / s["n_records"]
                       for s in runs if s["n_records"]]
            m, sd, _ = across_runs(per_run)
            row[f"mode_{mode}_share_runmean"] = m
            row[f"mode_{mode}_share_runsd"] = sd
        d = sum(s["on_stale"] for s in runs)
        row["decision_on_stale_state_count"] = d
        row["decision_on_stale_state_share_pooled"] = (d / total) if total else None
        per_run = [s["on_stale"] / s["n_records"] for s in runs if s["n_records"]]
        m, sd, _ = across_runs(per_run)
        row["decision_on_stale_state_share_runmean"] = m
        row["decision_on_stale_state_share_runsd"] = sd
        rows.append(row)
    return rows


# ───────────────────────── output ─────────────────────────────────────────
def write_csv(path, rows):
    if not rows:
        return 0, 0
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return len(rows), len(rows[0])


def write_readme(outdir, manifest, cfg, decomposed_arms, extra_cells, modes_sorted):
    L = [
        "# SQ2 result-section artefacts",
        "",
        "Generated by `analyze_sq2.py` from the router JSONL logs.",
        "Do not edit by hand -- rerun the script. Analysis only; no plots.",
        "",
        "## Warm-up exclusion — changed 2026-08-24",
        "",
        "**These CSVs exclude warm-up requests.** Methodology: *\"Each run issues three "
        "warm-up requests before the 100 measured requests; warm-up is excluded from "
        "every statistic, on both the client and the router side.\"* Until 2026-08-24 "
        "every SQ2 block counted all 103 router records per run.",
        "",
        "The predicate is the one `analyze_3e.py` already applies "
        "(`isinstance(request_id, int) and request_id >= num_warmup`), mirrored because "
        "it sits inline in that script's per-run loop. `num_warmup` is read per run from "
        "`run_meta.bench_summary` via `analyze_sq1.resolve_num_warmup`, not assumed, and "
        "any run where it is not 3 is flagged at runtime.",
        "",
        "Warm-up requests hit an idle cluster, so they are not a random sample of the "
        "run: the script therefore also scans them in isolation and prints their own "
        "select_us, latency-share and staleness figures beside the before/after "
        "columns, rather than leaving the difference to be inferred from the aggregate "
        "shift.",
        "",
        "## Which block the chapter should quote",
        "",
        "Every file carries a `scope` column with three values: `all`, `clean`, "
        "`not_clean`.",
        "",
        "| block | quote scope | why |",
        "|---|---|---|",
        "| 1 routing overhead | **all** | methodology Section 6: overhead is reported "
        "over all runs, including those excluded from the latency analysis, because "
        "selection cost depends on the router rather than the offered load |",
        "| 2 decomposition | **all** | same rule as block 1 |",
        "| 3 share of latency | **clean** | the denominator is end-to-end latency, which "
        "the latency-analysis exclusions do apply to |",
        "| 4 staleness | **clean** | describes the population the latency results speak to |",
        "| 5 stale handling | **clean** | same as block 4 |",
        "",
        "`clean` and `not_clean` are emitted for every block regardless, so the "
        "sensitivity of any figure to that choice can be read directly.",
        "",
        "## Files",
        "",
        "| file | rows | cols | SQ2 subsection it supports |",
        "|---|---:|---:|---|",
    ]
    for b in sorted(manifest):
        fn, desc, nr, nc = manifest[b]
        L.append(f"| `{fn}` | {nr} | {nc} | {desc} |")
    L += [
        "",
        "## Aggregation conventions",
        "",
        "- **Grouping is `(scope, arm, cell)`, never policy name.** The `wlc_conc` arm "
        "logs `policy == \"weighted_least_connections\"`; grouping on the policy string "
        "would merge it into the weighted-least-connections baseline.",
        "- `*_runmean` / `*_runsd`: the statistic is computed **per run**, then averaged "
        "across runs. The run is the unit of replication. SD is the **sample** SD, "
        "**ddof = 1**, from `analyze_divergence.agg`, which returns 0.0 when only one "
        "run contributes -- that is undefined, not zero variance, so `*_n_runs` is "
        "carried alongside.",
        "- `*_pooled`: the same statistic over all requests pooled, ignoring run "
        "boundaries. Reported beside the run-level figure so the two can be compared "
        "rather than conflated.",
        "- `age_ms` in block 4 is **pooled over decisions only**. A median of per-run "
        "medians is not the median of the union, so no run-level median is offered for "
        "it.",
        "- Percentiles are `analyze_divergence.pct`, linearly interpolated.",
        "- Units: `select_us`, `body_read_us`, `token_estimate_us`, `residual_us` are "
        "microseconds (the log stores nanoseconds; converted here). `total_ms` and "
        "`age_ms` are milliseconds.",
        "- **No statistical test is applied anywhere.** Every figure is descriptive.",
        "",
        "## Block 3: the ratio is reported two ways, deliberately",
        "",
        "`ratio_of_medians_*` divides median(select) by median(total_ms). "
        "`median_per_request_ratio_*` takes the ratio within each request and then the "
        "median of those. They are different quantities and can disagree; both are "
        "emitted so the chapter states which it means.",
        "",
        "## Observed configuration (read from run_meta, not assumed)",
        "",
        "| field | value(s) | runs |",
        "|---|---|---:|",
    ]
    for field, counts in cfg.items():
        for val, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            L.append(f"| `{field}` | {val} | {n} |")
    L += [
        "",
        "## Scope notes",
        "",
        f"- Arms whose timer also covers body read and token estimation, detected by the "
        f"presence of those sub-timers rather than by name: "
        f"{', '.join('`' + a + '`' for a in decomposed_arms) or 'none'}. Block 2 covers "
        f"only these.",
        f"- `stale_mode` values observed: "
        f"{', '.join('`' + str(m) + '`' for m in modes_sorted)}.",
    ]
    if extra_cells:
        L.append(f"- Cells outside `analyze_divergence.CELL_COLS`, carried here and "
                 f"flagged `in_reference_cell_list=False` rather than dropped: "
                 f"{', '.join('`' + c + '`' for c in extra_cells)}.")
    L.append("")
    p = os.path.join(outdir, "README.md")
    open(p, "w").write("\n".join(L) + "\n")
    return p


def print_conventions(decomposed_arms, modes_sorted):
    print("\n" + "#" * 100)
    print("# CONVENTIONS (printed at runtime, not only in comments)")
    print("#" * 100)
    print("1. Grouping is (scope, arm, cell). NEVER policy name: the wlc_conc arm logs")
    print("   policy == 'weighted_least_connections' and would otherwise be pooled with")
    print("   the weighted-least-connections baseline.")
    print("2. *_runmean / *_runsd: statistic computed PER RUN, then averaged across runs")
    print("   (the run is the unit of replication). SD is the SAMPLE SD, ddof = 1, via")
    print("   analyze_divergence.agg; it returns 0.0 for a single run, which is undefined")
    print("   rather than zero variance, so *_n_runs is carried beside it.")
    print("3. *_pooled: the same statistic over all requests pooled, ignoring run")
    print("   boundaries. Emitted beside the run-level figure, never instead of it.")
    print("4. Block 3 reports BOTH ratio-of-medians and median-of-per-request-ratios.")
    print("   They are different quantities; neither is chosen silently.")
    print("5. Block 4's age_ms is pooled over decisions only -- a median of per-run")
    print("   medians is not the median of the union, so no run-level median is offered.")
    print("6. Percentiles are analyze_divergence.pct (linear interpolation).")
    print("7. Units: select/body/estimate/residual in MICROSECONDS (log stores ns);")
    print("   total_ms and age_ms in milliseconds.")
    print(f"8. Block 2 covers arms detected by the PRESENCE of the sub-timers, not by")
    print(f"   name: {sorted(decomposed_arms)}")
    print(f"9. stale_mode values observed: {modes_sorted}")
    print("10. NO statistical test is applied. Everything here is descriptive.")


def main():
    ap = argparse.ArgumentParser(description="SQ2 result-section artefacts (read-only).")
    ap.add_argument("--outdir", default=HERE)
    args = ap.parse_args()
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    print("=" * 100)
    print("SQ2 — routing overhead and metric collection. Read-only on all inputs.")
    print("=" * 100)

    runs = AD.load_all_runs()
    usable = [r for r in runs if r.get("usable")]
    dropped = [r for r in runs if not r.get("usable")]
    print(f"{len(runs)} run directories, {len(usable)} usable "
          f"(selection imported from analyze_divergence.load_all_runs).")
    if dropped:
        print(f"*** {len(dropped)} run(s) dropped by that selection and therefore absent "
              f"from SQ2, even though overhead does not depend on analysability:")
        for r in dropped:
            print(f"    {r['dir']}: {r.get('reason')}")

    cfg = {"poll_interval_ms": Counter(), "poll_timeout_ms": Counter(),
           "max_state_age_ms": Counter()}
    # Three passes: measured requests (written), all requests (before/after comparison), warm-ups only.
    warmup_val, warmup_src = Counter(), Counter()
    scans, scans_all, scans_wu = [], [], []
    for r in usable:
        meta = json.loads(open(os.path.join(r["dir"], "run_meta.json")).read())
        fc = meta.get("frozen_config") or {}
        for k in cfg:
            cfg[k][fc.get(k)] += 1
        nw, src = resolve_num_warmup(meta, r["dir"])
        warmup_val[nw] += 1
        warmup_src[src] += 1
        log = os.path.join(r["dir"], meta["router_log_file"])
        s_all = scan(meta, log, nw, "all")
        s_wu = scan(meta, log, nw, "warmup")
        for extra, bucket in ((s_all, scans_all), (s_wu, scans_wu)):
            if extra is not None:
                extra.update({"scope_clean": r.get("verdict") == "clean",
                              "arm": r["arm"],
                              "cell": AD.col_label(r["workload"], r["conc"]),
                              "dir": r["dir"], "num_warmup": nw})
                bucket.append(extra)
        s = scan(meta, log, nw, "measured")
        if s is None:
            continue
        s.update({"scope_clean": r.get("verdict") == "clean", "arm": r["arm"],
                  "cell": AD.col_label(r["workload"], r["conc"]), "dir": r["dir"],
                  "num_warmup": nw})
        scans.append(s)

    print("\nWarm-up exclusion ACTIVE (methodology: warm-up excluded from every "
          "statistic, router side included).")
    print(f"  num_warmup values: {dict(warmup_val)}")
    print(f"  resolved from:     {dict(warmup_src)}")
    off = [s for s in scans if s["num_warmup"] != 3]
    if off:
        print(f"  *** {len(off)} run(s) where num_warmup is NOT 3:")
        for s in off:
            print(f"      {s['dir']}: num_warmup={s['num_warmup']}")
    else:
        print("  all runs have num_warmup == 3")
    n_before = sum(s["n_records"] for s in scans_all)
    n_after = sum(s["n_records"] for s in scans)
    n_wu = sum(s["n_records"] for s in scans_wu)
    print(f"  requests: {n_before} before, {n_after} after, {n_wu} warm-up removed "
          f"({'consistent' if n_before - n_after == n_wu else 'INCONSISTENT'})")

    print("\nConfiguration read from run_meta.frozen_config:")
    for field, counts in cfg.items():
        vals = sorted(counts.items(), key=lambda kv: -kv[1])
        shown = ", ".join(f"{v} ({n} runs)" for v, n in vals)
        print(f"  {field:<20} {shown}")
        if len(vals) > 1:
            modal = vals[0][0]
            print(f"    *** NOT UNIFORM. Runs differing from the modal value {modal}:")
            for r, m in zip(usable, scans):
                fc = json.loads(open(os.path.join(r["dir"], "run_meta.json")).read()
                                ).get("frozen_config") or {}
                if fc.get(field) != modal:
                    print(f"      {r['dir']}: {field}={fc.get(field)}")
    logged = sorted({v for s in scans for v in s["logged_max_age"]})
    print(f"  max_state_age_ms as logged per decision: {logged}")

    canonical, extra = enumerate_cells(scans)
    keys = group_keys(scans, canonical, extra)
    decomposed_arms = sorted({s["arm"] for s in scans if s["body_us"]})
    modes_sorted = sorted(m for m in STALE_MODES_SEEN if m is not None)

    print_conventions(decomposed_arms, modes_sorted)

    if extra:
        for c in extra:
            rs = [s for s in scans if s["cell"] == c]
            print(f"\nCell outside CELL_COLS carried here: {c} — {len(rs)} run(s), "
                  f"{sum(s['n_records'] for s in rs)} requests, "
                  f"arms={sorted({s['arm'] for s in rs})}")

    # ---------------- warm-up: before/after, and what warm-up itself looks like ----
    def sel_stats(src, arm=None, clean_only=False):
        rs = [s for s in src if (arm is None or s["arm"] == arm)
              and (not clean_only or s["scope_clean"])]
        return stats_of([v for s in rs for v in s["select_us"]])

    print("\n" + "#" * 100)
    print("# WARM-UP — BEFORE (all 103) vs AFTER (100 measured), and the warm-up "
          "requests alone")
    print("#" * 100)
    print("Block 1, select_us per arm, pooled over cells, scope=all (microseconds)")
    print(f"{'arm':<28}{'n bef':>7}{'n aft':>7}{'n wu':>6}"
          f"{'med bef':>9}{'med aft':>9}{'med WU':>9}"
          f"{'p90 bef':>9}{'p90 aft':>9}{'p90 WU':>9}"
          f"{'p99 bef':>9}{'p99 aft':>9}{'p99 WU':>9}")
    for arm in sorted({s["arm"] for s in scans}):
        b, a, w = (sel_stats(src, arm) for src in (scans_all, scans, scans_wu))
        if not b["n"]:
            continue
        g = lambda d, k: (f"{d[k]:.2f}" if d[k] is not None else "-")
        print(f"{arm:<28}{b['n']:>7}{a['n']:>7}{w['n']:>6}"
              f"{g(b,'median'):>9}{g(a,'median'):>9}{g(w,'median'):>9}"
              f"{g(b,'p90'):>9}{g(a,'p90'):>9}{g(w,'p90'):>9}"
              f"{g(b,'p99'):>9}{g(a,'p99'):>9}{g(w,'p99'):>9}")

    print("\nBlock 2, cost_predict_v3 decomposition (microseconds, medians)")
    print(f"{'component':<20}{'before':>10}{'after':>10}{'warm-up':>10}")
    for label, field in (("select_us", "select_us"), ("body_read_us", "body_us"),
                         ("token_estimate_us", "est_us"), ("residual_us", "resid_us")):
        vals = []
        for src in (scans_all, scans, scans_wu):
            v = [x for s in src if s["arm"] in decomposed_arms for x in s[field]]
            vals.append(AD.median(v) if v else None)
        g = lambda x: (f"{x:.2f}" if x is not None else "-")
        print(f"{label:<20}{g(vals[0]):>10}{g(vals[1]):>10}{g(vals[2]):>10}")

    print("\nBlock 3, overhead share of end-to-end latency, scope=clean "
          "(ratio of medians, ppm)")
    print(f"{'arm':<28}{'before':>10}{'after':>10}{'warm-up':>10}"
          f"{'tot ms bef':>12}{'tot ms aft':>12}{'tot ms WU':>12}")
    for arm in sorted({s["arm"] for s in scans}):
        out = []
        for src in (scans_all, scans, scans_wu):
            rs = [s for s in src if s["arm"] == arm and s["scope_clean"]]
            sel = [v for s in rs for v in s["sel_for_ratio"]]
            tot = [v for s in rs for v in s["tot_for_ratio"]]
            out.append(((AD.median(sel) / 1000.0) / AD.median(tot) * 1e6, AD.median(tot))
                       if sel and tot and AD.median(tot) else (None, None))
        g = lambda x, nd=3: (f"{x:.{nd}f}" if x is not None else "-")
        print(f"{arm:<28}{g(out[0][0]):>10}{g(out[1][0]):>10}{g(out[2][0]):>10}"
              f"{g(out[0][1],1):>12}{g(out[1][1],1):>12}{g(out[2][1],1):>12}")

    print("\nBlock 4, staleness per node, scope=clean")
    print(f"{'node':<12}{'rate bef':>10}{'rate aft':>10}{'rate WU':>9}"
          f"{'age med bef':>13}{'age med aft':>13}{'age med WU':>12}")
    for node in AD.NODES:
        cells = []
        for src in (scans_all, scans, scans_wu):
            cl = [s for s in src if s["scope_clean"]]
            ncp = sum(s["n_complete"] for s in cl)
            st = sum(s["node_stale"][node] for s in cl)
            ages = [a for s in cl for a in s["node_age"][node]]
            cells.append(((st / ncp) if ncp else None, AD.median(ages)))
        g = lambda x, nd=3: (f"{x:.{nd}f}" if x is not None else "-")
        print(f"{node:<12}{g(cells[0][0]):>10}{g(cells[1][0]):>10}{g(cells[2][0]):>9}"
              f"{g(cells[0][1],1):>13}{g(cells[1][1],1):>13}{g(cells[2][1],1):>12}")

    print("\nBlock 5, stale_mode shares, scope=clean (all arms pooled)")
    print(f"{'stale_mode':<24}{'before':>10}{'after':>10}{'warm-up':>10}")
    for m in modes_sorted:
        out = []
        for src in (scans_all, scans, scans_wu):
            cl = [s for s in src if s["scope_clean"]]
            t = sum(s["n_records"] for s in cl)
            out.append((sum(s["modes"].get(m, 0) for s in cl) / t) if t else None)
        g = lambda x: (f"{x:.4f}" if x is not None else "-")
        print(f"{m:<24}{g(out[0]):>10}{g(out[1]):>10}{g(out[2]):>10}")
    for label, key in (("decision_on_stale_state", "on_stale"),):
        out = []
        for src in (scans_all, scans, scans_wu):
            cl = [s for s in src if s["scope_clean"]]
            t = sum(s["n_records"] for s in cl)
            out.append((sum(s[key] for s in cl) / t) if t else None)
        g = lambda x: (f"{x:.4f}" if x is not None else "-")
        print(f"{label:<24}{g(out[0]):>10}{g(out[1]):>10}{g(out[2]):>10}")
    print("  'WU' columns describe the EXCLUDED warm-up requests themselves, so a")
    print("  systematic difference is visible directly rather than inferred from the")
    print("  size of the before/after shift.")

    blocks = {
        1: build_block1(scans, keys),
        2: build_block2(scans, keys, decomposed_arms),
        3: build_block3(scans, keys),
        4: build_block4(scans, keys),
        5: build_block5(scans, keys, modes_sorted),
    }
    manifest = {}
    for b, rows in blocks.items():
        fn, desc = BLOCK_FILES[b]
        nr, nc = write_csv(os.path.join(outdir, fn), rows)
        manifest[b] = (fn, desc, nr, nc)

    # ---- stdout summary: scope=all for overhead, scope=clean for freshness ----
    print("\n" + "#" * 100)
    print("# BLOCK 1 — ROUTING OVERHEAD, scope=all, per arm (cells pooled for this view)")
    print("#" * 100)
    print(f"{'arm':<28}{'runs':>6}{'reqs':>8}"
          f"{'med run':>10}{'±sd':>8}{'med pool':>10}"
          f"{'p90 run':>10}{'p90 pool':>10}{'p99 pool':>10}")
    for arm in sorted({s["arm"] for s in scans}):
        rs = [s for s in scans if s["arm"] == arm]
        pooled = [v for s in rs for v in s["select_us"]]
        if not pooled:
            continue
        med_runs = [AD.median(s["select_us"]) for s in rs if s["select_us"]]
        p90_runs = [AD.pct(s["select_us"], 90) for s in rs if s["select_us"]]
        m, sd, _ = across_runs(med_runs)
        pm, _, _ = across_runs(p90_runs)
        st = stats_of(pooled)
        print(f"{arm:<28}{len(rs):>6}{len(pooled):>8}"
              f"{m:>10.2f}{(sd if sd is not None else float('nan')):>8.2f}"
              f"{st['median']:>10.2f}{pm:>10.2f}{st['p90']:>10.2f}{st['p99']:>10.2f}")
    print("  microseconds. 'run' = per-run statistic averaged across runs (ddof=1 SD);")
    print("  'pool' = over all requests pooled.")

    print("\n" + "#" * 100)
    print("# BLOCK 2 — DECOMPOSITION, scope=all, per arm (cells pooled for this view)")
    print("#" * 100)
    for arm in decomposed_arms:
        rs = [s for s in scans if s["arm"] == arm]
        print(f"{arm}:")
        print(f"    {'component':<20}{'n':>8}{'median':>10}{'p90':>10}{'p99':>10}")
        for label, field in (("select_us", "select_us"), ("body_read_us", "body_us"),
                             ("token_estimate_us", "est_us"), ("residual_us", "resid_us")):
            v = [x for s in rs for x in s[field]]
            st = stats_of(v)
            print(f"    {label:<20}{st['n']:>8}{st['median']:>10.2f}"
                  f"{st['p90']:>10.2f}{st['p99']:>10.2f}")

    print("\n" + "#" * 100)
    print("# BLOCK 3 — OVERHEAD AS SHARE OF END-TO-END LATENCY, scope=clean, per arm")
    print("#" * 100)
    print(f"{'arm':<28}{'reqs':>8}{'sel med us':>12}{'total med ms':>14}"
          f"{'ratio of med':>15}{'  = ppm':>10}{'med of ratios':>16}{'  = ppm':>10}")
    for arm in sorted({s["arm"] for s in scans}):
        rs = [s for s in scans if s["arm"] == arm and s["scope_clean"]]
        sel = [v for s in rs for v in s["sel_for_ratio"]]
        tot = [v for s in rs for v in s["tot_for_ratio"]]
        rat = [v for s in rs for v in s["ratio_per_req"]]
        if not sel:
            continue
        rom = (AD.median(sel) / 1000.0) / AD.median(tot)
        mor = AD.median(rat)
        print(f"{arm:<28}{len(sel):>8}{AD.median(sel):>12.2f}{AD.median(tot):>14.1f}"
              f"{rom:>15.3e}{rom*1e6:>10.3f}{mor:>16.3e}{mor*1e6:>10.3f}")
    print("  'ratio of med' = median(select)/median(total). 'med of ratios' = median of")
    print("  the per-request ratio. Both dimensionless; ppm shown only because the raw")
    print("  ratio is ~1e-6 and rounds to zero at fixed precision. CSV holds the raw ratio.")

    print("\n" + "#" * 100)
    print("# BLOCK 4 — STALENESS, scope=clean, per node (arms and cells pooled)")
    print("#" * 100)
    print(f"{'node':<12}{'decisions':>11}{'stale':>9}{'rate':>8}"
          f"{'age med':>10}{'age p90':>10}{'age max':>11}")
    cl = [s for s in scans if s["scope_clean"]]
    nc = sum(s["n_complete"] for s in cl)
    for node in AD.NODES:
        st = sum(s["node_stale"][node] for s in cl)
        ages = [a for s in cl for a in s["node_age"][node]]
        print(f"{node:<12}{nc:>11}{st:>9}{(st/nc if nc else 0):>8.3f}"
              f"{AD.median(ages):>10.1f}{AD.pct(ages,90):>10.1f}"
              f"{(max(ages) if ages else 0):>11.1f}")
    a_s = sum(s["any_stale"] for s in cl)
    l_s = sum(s["all_stale"] for s in cl)
    print(f"  >=1 node stale: {a_s}/{nc} = {a_s/nc:.3f}   "
          f"all nodes stale: {l_s}/{nc} = {l_s/nc:.3f}")
    print(f"  max_state_age_ms observed: {logged}")

    print("\n" + "#" * 100)
    print("# BLOCK 5 — STALE HANDLING, scope=clean, per arm")
    print("#" * 100)
    hdr = "".join(f"{m[:17]:>19}" for m in modes_sorted)
    print(f"{'arm':<28}{'reqs':>8}{hdr}{'on_stale':>10}")
    for arm in sorted({s["arm"] for s in scans}):
        rs = [s for s in scans if s["arm"] == arm and s["scope_clean"]]
        tot = sum(s["n_records"] for s in rs)
        if not tot:
            continue
        cells = "".join(
            f"{sum(s['modes'].get(m,0) for s in rs)/tot:>19.3f}" for m in modes_sorted)
        d = sum(s["on_stale"] for s in rs)
        print(f"{arm:<28}{tot:>8}{cells}{d/tot:>10.3f}")
    print("  shares of that arm's clean requests, pooled; per-run means and SDs are in "
          "the CSV.")

    readme = write_readme(outdir, manifest, cfg, decomposed_arms, extra, modes_sorted)
    print(f"\nWritten to {outdir}:")
    for b in sorted(manifest):
        fn, _, nr, ncol = manifest[b]
        print(f"  block {b}: {fn}  ({nr} rows, {ncol} cols)")
    print(f"  {os.path.basename(readme)}")

    print("\n" + "#" * 100)
    print("# WHERE A TEST WOULD BE NEEDED (none applied here, as instructed)")
    print("#" * 100)
    print("Any claim that one arm's selection cost DIFFERS from another's needs a test;")
    print("the run-level SDs in block 1 are descriptive spread, not evidence of")
    print("separation, and the n per arm ranges over the arms unevenly. The same applies")
    print("to comparing stale rates across nodes or cells in block 4. Nothing in these")
    print("CSVs supports a difference claim on its own.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
