"""Produces every number in the routing-decision results (sub-question 1) in one pass."""
import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict

import analyze_divergence as AD
from analyze_skip_bias import COUNT_KEYS, TOL, classify, node_eligible, node_stale

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # data root: 3e_results
OUT_HERE = os.path.dirname(os.path.abspath(__file__))  # outputs: everythingcombined
REFERENCE_CSV = os.path.join(OUT_HERE, "divergence_analysis.csv")

BLOCK_FILES = {
    1: ("sq1_divergence.csv", "SQ1 divergence result: three counterfactual pairs — B vs B/w (isolates capacity knowledge), B/w vs R/w (isolates state-awareness), B vs R/w (the frozen headline, carried as `divergence_rate_router_*`) — with tie rates and a per-pair magnitude."),
    2: ("sq1_skip_accounting.csv", "SQ1 exclusion validity: what was excluded, why, and whether relaxing the staleness filter moves the result."),
    3: ("sq1_b_metrics_distribution.csv", "SQ1 signal saturation: how often b_metrics is pinned at the -np ceiling."),
    4: ("sq1_tie_composition.csv", "SQ1 tie structure: which nodes tie at minimum b, at what value, and how far apart their u is."),
}


# ───────────────────────── -np, read from the launch command ──────────────
def detect_np(driver_path):
    """Parse the configured slot count out of the driver's llama-server launch
    command. Never assumed: if the driver does not state it unambiguously this
    fails rather than guessing, because the ceiling in block 3 depends on it."""
    if not os.path.exists(driver_path):
        raise SystemExit(f"FATAL: driver not found at {driver_path}; cannot read -np. "
                         f"Pass --driver.")
    text = open(driver_path).read()
    hits = []
    for pat in (r"-np\s+(\d+)", r"--parallel[\s\"'=]+(\d+)"):
        for m in re.finditer(pat, text):
            line = text[:m.start()].count("\n") + 1
            hits.append((int(m.group(1)), line, m.group(0).strip()))
    if not hits:
        raise SystemExit(f"FATAL: no -np/--parallel found in {driver_path}; refusing to "
                         f"assume a slot count.")
    values = {h[0] for h in hits}
    if len(values) > 1:
        raise SystemExit(f"FATAL: conflicting slot counts in {driver_path}: "
                         + "; ".join(f"{v} (line {ln}, {txt!r})" for v, ln, txt in hits))
    return hits[0][0], hits


# ───────────────────────── one pass per run ───────────────────────────────
def resolve_num_warmup(meta, run_dir):
    """Warm-up count for this run. Preferred source is run_meta.bench_summary;
    falls back to the bench JSON, then to the same default analyze_3e.py uses.
    Returned with its source so a run that differs can be flagged."""
    bs = meta.get("bench_summary") or {}
    if isinstance(bs.get("num_warmup"), int):
        return bs["num_warmup"], "run_meta.bench_summary"
    bf = meta.get("bench_result_file")
    if bf:
        p = os.path.join(run_dir, bf)
        if os.path.exists(p):
            b = json.loads(open(p).read())
            if isinstance(b.get("num_warmup"), int):
                return b["num_warmup"], "bench JSON"
    return 3, "default (analyze_3e.py's b.get('num_warmup', 3))"


def scan(meta, log_path, np_ceiling, num_warmup):
    """One pass over one run's log, collecting every block's raw material.

    Mirrors analyze_divergence.analyze_run's check order exactly:
      state_vector missing/not a list -> incomplete node set -> eligibility and
      staleness gate -> undefined argmin -> analysed.

    WARM-UP EXCLUSION (added 2026-08-24). Methodology: "Each run issues three
    warm-up requests before the 100 measured requests; warm-up is excluded from
    every statistic, on both the client and the router side." The predicate is
    the one analyze_3e.py already uses at its line 224-225 --
        isinstance(request.get("request_id"), int) and request["request_id"] >= num_warmup
    -- mirrored rather than imported because it sits inline in that script's own
    per-run loop. Pass num_warmup=0 to keep warm-up in, which is what the
    reproduction check against divergence_analysis.csv needs, since that frozen
    CSV was produced without the filter.
    """
    weights = (meta.get("frozen_config") or {}).get("weights")
    if not weights or len(weights) != 3:
        return None
    weight_by_node = dict(zip(AD.NODES, weights))

    counts = Counter()
    # div_router is the B vs R/w pair (name kept to match divergence_analysis.csv); the _vs_ keys are added pairs.
    strict = {"div_router": [], "div_unc": [], "tie_router": [], "tie_unc": [],
              "div_b_vs_bw": [], "div_bw_vs_rw": []}
    relaxed = {"div_router": [], "div_unc": [], "tie_router": [], "tie_unc": []}
    mag_router, mag_unc = [], []
    mag_b_vs_bw, mag_bw_vs_rw = [], []
    bdist = {n: Counter() for n in AD.NODES}          # block 3
    n_complete = 0
    all_at_ceiling = 0
    ties = defaultdict(lambda: {"count": 0, "du": []})  # block 4: (tied_set, value)
    n_ties = 0

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
            # Warm-up exclusion, before any counter is touched so warm-up
            # decisions land in neither the analysed nor the skipped tally.
            rid = r.get("request_id")
            if num_warmup and not (isinstance(rid, int) and rid >= num_warmup):
                counts["warmup_excluded"] += 1
                continue

            sv = r.get("state_vector")
            if not sv or not isinstance(sv, list):
                counts["skip_malformed_sv"] += 1
                continue
            by_name = {e.get("replica"): e for e in sv}
            if not all(n in by_name for n in AD.NODES):
                counts["skip_incomplete_sv"] += 1
                continue

            # ---- block 3: every complete state vector, analysed or not ----
            n_complete += 1
            bvals = {}
            for n in AD.NODES:
                b = by_name[n].get("b_metrics")
                bdist[n][b] += 1
                bvals[n] = b
            if all(v == float(np_ceiling) for v in bvals.values()):
                all_at_ceiling += 1

            gate_fail, _, _ = classify(by_name)

            # ---- scores, built once and shared by both inclusion rules ----
            b_score = {n: by_name[n].get("b_metrics") for n in AD.NODES}
            deferred_score = {
                n: (by_name[n].get("b_metrics") or 0.0) + (by_name[n].get("requests_deferred") or 0.0)
                for n in AD.NODES
            }
            u_score = {n: by_name[n].get("u") for n in AD.NODES}
            rw_score = {}
            # Weighted least-connections: B_r / w_r, from the SAME per-run
            # frozen_config weight vector R/w already uses. Same None/zero guard.
            bw_score = {}
            for n in AD.NODES:
                u, w = u_score[n], weight_by_node.get(n)
                rw_score[n] = None if (u is None or w is None or w == 0) else u / w
                b = b_score[n]
                bw_score[n] = None if (b is None or w is None or w == 0) else b / w

            all_idx = list(range(3))
            n_eligible = sum(1 for n in AD.NODES if node_eligible(by_name[n]))

            for label, included, acc in (
                ("strict", gate_fail is None, strict),
                ("relaxed", n_eligible == 3, relaxed),
            ):
                if not included:
                    continue
                cand_b = AD.argmin_over(all_idx, lambda i: b_score[AD.NODES[i]])
                cand_bu = AD.argmin_over(all_idx, lambda i: deferred_score[AD.NODES[i]])
                cand_rw = AD.argmin_over(all_idx, lambda i: rw_score[AD.NODES[i]])
                # cand_bw is not part of the skip condition, so the analysed decisions match analyze_divergence.py.
                cand_bw = AD.argmin_over(all_idx, lambda i: bw_score[AD.NODES[i]])
                if cand_b is None or cand_bu is None or cand_rw is None:
                    if label == "strict":
                        counts["skip_undefined_argmin"] += 1
                    continue
                if label == "strict":
                    counts["analysed"] += 1

                min_b = min(b_score[n] for n in AD.NODES)
                tied = [i for i in all_idx if b_score[AD.NODES[i]] == min_b]
                acc["tie_router"].append(1 if len(tied) >= 2 else 0)
                min_bu = min(deferred_score[n] for n in AD.NODES)
                acc["tie_unc"].append(
                    1 if sum(1 for n in AD.NODES if deferred_score[n] == min_bu) >= 2 else 0)

                d_router = 1 if cand_b != cand_rw else 0
                d_unc = 1 if cand_bu != cand_rw else 0
                acc["div_router"].append(d_router)
                acc["div_unc"].append(d_unc)

                if label == "strict":
                    # magnitude rule, mirrored from analyze_divergence.py
                    if d_router:
                        a, b = rw_score[AD.NODES[cand_b]], rw_score[AD.NODES[cand_rw]]
                        if a is not None and b is not None and b > 0:
                            mag_router.append(a / b)
                    if d_unc:
                        a, b = rw_score[AD.NODES[cand_bu]], rw_score[AD.NODES[cand_rw]]
                        if a is not None and b is not None and b > 0:
                            mag_unc.append(a / b)

                    # Added rule pairs: magnitude uses the pair's second metric (e.g. B vs B/w in B/w units).
                    if cand_bw is not None:
                        d_b_bw = 1 if cand_b != cand_bw else 0
                        d_bw_rw = 1 if cand_bw != cand_rw else 0
                        acc["div_b_vs_bw"].append(d_b_bw)
                        acc["div_bw_vs_rw"].append(d_bw_rw)
                        if d_b_bw:
                            a, b = bw_score[AD.NODES[cand_b]], bw_score[AD.NODES[cand_bw]]
                            if a is not None and b is not None and b > 0:
                                mag_b_vs_bw.append(a / b)
                        if d_bw_rw:
                            a, b = rw_score[AD.NODES[cand_bw]], rw_score[AD.NODES[cand_rw]]
                            if a is not None and b is not None and b > 0:
                                mag_bw_vs_rw.append(a / b)

                    # ---- block 4: composition of this tie ----
                    if len(tied) >= 2:
                        n_ties += 1
                        key = ("{" + ",".join(str(i + 1) for i in tied) + "}", min_b)
                        ties[key]["count"] += 1
                        if len(tied) == 2:
                            ua, ub = u_score[AD.NODES[tied[0]]], u_score[AD.NODES[tied[1]]]
                            if ua is not None and ub is not None:
                                ties[key]["du"].append(abs(ua - ub))

            if gate_fail is not None:
                counts[gate_fail] += 1

    total = sum(counts[k] for k in COUNT_KEYS)
    if total == 0:
        return None
    return {
        "counts": dict(counts), "total": total,
        "strict": strict, "relaxed": relaxed,
        "mag_median_router": AD.median(mag_router), "mag_median_unc": AD.median(mag_unc),
        "mag_median_b_vs_bw": AD.median(mag_b_vs_bw),
        "mag_median_bw_vs_rw": AD.median(mag_bw_vs_rw),
        "bdist": bdist, "n_complete": n_complete, "all_at_ceiling": all_at_ceiling,
        "ties": {k: v for k, v in ties.items()}, "n_ties": n_ties,
    }


# ───────────────────────── cell enumeration ───────────────────────────────
def enumerate_cells(scans):
    """analyze_divergence.CELL_COLS in its canonical order, plus any cell the
    data contains that is NOT in that list. The extras (e.g. c4_sgpt) are
    dropped by analyze_divergence.all_cell_rows, which iterates CELL_COLS; they
    are carried here and flagged, never silently discarded."""
    canonical = [AD.col_label(w, c) for w, c in AD.CELL_COLS]
    observed = {s["cell"] for s in scans}
    extra = sorted(observed - set(canonical))
    return canonical, extra


def rate(vals):
    return (sum(vals) / len(vals)) if vals else None


def cell_runs(scans, clean, arm, cell):
    return [s for s in scans if s["clean"] is clean and s["arm"] == arm and s["cell"] == cell]


def agg_pair(runs, which, key):
    a = AD.agg([rate(r[which][key]) for r in runs if r[which][key]])
    return a["mean"], (a["std"] if a["n"] else None)


# ───────────────────────── block builders ─────────────────────────────────
def build_block1(scans, keys):
    rows = []
    for clean, arm, cell, in_cc in keys:
        runs = cell_runs(scans, clean, arm, cell)
        row = {"clean": clean, "arm": arm, "cell": cell, "n_runs": len(runs),
               "n_analyzed": sum(r["counts"].get("analysed", 0) for r in runs),
               "n_skipped": sum(r["total"] - r["counts"].get("analysed", 0) for r in runs)}
        for key, name in (("div_router", "divergence_rate_router"),
                          ("div_unc", "divergence_rate_uncensored"),
                          ("tie_router", "b_tie_rate_router"),
                          ("tie_unc", "b_tie_rate_uncensored")):
            m, s = agg_pair(runs, "strict", key)
            row[name + "_mean"], row[name + "_std"] = m, s
        for src, name in (("mag_median_router", "magnitude_median_router"),
                          ("mag_median_unc", "magnitude_median_uncensored")):
            vals = [r[src] for r in runs if r[src] is not None]
            row[name] = AD.median(vals)
            row[name.replace("median", "p90")] = AD.pct(vals, 90)

        # Added rule pairs; B vs R/w is reported in the columns above.
        for key, name in (("div_b_vs_bw", "divergence_rate_b_vs_bw"),
                          ("div_bw_vs_rw", "divergence_rate_bw_vs_rw")):
            m, s = agg_pair(runs, "strict", key)
            row[name + "_mean"], row[name + "_std"] = m, s
        for src, name in (("mag_median_b_vs_bw", "magnitude_median_b_vs_bw"),
                          ("mag_median_bw_vs_rw", "magnitude_median_bw_vs_rw")):
            vals = [r[src] for r in runs if r[src] is not None]
            row[name] = AD.median(vals)
            row[name.replace("median", "p90")] = AD.pct(vals, 90)
        row["magnitude_unit_b_vs_bw"] = "B/w"
        row["magnitude_unit_bw_vs_rw"] = "R/w"
        row["magnitude_unit_b_vs_rw"] = "R/w"

        row["in_reference_cell_list"] = in_cc
        rows.append(row)
    return rows


def build_block2(scans, keys):
    rows = []
    for clean, arm, cell, in_cc in keys:
        runs = cell_runs(scans, clean, arm, cell)
        row = {"clean": clean, "arm": arm, "cell": cell, "n_runs": len(runs)}
        for k in COUNT_KEYS:
            row[k] = sum(r["counts"].get(k, 0) for r in runs)
        row["skip_any_ineligible"] = row["skip_ineligible_only"] + row["skip_both"]
        row["skip_any_stale"] = row["skip_stale_only"] + row["skip_both"]
        row["skip_total"] = sum(row[k] for k in COUNT_KEYS if k != "analysed")
        total = sum(r["total"] for r in runs)
        row["total_decisions"] = total
        for k in COUNT_KEYS + ["skip_any_ineligible", "skip_any_stale", "skip_total"]:
            row["share_" + k] = (row[k] / total) if total else None
        for which, tag in (("strict", "strict"), ("relaxed", "relaxed")):
            row[f"n_analysed_{tag}"] = sum(len(r[which]["div_router"]) for r in runs)
            for key in ("div_router", "div_unc", "tie_router", "tie_unc"):
                m, s = agg_pair(runs, which, key)
                row[f"{tag}_{key}_mean"], row[f"{tag}_{key}_std"] = m, s
        for key in ("div_router", "div_unc", "tie_router", "tie_unc"):
            a, b = row[f"strict_{key}_mean"], row[f"relaxed_{key}_mean"]
            row[f"delta_{key}_relaxed_minus_strict"] = (b - a) if (a is not None and b is not None) else None
        row["in_reference_cell_list"] = in_cc
        rows.append(row)
    return rows


def build_block3(scans, keys, np_ceiling):
    rows = []
    for clean, arm, cell, in_cc in keys:
        runs = cell_runs(scans, clean, arm, cell)
        n_complete = sum(r["n_complete"] for r in runs)
        ceil_n = sum(r["all_at_ceiling"] for r in runs)
        observed = sorted({v for r in runs for n in AD.NODES for v in r["bdist"][n]},
                          key=lambda x: (x is None, x))
        for node in AD.NODES:
            tot = sum(sum(r["bdist"][node].values()) for r in runs)
            for v in observed:
                cnt = sum(r["bdist"][node].get(v, 0) for r in runs)
                rows.append({
                    "clean": clean, "arm": arm, "cell": cell, "node": node,
                    "b_metrics_value": v, "count": cnt,
                    "share_within_node": (cnt / tot) if tot else None,
                    "node_observations": tot,
                    "decisions_complete_sv": n_complete,
                    "np_ceiling": np_ceiling,
                    "all_nodes_at_ceiling_count": ceil_n,
                    "all_nodes_at_ceiling_share": (ceil_n / n_complete) if n_complete else None,
                    "in_reference_cell_list": in_cc,
                })
    return rows


def build_block4(scans, keys):
    rows = []
    for clean, arm, cell, in_cc in keys:
        runs = cell_runs(scans, clean, arm, cell)
        n_analysed = sum(r["counts"].get("analysed", 0) for r in runs)
        n_ties = sum(r["n_ties"] for r in runs)
        merged = defaultdict(lambda: {"count": 0, "du": []})
        for r in runs:
            for k, v in r["ties"].items():
                merged[k]["count"] += v["count"]
                merged[k]["du"].extend(v["du"])

        # Pool raw samples per tied-node set: a mean of per-value medians is not the median of the union.
        pooled_du = defaultdict(list)
        for (tied_k, _value), v in merged.items():
            pooled_du[tied_k].extend(v["du"])
        for (tied, value), v in sorted(merged.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            du = v["du"]
            rows.append({
                "clean": clean, "arm": arm, "cell": cell,
                "tied_nodes": tied, "n_tied_nodes": tied.count(",") + 1,
                "tied_b_value": value, "count": v["count"],
                "share_of_analysed": (v["count"] / n_analysed) if n_analysed else None,
                "share_of_ties": (v["count"] / n_ties) if n_ties else None,
                "du_n": len(du),
                "du_median": AD.median(du), "du_p90": AD.pct(du, 90),
                "du_max": max(du) if du else None,
                # pooled over all tied_b_value for this (cell, tied_nodes)
                "du_pooled_n": len(pooled_du[tied]),
                "du_pooled_median": AD.median(pooled_du[tied]),
                "du_pooled_p90": AD.pct(pooled_du[tied], 90),
                "n_analysed_in_cell": n_analysed, "n_ties_in_cell": n_ties,
                "in_reference_cell_list": in_cc,
            })
    return rows


# ───────────────────────── verification ───────────────────────────────────
def verify_block1(rows):
    """Compare block 1 against divergence_analysis.csv on every column the two
    share. The compared column list is derived from the two headers at runtime,
    not written down here."""
    if not os.path.exists(REFERENCE_CSV):
        return None, [f"{REFERENCE_CSV} not found -- cannot verify."], []
    with open(REFERENCE_CSV) as f:
        rdr = csv.DictReader(f)
        ref_cols = list(rdr.fieldnames)
        ref = {(r["clean"], r["arm"], r["cell"]): r for r in rdr}
    mine = {(str(r["clean"]), r["arm"], r["cell"]): r for r in rows}
    cols = [c for c in ref_cols if c in rows[0] and c not in ("clean", "arm", "cell")]

    problems, extra = [], sorted(set(mine) - set(ref))
    for k in sorted(set(ref) - set(mine)):
        problems.append(f"cell present in divergence_analysis.csv but missing here: {k}")
    worst, n_values = (0.0, None), 0
    for k in sorted(set(ref) & set(mine)):
        for c in cols:
            rv = (ref[k].get(c) or "").strip()
            mv = mine[k][c]
            if rv == "" and mv is None:
                continue
            if rv == "" or mv is None:
                problems.append(f"{k} {c}: reference={rv or '(empty)'} "
                                f"recomputed={'(empty)' if mv is None else mv}")
                continue
            d = abs(float(rv) - float(mv))
            n_values += 1
            if d > worst[0]:
                worst = (d, f"{k} {c}")
            if d > TOL:
                problems.append(f"{k} {c}: reference={float(rv)!r} recomputed={float(mv)!r} "
                                f"absdiff={d:.3e}")
    return ({"cells": len(set(ref) & set(mine)), "values": n_values,
             "columns": cols, "worst": worst}, problems, extra)


# ───────────────────────── output ─────────────────────────────────────────
def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return len(rows), len(rows[0])


def write_readme(outdir, manifest, np_ceiling, np_hits, driver, extra_cells, extra_runs):
    lines = [
        "# SQ1 result-section artefacts",
        "",
        "Generated by `analyze_sq1.py` in a single pass over the router JSONL logs.",
        "Do not edit by hand -- rerun the script.",
        "",
        f"- configured slot count (`-np`): **{np_ceiling}**, read from "
        f"`{os.path.basename(driver)}` ({np_hits[0][2]!r}, line {np_hits[0][1]})",
        f"- block 1 is verified against `divergence_analysis.csv` on every shared column "
        f"before any other output is produced, with warm-up INCLUDED (see below)",
        "",
        "## Warm-up exclusion — changed 2026-08-24",
        "",
        "**These CSVs exclude warm-up requests.** Methodology: *\"Each run issues three "
        "warm-up requests before the 100 measured requests; warm-up is excluded from "
        "every statistic, on both the client and the router side.\"* Until 2026-08-24 "
        "this script counted all 103 router decisions per run, warm-up included; it now "
        "drops them.",
        "",
        "The predicate is the one `analyze_3e.py` already applies (`isinstance("
        "request_id, int) and request_id >= num_warmup`), mirrored because it sits "
        "inline in that script's per-run loop. `num_warmup` is read per run from "
        "`run_meta.bench_summary`, not assumed.",
        "",
        "The reproduction check against `divergence_analysis.csv` is deliberately run "
        "with warm-up **included**, because that frozen CSV was produced before this "
        "change. It therefore still confirms the replication is faithful, and any "
        "difference between these CSVs and the frozen one is the correction, not a bug. "
        "`analyze_divergence.py` and `analyze_skip_bias.py` have not been changed and "
        "still count warm-up.",
        "",
        "## Files",
        "",
        "| file | rows | cols | SQ1 subsection it supports |",
        "|---|---:|---:|---|",
    ]
    for block in sorted(manifest):
        fn, desc, nrows, ncols = manifest[block]
        lines.append(f"| `{fn}` | {nrows} | {ncols} | {desc} |")
    lines += [
        "",
        "## Cell coverage",
        "",
        "Every cell carries an `in_reference_cell_list` flag. `False` marks a cell that "
        "`analyze_divergence.py` drops, because its `CELL_COLS` is a fixed list and "
        "`all_cell_rows` iterates it; such cells appear in no other CSV.",
        "",
    ]
    if extra_cells:
        lines.append("| cell | runs | decisions |")
        lines.append("|---|---:|---:|")
        for c in extra_cells:
            lines.append(f"| `{c}` | {extra_runs[c][0]} | {extra_runs[c][1]} |")
    else:
        lines.append("No cells outside `CELL_COLS` were observed.")
    lines.append("")
    path = os.path.join(outdir, "README.md")
    open(path, "w").write("\n".join(lines) + "\n")
    return path


def print_definitions(np_ceiling, np_hits, driver):
    print("\n" + "#" * 100)
    print("# DEFINITIONS CHOSEN HERE (not read out of existing code)")
    print("#" * 100)
    print(f"1. -np ceiling = {np_ceiling}, parsed from {driver} "
          f"(matched {np_hits[0][2]!r} at line {np_hits[0][1]}; "
          f"{len(np_hits)} match(es), all agreeing). Never assumed.")
    print("2. b_metrics distribution (block 3) is over EVERY decision with a complete "
          "3-node state vector -- analysed and skipped alike -- because saturation is a "
          "property of the signal, not of the analysed subset.")
    print("3. Tie composition (block 4) is over the STRICT-ANALYSED decisions only, so it "
          "decomposes the same b_tie_rate block 1 reports. Ties are on b_metrics (the "
          "router variant); the uncensored variant is not decomposed.")
    print("4. Two-node tie spread is the ABSOLUTE difference |u_a - u_b| of the raw u "
          "field, unweighted, so it does not depend on the weight vector.")
    print("5. Relaxed rule (block 2) = all 3 nodes eligible, stale flag ignored, argmin "
          "over all 3. Eligibility is still required; only staleness is relaxed.")
    print("6. p90 is analyze_divergence.pct's linearly-interpolated percentile, and it is "
          "taken over RUN-level values (run is the unit), matching that script.")
    print("7. Cells outside analyze_divergence.CELL_COLS are reported with "
          "in_reference_cell_list=False rather than dropped.")
    print("8. Weighted least-connections is B_r/w_r using each run's OWN "
          "frozen_config weight vector -- the same vector R/w uses -- with the "
          "same None/zero-weight guard and the same argmin_over tie-breaking.")
    print("9. Pair magnitudes generalise the frozen rule: the pair's SECOND metric, "
          "evaluated at the first policy's pick divided by its own pick, guarded on "
          "denominator > 0. So B vs B/w is in B/w UNITS and is NOT unit-comparable "
          "to the two R/w-denominated magnitudes; the unit is carried per row in "
          "magnitude_unit_*.")
    print("10. The added pairs are computed on the ROUTER b definition (b_metrics) "
          "and on the STRICT gate only. No uncensored (b+deferred) variant of B/w is "
          "invented, and the relaxed rule in block 2 is left as it was.")
    print("11. B vs R/w is not duplicated under a new name: it remains "
          "divergence_rate_router_* / magnitude_*_router, which is what the "
          "verification above compares.")


def main():
    ap = argparse.ArgumentParser(description="SQ1 result-section artefacts (read-only inputs).")
    ap.add_argument("--outdir", default=os.path.join(OUT_HERE, "sq1"))
    ap.add_argument("--driver", default=os.path.join(OUT_HERE, "run_3e.py"),
                    help="driver whose llama-server launch command states -np")
    args = ap.parse_args()

    outdir = os.path.abspath(args.outdir)
    for protected in (REFERENCE_CSV, os.path.join(OUT_HERE, "skip_bias_analysis.csv")):
        if os.path.dirname(protected) == outdir:
            sys.exit(f"FATAL: --outdir {outdir} would sit alongside {protected}; "
                     f"choose a new directory.")
    os.makedirs(outdir, exist_ok=True)

    np_ceiling, np_hits = detect_np(args.driver)

    print("=" * 100)
    print("SQ1 — one artefact for the whole section. Read-only on all inputs.")
    print("=" * 100)
    print(f"-np read from {args.driver}: {np_ceiling}  "
          f"({len(np_hits)} match(es): "
          + ", ".join(f"{t!r}@L{ln}" for _, ln, t in np_hits[:4]) + ")")

    runs = AD.load_all_runs()
    usable = [r for r in runs if r.get("usable")]
    print(f"{len(runs)} run directories, {len(usable)} usable "
          f"(selection imported from analyze_divergence.load_all_runs).")

    # Two passes: warm-ups excluded (written) and included (checked against divergence_analysis.csv).
    warmup_src = Counter()
    warmup_val = Counter()
    scans, scans_incl = [], []
    for r in usable:
        meta = json.loads(open(os.path.join(r["dir"], "run_meta.json")).read())
        nw, src = resolve_num_warmup(meta, r["dir"])
        warmup_src[src] += 1
        warmup_val[nw] += 1
        log = os.path.join(r["dir"], meta["router_log_file"])
        tags = {"clean": r.get("verdict") == "clean", "arm": r["arm"],
                "cell": AD.col_label(r["workload"], r["conc"]), "dir": r["dir"],
                "ref_n_analyzed": r["n_analyzed"], "ref_n_skipped": r["n_skipped"],
                "num_warmup": nw}
        s_excl = scan(meta, log, np_ceiling, nw)
        s_incl = scan(meta, log, np_ceiling, 0)
        for s, bucket in ((s_excl, scans), (s_incl, scans_incl)):
            if s is not None:
                s.update(tags)
                bucket.append(s)

    print(f"\nWarm-up exclusion ACTIVE (methodology: warm-up excluded from every "
          f"statistic, router side included).")
    print(f"  num_warmup values: {dict(warmup_val)}")
    print(f"  resolved from:     {dict(warmup_src)}")
    if len(warmup_val) > 1:
        modal = warmup_val.most_common(1)[0][0]
        print(f"  *** NOT UNIFORM. Runs differing from {modal}:")
        for s in scans:
            if s["num_warmup"] != modal:
                print(f"      {s['dir']}: num_warmup={s['num_warmup']}")
    dropped = sum(s["counts"].get("warmup_excluded", 0) for s in scans)
    kept = sum(s["total"] for s in scans)
    before = sum(s["total"] for s in scans_incl)
    print(f"  warm-up decisions dropped: {dropped}  "
          f"({before} decisions before, {kept} after; {before - kept} removed)")

    # The per-run cross-check must use the INCLUDED pass: analyze_run() counts
    # warm-up, so the filtered counters would legitimately differ from it.
    bad = [s for s in scans_incl if s["counts"].get("analysed", 0) != s["ref_n_analyzed"]
           or s["total"] - s["counts"].get("analysed", 0) != s["ref_n_skipped"]]

    canonical, extra_cells = enumerate_cells(scans)
    all_cells = canonical + extra_cells
    # Main grid cells are always listed (to match divergence_analysis.csv); other cells only where runs exist.
    present = {(s["clean"], s["arm"], s["cell"]) for s in scans}
    keys = []
    for clean in (True, False):
        for arm in sorted({s["arm"] for s in scans if s["clean"] is clean}):
            for cell in all_cells:
                if cell in canonical or (clean, arm, cell) in present:
                    keys.append((clean, arm, cell, cell in canonical))

    block1 = build_block1(scans, keys)                 # written: warm-up EXCLUDED
    block1_incl = build_block1(scans_incl, keys)       # checked: warm-up INCLUDED
    stats, problems, extra_rows = verify_block1(block1_incl)

    print("\n" + "#" * 100)
    print("# VERIFICATION — block 1 vs divergence_analysis.csv (before anything else)")
    print("#" * 100)
    print("Run with warm-up INCLUDED, because divergence_analysis.csv was produced")
    print("without the warm-up filter. This checks the replication is still faithful;")
    print("the CSVs written below are the warm-up-EXCLUDED numbers.")
    print(f"Per-run counters vs analyze_divergence.analyze_run(): "
          f"{len(scans_incl) - len(bad)}/{len(scans_incl)} runs agree.")
    for s in bad:
        print(f"  RUN MISMATCH {s['dir']}")
    if stats:
        print(f"Compared {stats['values']} values across {stats['cells']} shared cells and "
              f"{len(stats['columns'])} shared columns (tolerance {TOL:g}).")
        print(f"Largest absolute difference: {stats['worst'][0]:.3e}"
              + (f" at {stats['worst'][1]}" if stats['worst'][1] else ""))
    if problems:
        print(f"\n*** {len(problems)} MISMATCH(ES) — block 1 does NOT reproduce the CSV. "
              f"Every number below is suspect. ***")
        for p in problems:
            print("  " + p)
    else:
        print("\nRESULT: block 1 reproduces divergence_analysis.csv exactly.")
    if extra_rows:
        print(f"\nAdditional cells present here but absent from divergence_analysis.csv "
              f"(reported, not dropped): {len(extra_rows)}")
        for k in extra_rows:
            print(f"  {k}")

    print_definitions(np_ceiling, np_hits, args.driver)

    extra_runs = {}
    for c in extra_cells:
        rs = [s for s in scans if s["cell"] == c]
        extra_runs[c] = (len(rs), sum(s["total"] for s in rs))
        print(f"\nCell outside CELL_COLS: {c} — {len(rs)} run(s), "
              f"{sum(s['total'] for s in rs)} decisions, "
              f"arms={sorted({s['arm'] for s in rs})}, "
              f"clean={sorted({s['clean'] for s in rs})}")

    # ---- before/after: what the warm-up filter moved ----
    print("\n" + "#" * 100)
    print("# WARM-UP EXCLUSION — before (warm-up in) vs after (warm-up out), clean runs")
    print("#" * 100)
    print(f"{'cell':<9}{'decis before':>13}{'after':>8}"
          f"{'B|R/w before':>14}{'after':>9}{'tie before':>12}{'after':>9}"
          f"{'tie2 bef':>10}{'after':>8}{'|du| med bef':>14}{'after':>9}")
    for cell in all_cells:
        pair = []
        for src in (scans_incl, scans):
            rs = [s for s in src if s["clean"] and s["cell"] == cell]
            if not rs:
                pair.append(None)
                continue
            du = [d for s in rs for k, v in s["ties"].items()
                  if k[0].count(",") == 1 for d in v["du"]]
            pair.append({
                "tot": sum(s["total"] for s in rs),
                "div": rate([v for s in rs for v in s["strict"]["div_router"]]),
                "tie": rate([v for s in rs for v in s["strict"]["tie_router"]]),
                "t2": sum(v["count"] for s in rs for k, v in s["ties"].items()
                          if k[0].count(",") == 1),
                "du": AD.median(du),
            })
        b, a = pair
        if b is None or a is None:
            continue
        g = lambda d, k, nd=3: (f"{d[k]:.{nd}f}" if d[k] is not None else "-")
        print(f"{cell:<9}{b['tot']:>13}{a['tot']:>8}"
              f"{g(b,'div'):>14}{g(a,'div'):>9}{g(b,'tie'):>12}{g(a,'tie'):>9}"
              f"{b['t2']:>10}{a['t2']:>8}{g(b,'du',6):>14}{g(a,'du',6):>9}")
    print("  'decis' = decisions entering the analysis; 'tie2' = two-node tie count;")
    print("  '|du| med' = pooled median |delta u| over two-node ties (block 4 grain).")

    blocks = {1: block1, 2: build_block2(scans, keys),
              3: build_block3(scans, keys, np_ceiling), 4: build_block4(scans, keys)}
    manifest = {}
    for b, rows in blocks.items():
        fn, desc = BLOCK_FILES[b]
        nrows, ncols = write_csv(os.path.join(outdir, fn), rows)
        manifest[b] = (fn, desc, nrows, ncols)

    # ---- stdout summary ----
    print("\n" + "#" * 100)
    print("# SUMMARY (clean runs, pooled across arms; per (clean, arm, cell) detail in the CSVs)")
    print("#" * 100)
    print("Divergence pairs: B|B/w isolates capacity knowledge, B/w|R/w isolates "
          "state-awareness, B|R/w is the frozen headline.")
    print(f"{'cell':<9}{'decisions':>10}{'analysed':>10}{'skip%':>8}"
          f"{'B|B/w':>9}{'B/w|R/w':>9}{'B|R/w':>9}{'B|R/w rel':>10}"
          f"{'tie(str)':>10}{'all@np':>9}{'tie2':>8}{'tie3':>8}")
    for cell in all_cells:
        rs = [s for s in scans if s["clean"] and s["cell"] == cell]
        if not rs:
            continue
        tot = sum(s["total"] for s in rs)
        ana = sum(s["counts"].get("analysed", 0) for s in rs)
        comp = sum(s["n_complete"] for s in rs)
        ceil_n = sum(s["all_at_ceiling"] for s in rs)
        ds = rate([v for s in rs for v in s["strict"]["div_router"]])
        dr = rate([v for s in rs for v in s["relaxed"]["div_router"]])
        d_b_bw = rate([v for s in rs for v in s["strict"]["div_b_vs_bw"]])
        d_bw_rw = rate([v for s in rs for v in s["strict"]["div_bw_vs_rw"]])
        ts = rate([v for s in rs for v in s["strict"]["tie_router"]])
        t2 = sum(v["count"] for s in rs for k, v in s["ties"].items() if k[0].count(",") == 1)
        t3 = sum(v["count"] for s in rs for k, v in s["ties"].items() if k[0].count(",") == 2)
        f = lambda x: f"{x:.3f}" if x is not None else "-"
        print(f"{cell:<9}{tot:>10}{ana:>10}{(1 - ana / tot if tot else 0):>8.3f}"
              f"{f(d_b_bw):>9}{f(d_bw_rw):>9}{f(ds):>9}{f(dr):>10}"
              f"{f(ts):>10}{(ceil_n / comp if comp else 0):>9.3f}{t2:>8}{t3:>8}")
    print("  div/tie pooled over decisions here for compactness; the CSVs carry the "
          "run-level means and SDs that the chapter should quote.")

    readme = write_readme(outdir, manifest, np_ceiling, np_hits, args.driver,
                          extra_cells, extra_runs)
    print(f"\nWritten to {outdir}:")
    for b in sorted(manifest):
        fn, _, nrows, ncols = manifest[b]
        print(f"  block {b}: {fn}  ({nrows} rows, {ncols} cols)")
    print(f"  {os.path.basename(readme)}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
