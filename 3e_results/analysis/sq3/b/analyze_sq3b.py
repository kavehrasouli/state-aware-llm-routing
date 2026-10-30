"""Latency, throughput, TTFT/decode, and dispatch-share results for the retained runs.

Compares state-aware routing and the supporting policies with WLC, and state-aware with WRR.
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, namedtuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))       # .../3e_results
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import analyze_3e as A3  # noqa: E402
from analyze_sq1 import resolve_num_warmup  # noqa: E402
from compare_ttft_decode_split import (  # noqa: E402
    bench_raw, metric_summary, per_request_decode,
)

REPORT_MD = os.path.join(ROOT, "3e_report.md")
RUNGS_MD = os.path.join(ROOT, "state_aware_rungs.md")

BLOCK_FILES = {
    1: ("sq3b_latency_throughput.csv",
        "SQ3b descriptives: end-to-end latency, TPOT and throughput per arm and "
        "cell, as median-of-runs with min-max across repetitions, beside the "
        "number of runs the validity gate excluded."),
    2: ("sq3b_vs_wlc.csv",
        "SQ3b headline comparison: state_aware and the three extensions against "
        "weighted_least_connections, per cell, on the four latency metrics, "
        "throughput and completion rate, with p, bootstrap CI, resolution limit "
        "and leave-one-out."),
    3: ("sq3b_vs_wrr.csv",
        "SQ3b capacity control: state_aware against weighted_round_robin, the "
        "pure capacity baseline, isolating whether live state adds anything "
        "over knowing capacity alone."),
    4: ("sq3b_ttft_decode.csv",
        "SQ3b latency decomposition: ttft and the per-request decode interval "
        "per arm and cell, with the state_aware contrasts against both weighted "
        "baselines."),
    5: ("sq3b_dispatch_share.csv",
        "SQ3b dispatch share: the fraction of measured requests each replica "
        "received per arm and cell, warm-up excluded."),
}

# Baselines and the arms compared against each (state_aware added to analyze_3e's list).
WLC_BASE = "weighted_least_connections"
WRR_BASE = "weighted_round_robin"
VS_WLC_ARMS = ["state_aware"] + list(A3.CANDIDATE_ARMS)
VS_WRR_ARMS = ["state_aware"]

# Metrics: summariser, direction (lower_is_better), run scope, and scale.
# Scope "clean" = retained runs (latency, throughput); "all" = every run (completion).
# Completion is scaled to percentage points.
Metric = namedtuple("Metric",
                    "key summarise lower_is_better scope scale nd unit")


def _latency(metric, percentile):
    """analyze_3e.latency_summary bound to one metric and percentile."""
    def summarise(runs):
        return A3.latency_summary(runs, "overall", metric, percentile)
    return summarise


METRICS = [
    Metric(f"{m}_{p}", _latency(m, p), True, "clean", 1.0,
           (1 if m == "tpot" else 0), "ms")
    for m, p in A3.DECISION_METRICS
] + [
    Metric("throughput_req_s", A3.throughput_summary, False, "clean", 1.0,
           3, "req/s"),
    Metric("completion_rate_pp", A3.completion_summary, False, "all", 100.0,
           1, "pp"),
]
ND = {m.key: m.nd for m in METRICS}

# The WLC -> state-aware completion test is run here too, so it falls inside a BH family;
# it must agree with rung E of sq3a_ladder_completion.csv.
COMPLETION_SKIP = set()

DECODE_METRICS = [("ttft", 50), ("ttft", 95), ("decode", 50), ("decode", 95)]


# ═════════════════════ markdown parsing (cross-check only) ════════════════
def parse_md_tables(path):
    """Every pipe table in a markdown file, tagged with its enclosing headings.

    Mirrored, not imported: 3e_report.md and state_aware_rungs.md are outputs,
    so there is nothing to import. Yields (h2, h3, headers, rows).
    """
    if not os.path.exists(path):
        return []
    out, h2, h3 = [], None, None
    headers, rows, in_table = None, [], False

    def flush():
        nonlocal headers, rows, in_table
        if headers and rows:
            out.append((h2, h3, headers, rows))
        headers, rows, in_table = None, [], False

    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith("## "):
            flush()
            h2, h3 = line[3:].strip(), None
            continue
        if line.startswith("### "):
            flush()
            h3 = line[4:].strip()
            continue
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if headers is None:
                headers, in_table = cells, True
            elif set("".join(cells)) <= set("-: "):
                continue                      # the ---|--- separator row
            else:
                rows.append(cells)
            continue
        if in_table:
            flush()
    flush()
    return out


def balanced_tail(text, start):
    """From an opening '(' at `start`, return the substring inside it, honouring
    nested parentheses. The report's win bullets embed '(fragile)' and
    '(LOO p=...)', so a non-greedy [^)]* would truncate them."""
    depth, i = 0, start
    while i < len(text):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
        i += 1
    return None, len(text)


def parse_report_wins(path):
    """§3's 'Run-level wins over WLC' bullets -> [(arm, cell, metric, fields)].

    Each entry looks like
        c8_bim e2el_p95 (BEAT; p=0.002; Δ=-14896 ms; CI [-20982, -1887])
    with optional '(fragile)' and a trailing leave-one-out clause.
    """
    if not os.path.exists(path):
        return []
    text = open(path).read()
    start = text.find("### Run-level wins over WLC")
    if start < 0:
        return []
    end = text.find("###", start + 5)
    block = text[start:end if end > 0 else len(text)]
    out = []
    for line in block.splitlines():
        m = re.match(r"^- \*\*(?P<arm>[a-z0-9_]+):\*\*\s*(?P<rest>.*)$", line)
        if not m:
            continue
        arm, rest = m.group("arm"), m.group("rest")
        if rest.strip() == "none":
            continue
        pos = 0
        pat = re.compile(r"(c\d+_[a-z]+)\s+([a-z0-9_]+)\s*\(")
        while True:
            mm = pat.search(rest, pos)
            if not mm:
                break
            inner, pos = balanced_tail(rest, mm.end() - 1)
            if inner is None:
                break
            out.append((arm, mm.group(1), mm.group(2), inner))
    return out


def fields_of(inner):
    """'BEAT; p=0.002; Δ=-14896 ms; CI [-20982, -1887]' -> a dict of the
    quantities, ignoring the prose clauses."""
    d = {}
    for part in inner.split("; "):
        part = part.strip()
        if part.startswith("p="):
            d["p"] = part[2:]
        elif part.startswith("Δ="):
            d["delta"] = part[2:].replace(" ms", "").replace(" req/s", "")
        elif part.startswith("CI ["):
            d["ci"] = part[3:]
        elif part.startswith(("BEAT", "LOSE")):
            d["label"] = part
    return d


# ═════════════════════ formatting, matching the sources ═══════════════════
def fmt_delta(v, nd):
    return "n/a" if v is None else f"{v:+.{nd}f}"


def fmt_ci(lo, hi, nd):
    return "n/a" if lo is None else f"[{lo:+.{nd}f}, {hi:+.{nd}f}]"


def fmt_p(p):
    return "n/a" if p is None else f"{p:.3f}"


# ═════════════════════ comparison engine ══════════════════════════════════
def compare(b_runs, c_runs, spec):
    """analyze_3e.run_level_verdict on the two per-run series, plus the two
    resolution figures.

    The run lists arrive already scoped by the caller to whatever spec.scope
    asks for, so nb/nc below are the counts the test actually used -- clean for
    latency and throughput, all runs for completion.
    """
    B, C = spec.summarise(b_runs), spec.summarise(c_runs)
    if spec.scale != 1.0:
        # Scaling also applies to the values, so tests and intervals are in the reported unit.
        B, C = A3.scaled(B, spec.scale), A3.scaled(C, spec.scale)
    nb, nc = len(b_runs), len(c_runs)
    amp = A3.attainable_min_p(nb, nc)
    if B["n"] < 2 or C["n"] < 2:
        return {"tested": False, "n_base": nb, "n_cand": nc,
                "base_median": B["median"], "cand_median": C["median"],
                "attainable_min_p": amp}
    v = A3.run_level_verdict(B, C, lower_is_better=spec.lower_is_better)
    lo, hi = v["ci"]
    return {"tested": True, "n_base": nb, "n_cand": nc,
            "base_median": B["median"], "cand_median": C["median"],
            "diff": v["diff"], "ci_low": lo, "ci_high": hi, "p": v["p"],
            "enumeration": "exhaustive" if v["exact"] else "estimated",
            "exact": v["exact"],
            "perm_p_floor_combinatorial": v["floor"],
            "attainable_min_p": amp,
            "loo_worst_p": v["loo"], "loo_damaging": v["loo_damaging"],
            "loo_resolution_limited": v["loo_resolution_limited"],
            "underpowered_by_floor": v["underpowered"],
            "verdict": v["label"],
            "cleared_alpha": (v["p"] is not None and v["p"] <= A3.ALPHA)}


def build_comparison_block(cells, base_arm, arms):
    """One row per (arm, cell, metric). EVERY comparison is emitted, including
    the ones where nothing cleared alpha and the ones with too few clean runs:
    suppressing null rows would make the surviving p-values uninterpretable."""
    rows = []
    for arm in arms:
        for workload, conc in A3.CELL_COLS:
            cell = A3.col_label(workload, conc)
            b_all = A3.valid_runs(cells, base_arm, conc, workload)
            c_all = A3.valid_runs(cells, arm, conc, workload)
            b_clean = A3.valid_runs(cells, base_arm, conc, workload, clean_only=True)
            c_clean = A3.valid_runs(cells, arm, conc, workload, clean_only=True)
            for spec in METRICS:
                if spec.scope == "all" and (base_arm, arm) in COMPLETION_SKIP:
                    continue
                # Run scope per metric: latency and throughput use retained runs; completion uses all runs.
                b_runs, c_runs = ((b_all, c_all) if spec.scope == "all"
                                  else (b_clean, c_clean))
                r = compare(b_runs, c_runs, spec)
                row = {
                    "base_arm": base_arm, "arm": arm, "cell": cell,
                    "workload": workload, "concurrency": conc,
                    "metric": spec.key, "unit": spec.unit,
                    "lower_is_better": spec.lower_is_better,
                    "run_scope": spec.scope,
                    "n_base": r["n_base"], "n_cand": r["n_cand"],
                    # Survivorship, on every row.
                    "n_base_all": len(b_all),
                    "n_cand_all": len(c_all),
                    "n_base_excluded": len(b_all) - len(b_runs),
                    "n_cand_excluded": len(c_all) - len(c_runs),
                    "base_excluded_pct": ((len(b_all) - len(b_runs)) / len(b_all)
                                          if b_all else None),
                    "cand_excluded_pct": ((len(c_all) - len(c_runs)) / len(c_all)
                                          if c_all else None),
                    "base_median": r["base_median"],
                    "cand_median": r["cand_median"],
                    "attainable_min_p": r["attainable_min_p"],
                    "tested": r["tested"],
                }
                if r["tested"]:
                    row.update({
                        "diff": r["diff"], "ci_low": r["ci_low"],
                        "ci_high": r["ci_high"], "p_permutation": r["p"],
                        "enumeration": r["enumeration"],
                        "perm_p_floor_combinatorial":
                            r["perm_p_floor_combinatorial"],
                        "underpowered_by_floor": r["underpowered_by_floor"],
                        "loo_worst_p": r["loo_worst_p"],
                        "loo_damaging": r["loo_damaging"],
                        "loo_resolution_limited": r["loo_resolution_limited"],
                        "verdict": r["verdict"],
                        "cleared_alpha": r["cleared_alpha"],
                        "untestable_reason": "",
                    })
                else:
                    noun = "run" if spec.scope == "all" else "clean run"
                    why = []
                    if r["n_base"] < 2:
                        why.append(f"{base_arm} has {r['n_base']} {noun}"
                                   f"{'s' if r['n_base'] != 1 else ''}")
                    if r["n_cand"] < 2:
                        why.append(f"{arm} has {r['n_cand']} {noun}"
                                   f"{'s' if r['n_cand'] != 1 else ''}")
                    row.update({
                        "diff": None, "ci_low": None, "ci_high": None,
                        "p_permutation": None, "enumeration": None,
                        "perm_p_floor_combinatorial": None,
                        "underpowered_by_floor": None, "loo_worst_p": None,
                        "loo_damaging": None, "loo_resolution_limited": None,
                        "verdict": "untestable", "cleared_alpha": False,
                        "untestable_reason": " and ".join(why)
                                             + " (a permutation test needs >=2 per arm)",
                    })
                rows.append(row)
    return rows


# ═════════════════════ blocks ═════════════════════════════════════════════
def arms_present(cells):
    observed = {k[0] for k in cells}
    ordered = [a for a in A3.POLICIES if a in observed]
    return ordered + sorted(observed - set(ordered))


def build_block1(cells, arms):
    rows = []
    for arm in arms:
        for workload, conc in A3.CELL_COLS:
            all_runs = A3.valid_runs(cells, arm, conc, workload)
            runs = A3.valid_runs(cells, arm, conc, workload, clean_only=True)
            if not all_runs:
                continue
            row = {
                "arm": arm, "cell": A3.col_label(workload, conc),
                "workload": workload, "concurrency": conc,
                "n_runs": len(runs), "n_all": len(all_runs),
                "n_excluded": len(all_runs) - len(runs),
                "excluded_pct": (len(all_runs) - len(runs)) / len(all_runs),
            }
            for spec in METRICS:
                if spec.scope != "clean":
                    # Completion on all runs is reported in SQ3a, so it is not repeated here.
                    continue
                s = spec.summarise(runs)
                row[f"{spec.key}_median_of_runs"] = s["median"]
                row[f"{spec.key}_min"] = s["min"]
                row[f"{spec.key}_max"] = s["max"]
                row[f"{spec.key}_n_runs"] = s["n"]
                # analyze_3e's own rendering, so the chapter can quote the same
                # string this project already prints in 3e_report.md §2 and §3.
                row[f"{spec.key}_med_range"] = A3.fmt_med_range(s, spec.nd)
            tok = A3.agg([r.get("throughput_tok_s") for r in runs])
            row["throughput_tok_s_median_of_runs"] = tok["median"]
            row["throughput_tok_s_min"] = tok["min"]
            row["throughput_tok_s_max"] = tok["max"]
            row["throughput_tok_s_n_runs"] = tok["n"]
            row["throughput_tok_s_med_range"] = A3.fmt_med_range(tok, 2)
            rows.append(row)
    return rows


def build_block4(cells, arms):
    """ttft and decode per (arm, cell), plus the two state_aware contrasts.

    decode is e2el - ttft computed PER REQUEST and only then reduced to a
    percentile, via compare_ttft_decode_split.per_request_decode. Percentiles of
    ttft and decode do NOT sum to the percentile of e2el -- they are
    percentiles of three different distributions -- and the worked example
    printed at runtime demonstrates it on real data.
    """
    rows = []
    for arm in arms:
        for workload, conc in A3.CELL_COLS:
            cell = A3.col_label(workload, conc)
            all_runs = A3.valid_runs(cells, arm, conc, workload)
            runs = A3.valid_runs(cells, arm, conc, workload, clean_only=True)
            if not all_runs:
                continue
            row = {"arm": arm, "cell": cell, "workload": workload,
                   "concurrency": conc, "n_runs": len(runs),
                   "n_all": len(all_runs),
                   "n_excluded": len(all_runs) - len(runs),
                   "excluded_pct": (len(all_runs) - len(runs)) / len(all_runs)}
            for metric, p in DECODE_METRICS:
                s = metric_summary(runs, metric, p)
                row[f"{metric}_p{p}_median_of_runs"] = s["median"]
                row[f"{metric}_p{p}_min"] = s["min"]
                row[f"{metric}_p{p}_max"] = s["max"]
                row[f"{metric}_p{p}_med_range"] = A3.fmt_med_range(s, 0)
            # e2el carried alongside so the non-additivity is checkable in the
            # CSV itself, not only in the printed example.
            for p in (50, 95):
                s = A3.latency_summary(runs, "overall", "e2el", f"p{p}")
                row[f"e2el_p{p}_median_of_runs"] = s["median"]
                row[f"ttft_plus_decode_p{p}_median_of_runs"] = (
                    (row[f"ttft_p{p}_median_of_runs"]
                     + row[f"decode_p{p}_median_of_runs"])
                    if row.get(f"ttft_p{p}_median_of_runs") is not None
                    and row.get(f"decode_p{p}_median_of_runs") is not None
                    else None)
                a, b = (row[f"ttft_plus_decode_p{p}_median_of_runs"],
                        row[f"e2el_p{p}_median_of_runs"])
                row[f"percentile_sum_residual_p{p}"] = (
                    (a - b) if a is not None and b is not None else None)
            # Pairwise contrasts are shown on the state_aware rows only.
            if arm == "state_aware":
                for base_arm, tag in ((WRR_BASE, "vs_wrr"), (WLC_BASE, "vs_wlc")):
                    b_runs = A3.valid_runs(cells, base_arm, conc, workload,
                                           clean_only=True)
                    for metric, p in DECODE_METRICS:
                        B = metric_summary(b_runs, metric, p)
                        C = metric_summary(runs, metric, p)
                        pre = f"{metric}_p{p}_{tag}"
                        if B["n"] < 2 or C["n"] < 2:
                            row[f"{pre}_diff"] = None
                            row[f"{pre}_p"] = None
                            row[f"{pre}_ci_low"] = None
                            row[f"{pre}_ci_high"] = None
                            row[f"{pre}_enumeration"] = None
                            row[f"{pre}_attainable_min_p"] = \
                                A3.attainable_min_p(B["n"], C["n"])
                            continue
                        v = A3.run_level_verdict(B, C)
                        lo, hi = v["ci"]
                        row[f"{pre}_diff"] = v["diff"]
                        row[f"{pre}_p"] = v["p"]
                        row[f"{pre}_ci_low"] = lo
                        row[f"{pre}_ci_high"] = hi
                        row[f"{pre}_enumeration"] = ("exhaustive" if v["exact"]
                                                     else "estimated")
                        row[f"{pre}_attainable_min_p"] = \
                            A3.attainable_min_p(B["n"], C["n"])
                        row[f"{pre}_verdict"] = v["label"]
            rows.append(row)
    return rows


def dispatch_counts(run):
    """chosen_replica tallies for one run, MEASURED REQUESTS ONLY.

    Mirrors analyze_3e.per_run's warm-up predicate --
        isinstance(request_id, int) and request_id >= num_warmup
    -- because that predicate sits inline in per_run's own loop. num_warmup is
    resolved by analyze_sq1.resolve_num_warmup, the same resolution analyze_sq1
    and analyze_sq2 use, rather than assumed.

    Returns (measured_counts, all_counts, num_warmup, source). all_counts keeps
    the warm-up-inclusive tally so the size of the correction is reportable: the
    `chosen_replica_counts` recorded in run_meta by run_3e.compute_verdict is
    computed over ALL router records, warm-up included.
    """
    meta = json.loads(open(os.path.join(run["dir"], "run_meta.json")).read())
    nw, src = resolve_num_warmup(meta, run["dir"])
    path = os.path.join(run["dir"], meta.get("router_log_file", "router_log.jsonl"))
    measured, every = Counter(), Counter()
    if not os.path.exists(path):
        return measured, every, nw, src
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("type") == "reachability_event":
                continue
            node = r.get("chosen_replica")
            if not node:
                continue
            every[node] += 1
            rid = r.get("request_id")
            if isinstance(rid, int) and rid >= nw:
                measured[node] += 1
    return measured, every, nw, src


def build_block5(cells, arms, disp):
    """Per (arm, cell, node) dispatch share.

    The share is computed PER RUN and then averaged across runs: the run is the
    unit of replication. SD is the SAMPLE SD (ddof = 1, analyze_3e.agg), which
    returns 0.0 at n == 1 -- undefined rather than zero variance -- so n_runs is
    carried beside it. The pooled share over all requests is emitted alongside,
    never instead.
    """
    rows = []
    for arm in arms:
        for workload, conc in A3.CELL_COLS:
            runs = A3.valid_runs(cells, arm, conc, workload, clean_only=True)
            all_runs = A3.valid_runs(cells, arm, conc, workload)
            if not all_runs:
                continue
            pooled, pooled_wu = Counter(), Counter()
            per_run = {n: [] for n in A3.NODES}
            for run in runs:
                m, e, _nw, _s = disp[run["dir"]]
                pooled.update(m)
                pooled_wu.update(e)
                tot = sum(m.values())
                if tot:
                    for n in A3.NODES:
                        per_run[n].append(m.get(n, 0) / tot)
            tot_m, tot_e = sum(pooled.values()), sum(pooled_wu.values())
            for node in A3.NODES:
                a = A3.agg(per_run[node])
                rows.append({
                    "arm": arm, "cell": A3.col_label(workload, conc),
                    "workload": workload, "concurrency": conc, "node": node,
                    "n_runs": len(runs), "n_all": len(all_runs),
                    "n_excluded": len(all_runs) - len(runs),
                    "excluded_pct": (len(all_runs) - len(runs)) / len(all_runs),
                    "share_runmean": a["mean"],
                    "share_runsd": (a["std"] if a["n"] >= 2 else None),
                    "share_runsd_is_defined": a["n"] >= 2,
                    "share_n_runs": a["n"],
                    "share_min": a["min"], "share_max": a["max"],
                    "requests_measured": pooled.get(node, 0),
                    "requests_measured_total": tot_m,
                    "share_pooled": (pooled.get(node, 0) / tot_m) if tot_m else None,
                    # Warm-up-inclusive, for comparison only. Never quote this.
                    "requests_incl_warmup": pooled_wu.get(node, 0),
                    "share_pooled_incl_warmup": ((pooled_wu.get(node, 0) / tot_e)
                                                 if tot_e else None),
                    "warmup_requests_removed": tot_e - tot_m,
                })
    return rows


# ═════════════════════ output ═════════════════════════════════════════════
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


# ═════════════════════ cross-check ════════════════════════════════════════
def check(results, source, key, mine, theirs):
    ok = (mine == theirs)
    results.append({"source": source, "key": key, "mine": mine,
                    "theirs": theirs, "ok": ok})
    return ok


def crosscheck_rungs(cells, results):
    """state_aware_rungs.md's two latency tables, field by field.

    Its rows are produced by compare_state_aware_rungs.latency_block, which
    calls the same analyze_3e helpers this script calls, so an exact string
    match at the printed precision is the correct expectation -- not a
    tolerance.
    """
    n = 0
    for h2, h3, headers, rows in parse_md_tables(RUNGS_MD):
        if not h2 or "->" not in h2 or h3 != "Latency (clean runs only)":
            continue
        base_arm, cand_arm = [s.strip() for s in h2.split("->")]
        idx = {h: i for i, h in enumerate(headers)}
        for r in rows:
            cell, key = r[idx["cell"]], r[idx["metric"]]
            metric, pctl = key.rsplit("_", 1)
            conc, wl = cell_to_wc(cell)
            b_runs = A3.valid_runs(cells, base_arm, conc, wl, clean_only=True)
            c_runs = A3.valid_runs(cells, cand_arm, conc, wl, clean_only=True)
            B = A3.latency_summary(b_runs, "overall", metric, pctl)
            C = A3.latency_summary(c_runs, "overall", metric, pctl)
            v = A3.run_level_verdict(B, C)
            lo, hi = v["ci"]
            nd = 1 if metric == "tpot" else 0
            amp = A3.attainable_min_p(len(b_runs), len(c_runs))
            tag = f"{base_arm}->{cand_arm} {cell} {key}"
            check(results, "state_aware_rungs.md", tag + " n",
                  f"{len(b_runs)}v{len(c_runs)}", r[idx["n"]])
            check(results, "state_aware_rungs.md", tag + " delta",
                  fmt_delta(v["diff"], nd), r[idx["Δ median"]])
            check(results, "state_aware_rungs.md", tag + " ci",
                  fmt_ci(lo, hi, nd), r[idx["95% CI"]])
            check(results, "state_aware_rungs.md", tag + " p",
                  fmt_p(v["p"]), r[idx["p"]])
            check(results, "state_aware_rungs.md", tag + " enum",
                  ("exact" if v["exact"] else "sampled"), r[idx["enum"]])
            check(results, "state_aware_rungs.md", tag + " min_p",
                  (f"{amp:.3f}" if amp is not None else "n/a"), r[idx["min p"]])
            check(results, "state_aware_rungs.md", tag + " verdict",
                  v["label"], r[idx["verdict"]])
            n += 1
    return n


def crosscheck_report_wins(cells, results):
    """§3's 'Run-level wins over WLC' bullets: Δ, p and CI for every win."""
    n = 0
    for arm, cell, key, inner in parse_report_wins(REPORT_MD):
        f = fields_of(inner)
        conc, wl = cell_to_wc(cell)
        base = A3.cell_policy_stats(cells, WLC_BASE, conc, wl)
        cand = A3.cell_policy_stats(cells, arm, conc, wl)
        if key == "goodput":
            v = A3.run_level_verdict(base["throughput"], cand["throughput"],
                                     lower_is_better=False)
            nd = 3
        else:
            v = A3.run_level_verdict(base[key], cand[key])
            nd = 0
        lo, hi = v["ci"]
        tag = f"{arm} {cell} {key}"
        # fmt_verdict appends '~' to a sampled p; strip it and check the flag
        # separately rather than letting a formatting artefact hide a mismatch.
        their_p = f.get("p", "")
        sampled = their_p.endswith("~")
        check(results, "3e_report.md §3 wins", tag + " p",
              fmt_p(v["p"]), their_p.rstrip("~"))
        check(results, "3e_report.md §3 wins", tag + " enum",
              ("sampled" if sampled else "exact"),
              ("sampled" if not v["exact"] else "exact"))
        check(results, "3e_report.md §3 wins", tag + " delta",
              fmt_delta(v["diff"], nd), f.get("delta", ""))
        check(results, "3e_report.md §3 wins", tag + " ci",
              fmt_ci(lo, hi, nd), f.get("ci", ""))
        check(results, "3e_report.md §3 wins", tag + " label",
              v["label"], f.get("label", ""))
        n += 1
    return n


def crosscheck_report_verdicts(cells, results):
    """§2's 'Cell-by-cell verdict' table: the run-level test column, rebuilt
    with analyze_3e.comparison_runlevel_text -- the function that wrote it."""
    n = 0
    for h2, h3, headers, rows in parse_md_tables(REPORT_MD):
        if not h2 or not h2.startswith("2.") or h3 != "Cell-by-cell verdict":
            continue
        idx = {h: i for i, h in enumerate(headers)}
        for r in rows:
            cell = r[idx["cell"]]
            conc, wl = cell_to_wc(cell)
            base = A3.cell_policy_stats(cells, WLC_BASE, conc, wl)
            cand = A3.cell_policy_stats(cells, "state_aware", conc, wl)
            check(results, "3e_report.md §2 verdicts", f"{cell} run-level test",
                  A3.comparison_runlevel_text(base, cand), r[idx["run-level test"]])
            n += 1
    return n


REPORT_DESC_COLS = [
    ("e2el p50", "e2el_p50", 0), ("e2el p95", "e2el_p95", 0),
    ("tpot p50", "tpot_p50", 1), ("tpot p95", "tpot_p95", 1),
    ("goodput req/s", "throughput_req_s", 3),
]
_WL = {"uni": "uniform", "bim": "bimodal", "sgpt": "sharegpt"}


def cell_to_wc(cell):
    """'c12_bim' -> (12, 'bimodal'). The inverse of analyze_3e.col_label."""
    c, w = cell.split("_")
    return int(c[1:]), _WL[w]


def crosscheck_report_descriptives(block1, results):
    """§2 and §3's per-cell tables, against block 1's median [min-max] strings.

    Not required by the SQ3b brief, but free: both sections render exactly the
    quantities block 1 reports, through analyze_3e.fmt_med_range, so any drift
    in the descriptives would show up here rather than in the chapter.
    """
    by = {(r["arm"], r["cell"]): r for r in block1}
    n = 0
    for h2, h3, headers, rows in parse_md_tables(REPORT_MD):
        if not h2 or not (h2.startswith("2.") or h2.startswith("3.")):
            continue
        if not h3 or "policy" not in headers:
            continue
        m = re.match(r"^(uniform|bimodal|sharegpt) c(\d+)$", h3)
        if not m:
            continue
        cell = A3.col_label(m.group(1), int(m.group(2)))
        idx = {h: i for i, h in enumerate(headers)}
        for r in rows:
            arm = r[idx["policy"]]
            mine = by.get((arm, cell))
            if mine is None:
                continue
            src = f"3e_report.md §{h2[0]} descriptives"
            check(results, src, f"{arm} {cell} clean reps",
                  str(mine["n_runs"]), r[idx["clean reps"]])
            for col, key, _nd in REPORT_DESC_COLS:
                if col in idx:
                    check(results, src, f"{arm} {cell} {col}",
                          mine[f"{key}_med_range"], r[idx[col]])
            n += 1
    return n


def warmup_provenance(runs):
    """Do the cross-check sources count warm-up for these metrics?

    Every latency and throughput figure in 3e_report.md and
    state_aware_rungs.md comes through analyze_3e.per_run, whose `lat` block
    and throughput fields read the bench JSON's `raw` array. The check below
    confirms on real data that `raw` holds exactly the measured requests --
    len(raw) == num_prompts, with num_warmup additional requests issued and not
    present -- so those figures are already warm-up-free, and that
    throughput_req_s uses the same warm-up-free numerator.
    """
    out = {"runs": 0, "raw_eq_prompts": 0, "raw_eq_prompts_plus_warmup": 0,
           "thr_matches_completed_over_duration": 0, "warmups": Counter(),
           "bad": []}
    for run in runs:
        meta_p = os.path.join(run["dir"], "run_meta.json")
        try:
            meta = json.loads(open(meta_p).read())
        except OSError:
            continue
        raw = bench_raw(run)
        bs = meta.get("bench_summary") or {}
        nprompts, nwarm = bs.get("num_prompts"), bs.get("num_warmup")
        if nprompts is None or nwarm is None:
            continue
        out["runs"] += 1
        out["warmups"][nwarm] += 1
        if len(raw) == nprompts:
            out["raw_eq_prompts"] += 1
        elif len(raw) == nprompts + nwarm:
            out["raw_eq_prompts_plus_warmup"] += 1
            out["bad"].append(run["dir"])
        thr, dur, comp = (bs.get("throughput_req_s"), bs.get("duration_s"),
                          bs.get("completed"))
        if thr and dur and comp is not None and abs(thr - comp / dur) < 5e-4:
            out["thr_matches_completed_over_duration"] += 1
    return out


# ═════════════════════ README ═════════════════════════════════════════════
def write_readme(outdir, manifest, ctx):
    L = [
        "# SQ3b result-section artefacts",
        "",
        "Generated by `analyze_sq3b.py` from the run folders under `3e_results/`.",
        "Do not edit by hand -- rerun the script. Analysis only; no plots.",
        "",
        "## Files",
        "",
        "| file | rows | cols | SQ3b subsection it supports |",
        "|---|---:|---:|---|",
    ]
    for b in sorted(manifest):
        fn, desc, nr, nc = manifest[b]
        L.append(f"| `{fn}` | {nr} | {nc} | {desc} |")
    L += [
        "",
        "| subsection | read this file | quote these columns |",
        "|---|---|---|",
        f"| SQ3b.1 latency and throughput by arm and cell | `{manifest[1][0]}` | "
        "`*_med_range` (median [min-max]), `n_runs`, `n_excluded` |",
        f"| SQ3b.2 does state_aware or any extension beat WLC | `{manifest[2][0]}` | "
        "`diff`, `p_permutation`, `ci_low`/`ci_high`, `enumeration`, "
        "`attainable_min_p`, `loo_worst_p` |",
        f"| SQ3b.3 does live state add anything over capacity alone | "
        f"`{manifest[3][0]}` | same columns as SQ3b.2 |",
        f"| SQ3b.4 where in the request the difference sits | `{manifest[4][0]}` | "
        "`ttft_p*_med_range`, `decode_p*_med_range`, `*_vs_wrr_*`, `*_vs_wlc_*` |",
        f"| SQ3b.5 how requests were spread across replicas | `{manifest[5][0]}` | "
        "`share_runmean`, `share_runsd`, `share_n_runs` |",
        "",
        "## Scope: runs passing the validity gate",
        "",
        "Latency and throughput are reported on clean runs only "
        "(`analyze_3e.valid_runs(..., clean_only=True)`), per the methodology. "
        "Unlike completion in SQ3a, these metrics describe the experience of "
        "successful, non-truncated requests in retained runs. Rejected runs "
        "remain part of the completion analysis and may reflect policy/load failures.",
        "",
        "**That filter is not neutral across arms, so the survivorship is on "
        "every row of every block**: `n_all` is the runs available, `n_runs` the "
        "clean subset, `n_excluded` and `excluded_pct` the difference. In the "
        "comparison blocks both arms carry their own "
        "`n_base_all`/`n_base_excluded` and `n_cand_all`/`n_cand_excluded`. A "
        "clean-only comparison conditions on a gate partly determined by completion. "
        "Selection can change latency contrasts in either direction. A significant "
        "result is not necessarily conservative; a null is not equivalence.",
        "",
        "## Aggregation conventions",
        "",
        "- **Grouping is `(arm, cell)`, never policy name.** The `wlc_conc` arm "
        "logs `policy == \"weighted_least_connections\"`; keying on the policy "
        "string would pool the re-weighted control into the baseline it exists "
        "to control for. The arm comes from `analyze_3e.per_run`, which resolves "
        "`run_meta[\"arm\"] or run_meta[\"policy\"]`.",
        "- **The run is the unit of replication.** Every block computes the "
        "statistic PER RUN first and only then reduces across runs. Requests "
        "inside one run share a router process, a queue and a thermal state, so "
        "a pooled request-level statistic would understate the variance.",
        "- **Block 1** reports the MEDIAN across runs of each run's own "
        "statistic, with min and max across repetitions -- the "
        "`median [min-max]` convention `analyze_3e.fmt_med_range` prints in "
        "`3e_report.md` §2 and §3, rendered here by that same function.",
        "- **Blocks 2 and 3** test the difference of per-run MEDIANS with "
        "`analyze_3e.perm_test` (two-sided permutation, exhaustive at or below "
        f"`PERM_EXACT_LIMIT` = {A3.PERM_EXACT_LIMIT:,} splits, Monte Carlo with "
        f"{A3.BOOT_ITERS:,} shuffles above it) and interval it with "
        f"`analyze_3e.bootstrap_ci` (percentile CI resampling runs, "
        f"{A3.BOOT_ITERS:,} iterations, seed {A3.BOOT_SEED}). "
        "Each metric carries its own summariser, direction, run scope and "
        "scale: `lower_is_better` is False for throughput and completion and "
        "True for every latency metric, and completion is the one metric "
        "tested on ALL runs rather than clean runs, in percentage points.",
        "- **Block 5** computes each replica's share PER RUN and then averages "
        "across runs. `share_runsd` is the **sample** SD, **ddof = 1**, from "
        "`analyze_3e.agg`, which returns 0.0 when a single run contributes -- "
        "that is undefined, not zero variance, so it is emitted as empty and "
        "`share_runsd_is_defined` marks it, with `share_n_runs` alongside. "
        "`share_pooled` is the same share over all requests pooled, reported "
        "beside the run-level figure and never instead of it.",
        "- Percentiles are `analyze_3e.pct`, linearly interpolated (numpy "
        "type-7).",
        "- `attainable_min_p` is the smallest p a pair of run counts can produce "
        f"under perfect separation. Where it exceeds alpha ({A3.ALPHA}) a large "
        "p is a resolution limit, not evidence of no difference. It is emitted "
        "for every row including the untestable ones, because it depends only on "
        "the run counts. `perm_p_floor_combinatorial` is the different and "
        "smaller 2/C (equal sizes) or 1/C quantity that "
        "`run_level_verdict`'s `underpowered` label keys on; both are emitted so "
        "they are not confused.",
        "",
        "## Block 4: the percentiles do not sum",
        "",
        "`e2el = ttft + decode` holds **per request**, and the decode interval "
        "is therefore derived per request (`e2el_ms - ttft_ms`, under "
        "`analyze_3e.lat_for`'s own success-and-not-truncated filter) and only "
        "then reduced to a percentile, via "
        "`compare_ttft_decode_split.per_request_decode`.",
        "",
        "**It does not follow that p95(ttft) + p95(decode) = p95(e2el).** Those "
        "are percentiles of three different distributions, and their sum is a "
        "percentile of nothing. The CSV carries "
        "`ttft_plus_decode_p*_median_of_runs` and "
        "`percentile_sum_residual_p*` so the size of the gap is visible on every "
        "row rather than assumed away.",
        "",
        ctx["worked_example"],
        "",
        "## Block 5: warm-up is excluded",
        "",
        "The dispatch share counts **measured requests only**. The predicate is "
        "`analyze_3e`'s own (`isinstance(request_id, int) and request_id >= "
        "num_warmup`), mirrored because it sits inline in that script's per-run "
        "loop, with `num_warmup` resolved per run by "
        "`analyze_sq1.resolve_num_warmup` rather than assumed.",
        "",
        f"**An earlier tally counted warm-up.** `run_meta.router_metrics."
        f"chosen_replica_counts`, written by `run_3e.compute_verdict`, is "
        f"computed over ALL router records, warm-up included. Across the runs in "
        f"this block that is {ctx['warmup_removed']} extra requests "
        f"({ctx['warmup_removed_pct']:.2%} of the inclusive total). The CSV "
        f"carries `requests_incl_warmup` and `share_pooled_incl_warmup` for "
        f"comparison only -- **do not quote them**; `share_runmean` and "
        f"`share_pooled` are the warm-up-free figures.",
        "",
        f"The warm-up-free counts computed here were checked request-for-request "
        f"against `analyze_3e.per_run`'s own `router.chosen`: "
        f"{ctx['dispatch_agree']} of {ctx['dispatch_total']} runs agree exactly.",
        "",
        "## Cross-check against the existing reports",
        "",
        f"Before any block is written, blocks 2 and 3 are verified against every "
        f"value `state_aware_rungs.md` and `3e_report.md` print for the "
        f"overlapping comparisons, at the precision they print. "
        f"**{ctx['checks_ok']} of {ctx['checks_total']} compared values match"
        + ("**; no mismatch." if ctx["checks_ok"] == ctx["checks_total"]
           else f", {ctx['checks_total'] - ctx['checks_ok']} MISMATCH.**"),
        "",
        "| source | values compared | matching | mismatched |",
        "|---|---:|---:|---:|",
    ]
    for src, (tot, ok) in sorted(ctx["by_source"].items()):
        L.append(f"| `{src}` | {tot} | {ok} | {tot - ok} |")
    L += [
        "",
        "### Do those sources count warm-up?",
        "",
        "**No, and not because they filter it out -- the bench harness never "
        "puts it in.** Every latency and throughput figure in both files comes "
        "through `analyze_3e.per_run`, whose `lat` block and throughput fields "
        "read the bench JSON's `raw` array. Verified on real data: "
        f"{ctx['raw_eq_prompts']} of {ctx['wp_runs']} runs have "
        f"`len(raw) == num_prompts` exactly, with `num_warmup` "
        f"({', '.join(str(k) for k in sorted(ctx['warmup_values']))}) additional "
        f"requests issued and absent from `raw`; "
        f"{ctx['raw_eq_prompts_plus_warmup']} runs have "
        f"`len(raw) == num_prompts + num_warmup`. "
        f"`throughput_req_s` reproduces `completed / duration_s` in "
        f"{ctx['thr_ok']} of {ctx['wp_runs']} runs, so its denominator is the "
        "measured window too.",
        "",
        "The warm-up question therefore bites only on the **router-side** "
        "counts, which is why block 5 addresses it and blocks 1-4 do not need "
        "to.",
        "",
        "## Resolution note",
        "",
        "Above the enumeration limit, attainable_min_p is a sampling-resolution fallback, not an exhaustive minimum. For SA contrasts use observed_pool_min_p in the sensitivity/report tables, which includes observed ties exactly.",
        "",
        "## Multiplicity",
        "",
        f"This script executed **{ctx['tests_blocks']} permutation tests** while "
        f"building blocks 2, 3 and 4 -- the family the chapter reports. "
        f"**These CSVs contain raw p-values. The separate with_correction stage applies BH; definition B is reported.** At alpha = {A3.ALPHA}, "
        f"about {ctx['tests_blocks'] * A3.ALPHA:.1f} significant results would "
        f"be expected by chance alone across a family this size, so an "
        f"individual p-value describes a single comparison and is not a "
        f"family-wise error rate.",
        "",
        f"Of the {ctx['n_tested']} testable comparisons in blocks 2 and 3, "
        f"{ctx['n_cleared']} cleared alpha without correction; the remaining "
        f"{ctx['tests_blocks'] - ctx['n_tested']} tests in the family are "
        f"block 4's decode contrasts. "
        f"The cross-check re-ran a further {ctx['tests_check']} tests purely to "
        f"reproduce published values; those are verifications, not additional "
        f"inferences, and are excluded from the family count. The count also "
        f"excludes the refits inside `analyze_3e.loo_diagnostics`, matching that "
        f"script's own accounting convention.",
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
    print("2. SCOPE is runs passing the validity gate (clean_only=True), per the")
    print("   methodology. n_all / n_runs / n_excluded / excluded_pct are on EVERY row of")
    print("   EVERY block, both arms in the comparison blocks, so the survivorship is")
    print("   visible beside each result. The filter biases latency comparisons TOWARD")
    print("   zero, so a null result is not evidence of equivalence.")
    print("3. The RUN is the unit of replication. Every statistic is computed per run")
    print("   first, then reduced across runs.")
    print("4. Block 1 reports MEDIAN across runs with [min-max] across repetitions, via")
    print("   analyze_3e.fmt_med_range -- the same rendering 3e_report.md uses.")
    print("5. Blocks 2/3 use analyze_3e.perm_test and bootstrap_ci through")
    print("   run_level_verdict; lower_is_better=False for throughput, True for latency.")
    print("6. Block 5 SD is the SAMPLE SD, ddof = 1 (analyze_3e.agg). It returns 0.0 at")
    print("   n=1, which is undefined rather than zero variance, so it is emitted empty")
    print("   with share_runsd_is_defined=False and share_n_runs alongside.")
    print("7. Percentiles are analyze_3e.pct (linear interpolation, numpy type-7).")
    print("8. attainable_min_p is the smallest p a pair of run counts can produce at all.")
    print("   Above alpha it means unresolvable, NOT equal. Distinct from perm_test's")
    print("   combinatorial floor; both are emitted.")
    print("9. Every comparison is reported, including null and untestable ones.")
    print("10. Block 4: e2el = ttft + decode PER REQUEST. The PERCENTILES DO NOT SUM.")


def worked_example(cells):
    """One real cell, showing that the split is exact per request and that the
    percentiles nonetheless do not add up."""
    for workload, conc in A3.CELL_COLS:
        runs = A3.valid_runs(cells, "state_aware", conc, workload, clean_only=True)
        if not runs:
            continue
        run = runs[0]
        raw = [r for r in bench_raw(run)
               if r.get("success") and not A3.truthy_trunc(r.get("truncated"))
               and isinstance(r.get("e2el_ms"), (int, float))
               and isinstance(r.get("ttft_ms"), (int, float))]
        if len(raw) < 10:
            continue
        dec = per_request_decode(run)
        worst = max(abs((r["ttft_ms"] + d) - r["e2el_ms"])
                    for r, d in zip(raw, dec))
        t95, d95 = A3.pct([r["ttft_ms"] for r in raw], 95), A3.pct(dec, 95)
        e95 = A3.pct([r["e2el_ms"] for r in raw], 95)
        cell = A3.col_label(workload, conc)
        return {
            "cell": cell, "dir": os.path.relpath(run["dir"], ROOT),
            "n": len(raw), "worst": worst,
            "t95": t95, "d95": d95, "e95": e95, "resid": (t95 + d95) - e95,
        }
    return None


def main():
    ap = argparse.ArgumentParser(description="SQ3b result-section artefacts (read-only).")
    ap.add_argument("--outdir", default=HERE)
    ap.add_argument("--allow-mismatch", action="store_true",
                    help="produce the blocks even if the cross-check fails")
    args = ap.parse_args()
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    print("=" * 100)
    print("SQ3b — latency, throughput, where the difference sits, and dispatch "
          "share.")
    print("Read-only on all inputs.")
    print("=" * 100)

    runs = A3.load_all()
    cells = A3.group_cells(runs)
    arms = arms_present(cells)
    print(f"{len(runs)} run directories loaded; {len(cells)} (arm, cell) groups; "
          f"{len(arms)} arms.")

    # ══════════════ WARM-UP PROVENANCE, before anything else ══════════════
    print("\n" + "#" * 100)
    print("# DO 3e_report.md AND state_aware_rungs.md COUNT WARM-UP?")
    print("#" * 100)
    grid = [r for r in runs if r.get("grid") and "router" in r]
    wp = warmup_provenance(grid)
    print("Both files take every latency and throughput figure through")
    print("analyze_3e.per_run, whose lat block and throughput fields read the bench")
    print("JSON's `raw` array. So the question is whether `raw` contains warm-up.")
    print(f"  runs examined:                        {wp['runs']}")
    print(f"  len(raw) == num_prompts:              {wp['raw_eq_prompts']}")
    print(f"  len(raw) == num_prompts + num_warmup: "
          f"{wp['raw_eq_prompts_plus_warmup']}")
    print(f"  num_warmup values observed:           {dict(wp['warmups'])}")
    print(f"  throughput_req_s == completed/duration_s: "
          f"{wp['thr_matches_completed_over_duration']}")
    if wp["raw_eq_prompts"] == wp["runs"] and not wp["bad"]:
        print("\nANSWER: NEITHER FILE COUNTS WARM-UP for these metrics, and not because")
        print("either filters it out -- the bench harness never puts warm-up into `raw`.")
        print("The warm-up requests are issued and discarded client-side, so every")
        print("latency, TPOT and throughput figure in both files is already warm-up-free,")
        print("as is every figure in blocks 1-4 here. The warm-up question bites only on")
        print("ROUTER-side counts, which is why block 5 addresses it and blocks 1-4 do not.")
    else:
        print("\n*** ANSWER: `raw` does NOT uniformly hold exactly the measured requests.")
        for d in wp["bad"]:
            print(f"    {os.path.relpath(d, ROOT)}")

    # ══════════════ CROSS-CHECK, before any block is produced ══════════════
    print("\n" + "#" * 100)
    print("# CROSS-CHECK against 3e_report.md and state_aware_rungs.md")
    print("#" * 100)
    results = []
    n_rungs = crosscheck_rungs(cells, results)
    n_wins = crosscheck_report_wins(cells, results)
    n_verd = crosscheck_report_verdicts(cells, results)
    print(f"state_aware_rungs.md  latency rows compared: {n_rungs}")
    print(f"3e_report.md §3       win entries compared:  {n_wins}")
    print(f"3e_report.md §2       verdict cells compared:{n_verd:>4}")
    tests_check = A3.TESTS_RUN["count"]

    block1 = build_block1(cells, arms)
    n_desc = crosscheck_report_descriptives(block1, results)
    print(f"3e_report.md §2/§3    descriptive rows compared: {n_desc}")

    by_source = {}
    for r in results:
        tot, ok = by_source.get(r["source"], (0, 0))
        by_source[r["source"]] = (tot + 1, ok + (1 if r["ok"] else 0))
    checks_total = len(results)
    checks_ok = sum(1 for r in results if r["ok"])
    print(f"\n{'source':<34}{'compared':>10}{'match':>8}{'MISMATCH':>10}")
    for src, (tot, ok) in sorted(by_source.items()):
        print(f"{src:<34}{tot:>10}{ok:>8}{tot - ok:>10}")
    print(f"{'TOTAL':<34}{checks_total:>10}{checks_ok:>8}"
          f"{checks_total - checks_ok:>10}")

    bad = [r for r in results if not r["ok"]]
    if bad:
        print(f"\n*** {len(bad)} MISMATCH(ES). Every one is listed; nothing is "
              f"summarised away.")
        for r in bad:
            print(f"    [{r['source']}] {r['key']}")
            print(f"        recomputed here: {r['mine']!r}")
            print(f"        as published:    {r['theirs']!r}")
        if not args.allow_mismatch:
            print("\nABORTING before producing any block. The published values and the")
            print("values this script computes from the run folders disagree, so one of")
            print("them is stale; producing SQ3b output on top of that would bury the")
            print("disagreement. Re-run the upstream script, or pass --allow-mismatch to")
            print("produce the blocks anyway with this list on the record.")
            return 2
        print("\n--allow-mismatch given: continuing with the mismatches on the record.")
    else:
        print("\nNO MISMATCH. Every value the two files print for the overlapping")
        print("comparisons is reproduced here at the precision they print it, including")
        print("the difference, bootstrap CI, permutation p, exact/sampled flag,")
        print("attainable minimum p and verdict label.")

    # ══════════════ blocks ══════════════
    tests_before = A3.TESTS_RUN["count"]
    block2 = build_comparison_block(cells, WLC_BASE, VS_WLC_ARMS)
    block3 = build_comparison_block(cells, WRR_BASE, VS_WRR_ARMS)
    block4 = build_block4(cells, arms)
    disp = {}
    for cellruns in cells.values():
        for run in cellruns:
            if run["dir"] not in disp:
                disp[run["dir"]] = dispatch_counts(run)
    block5 = build_block5(cells, arms, disp)
    tests_blocks = A3.TESTS_RUN["count"] - tests_before

    print_conventions()

    # dispatch cross-check against analyze_3e's own warm-up-free counts
    agree = total = 0
    wu_removed = wu_total = 0
    for cellruns in cells.values():
        for run in cellruns:
            m, e, _nw, _s = disp[run["dir"]]
            total += 1
            if dict(m) == {k: v for k, v in
                           (run.get("router", {}).get("chosen") or {}).items() if v}:
                agree += 1
            wu_removed += sum(e.values()) - sum(m.values())
            wu_total += sum(e.values())

    manifest = {}
    for b, rows in ((1, block1), (2, block2), (3, block3), (4, block4),
                    (5, block5)):
        fn, desc = BLOCK_FILES[b]
        nr, nc = write_csv(os.path.join(outdir, fn), rows)
        manifest[b] = (fn, desc, nr, nc)

    # ══════════════ stdout summaries ══════════════
    print("\n" + "#" * 100)
    print("# BLOCK 1 — LATENCY AND THROUGHPUT, clean runs, per arm (cells pooled view)")
    print("#" * 100)
    print(f"{'arm':<28}{'cells':>6}{'runs':>6}{'excl':>6}"
          f"{'e2el p50':>10}{'e2el p95':>10}{'tpot p50':>10}{'tpot p95':>10}"
          f"{'req/s':>8}{'tok/s':>9}")
    for arm in arms:
        rs = [r for r in block1 if r["arm"] == arm]
        if not rs:
            continue
        g = lambda k, nd: A3.median([r[k] for r in rs if r[k] is not None])
        f = lambda v, nd: ("-" if v is None else f"{v:.{nd}f}")
        print(f"{arm:<28}{len(rs):>6}{sum(r['n_runs'] for r in rs):>6}"
              f"{sum(r['n_excluded'] for r in rs):>6}"
              f"{f(g('e2el_p50_median_of_runs', 0), 0):>10}"
              f"{f(g('e2el_p95_median_of_runs', 0), 0):>10}"
              f"{f(g('tpot_p50_median_of_runs', 1), 1):>10}"
              f"{f(g('tpot_p95_median_of_runs', 1), 1):>10}"
              f"{f(g('throughput_req_s_median_of_runs', 3), 3):>8}"
              f"{f(g('throughput_tok_s_median_of_runs', 2), 2):>9}")
    print("  Each column is the MEDIAN ACROSS CELLS of that cell's median-across-runs,")
    print("  shown only to orient; the CSV carries the per-cell median [min-max].")

    for blk, base_arm, label in ((block2, WLC_BASE, "BLOCK 2"),
                                 (block3, WRR_BASE, "BLOCK 3")):
        print("\n" + "#" * 100)
        print(f"# {label} — vs {base_arm}, clean runs, run-level tests")
        print("#" * 100)
        print(f"{'arm':<20}{'cell':<9}{'metric':<17}{'n':>8}{'excl':>9}"
              f"{'diff':>11}{'p':>8}{'enum':>11}{'min p':>8}{'LOO':>8}  verdict")
        for r in blk:
            n = f"{r['n_base']}v{r['n_cand']}"
            ex = f"{r['n_base_excluded']}/{r['n_cand_excluded']}"
            amp = ("-" if r["attainable_min_p"] is None
                   else f"{r['attainable_min_p']:.3f}")
            if not r["tested"]:
                print(f"{r['arm']:<20}{r['cell']:<9}{r['metric']:<17}{n:>8}{ex:>9}"
                      f"{'-':>11}{'-':>8}{'-':>11}{amp:>8}{'-':>8}  untestable")
                continue
            nd = ND.get(r["metric"], 0)
            loo = "-" if r["loo_worst_p"] is None else f"{r['loo_worst_p']:.3f}"
            print(f"{r['arm']:<20}{r['cell']:<9}{r['metric']:<17}{n:>8}{ex:>9}"
                  f"{r['diff']:>+11.{nd}f}{r['p_permutation']:>8.3f}"
                  f"{r['enumeration']:>11}{amp:>8}{loo:>8}  {r['verdict']}")
        cleared = [r for r in blk if r.get("cleared_alpha")]
        print(f"  {len(cleared)} of {sum(1 for r in blk if r['tested'])} tested "
              f"comparisons cleared alpha={A3.ALPHA}. 'excl' is base/candidate runs")
        print("  removed by the validity gate. Negative diff favours the candidate on")
        print("  latency; POSITIVE favours it on throughput.")

    print("\n" + "#" * 100)
    print("# BLOCK 4 — TTFT AND DECODE. THE PERCENTILES DO NOT SUM.")
    print("#" * 100)
    ex = worked_example(cells)
    if ex:
        print(f"Worked example, {ex['dir']} (cell {ex['cell']}, "
              f"{ex['n']} requests):")
        print(f"  The split is EXACT per request: max |ttft + decode - e2el| = "
              f"{ex['worst']:.2e} ms")
        print(f"  The percentiles are NOT additive:")
        print(f"    p95(ttft)   = {ex['t95']:>12.1f} ms")
        print(f"    p95(decode) = {ex['d95']:>12.1f} ms")
        print(f"    sum         = {ex['t95'] + ex['d95']:>12.1f} ms")
        print(f"    p95(e2el)   = {ex['e95']:>12.1f} ms   <- NOT the sum")
        print(f"    residual    = {ex['resid']:>+12.1f} ms")
        print("  p95(ttft) and p95(decode) are percentiles of two DIFFERENT")
        print("  distributions; the request that is 95th slowest to first token is not")
        print("  the request that is 95th slowest to decode. Their sum is a percentile")
        print("  of nothing. This is why decode is derived per request and only then")
        print("  reduced -- p95(e2el) - p95(ttft) would not be a decode percentile.")
    print(f"\n{'arm':<28}{'cell':<9}{'runs':>5}{'ttft p50':>10}{'ttft p95':>10}"
          f"{'dec p50':>10}{'dec p95':>10}{'e2el p95':>10}{'resid p95':>11}")
    for r in block4:
        f = lambda k: ("-" if r.get(k) is None else f"{r[k]:.0f}")
        print(f"{r['arm']:<28}{r['cell']:<9}{r['n_runs']:>5}"
              f"{f('ttft_p50_median_of_runs'):>10}{f('ttft_p95_median_of_runs'):>10}"
              f"{f('decode_p50_median_of_runs'):>10}{f('decode_p95_median_of_runs'):>10}"
              f"{f('e2el_p95_median_of_runs'):>10}"
              f"{f('percentile_sum_residual_p95'):>11}")
    print("  'resid p95' = p95(ttft) + p95(decode) - p95(e2el), per cell. It is not")
    print("  zero and is not expected to be.")
    sa = [r for r in block4 if r["arm"] == "state_aware"]
    print(f"\nstate_aware contrasts (clean runs, negative favours state_aware):")
    print(f"{'cell':<9}{'metric':<12}"
          f"{'d vs wrr':>11}{'p':>8}{'CI vs wrr':>22}"
          f"{'d vs wlc':>11}{'p':>8}{'CI vs wlc':>22}")
    for r in sa:
        for metric, p in DECODE_METRICS:
            k = f"{metric}_p{p}"
            g = lambda tag, fld, nd=0: (
                "-" if r.get(f"{k}_{tag}_{fld}") is None
                else f"{r[f'{k}_{tag}_{fld}']:.{nd}f}")
            ci = lambda tag: ("-" if r.get(f"{k}_{tag}_ci_low") is None else
                              f"[{r[f'{k}_{tag}_ci_low']:+.0f}, "
                              f"{r[f'{k}_{tag}_ci_high']:+.0f}]")
            print(f"{r['cell']:<9}{k:<12}"
                  f"{g('vs_wrr', 'diff'):>11}{g('vs_wrr', 'p', 3):>8}{ci('vs_wrr'):>22}"
                  f"{g('vs_wlc', 'diff'):>11}{g('vs_wlc', 'p', 3):>8}{ci('vs_wlc'):>22}")

    print("\n" + "#" * 100)
    print("# BLOCK 5 — DISPATCH SHARE, clean runs, WARM-UP EXCLUDED")
    print("#" * 100)
    print(f"Warm-up predicate: analyze_3e's own (request_id >= num_warmup), mirrored,")
    print(f"num_warmup from analyze_sq1.resolve_num_warmup. Counts agree with")
    print(f"analyze_3e.per_run's router.chosen in {agree} of {total} runs.")
    print(f"Warm-up requests removed: {wu_removed} of {wu_total} "
          f"({wu_removed / wu_total:.2%}) — run_meta.router_metrics.")
    print(f"chosen_replica_counts includes these; this block does not.")
    print(f"\n{'arm':<28}" + "".join(f"{n:>16}" for n in A3.NODES))
    for arm in arms:
        rs = [r for r in block5 if r["arm"] == arm]
        if not rs:
            continue
        out = ""
        for node in A3.NODES:
            v = [r for r in rs if r["node"] == node]
            tot = sum(r["requests_measured"] for r in v)
            grand = sum(r["requests_measured"] for r in rs)
            m = A3.agg([r["share_runmean"] for r in v])["mean"]
            out += f"{f'{tot / grand:.3f} ({m:.3f})' if grand else '-':>16}"
        print(f"{arm:<28}{out}")
    print("  pooled share (mean of per-cell run-means). Per-cell run-level means and")
    print("  sample SDs (ddof=1) are in the CSV.")

    # ══════════════ multiplicity ══════════════
    n_cleared = sum(1 for r in block2 + block3 if r.get("cleared_alpha"))
    n_tested = sum(1 for r in block2 + block3 if r["tested"])
    print("\n" + "#" * 100)
    print("# MULTIPLICITY — family size, stated so the chapter can address it")
    print("#" * 100)
    print(f"Permutation tests executed while building blocks 2, 3 and 4: {tests_blocks}")
    print(f"  block 2 comparisons enumerated: {len(block2)}")
    print(f"  block 3 comparisons enumerated: {len(block3)}")
    print(f"  of blocks 2+3, testable: {n_tested}; clearing alpha: {n_cleared}")
    print(f"NO multiplicity correction is applied. At alpha={A3.ALPHA}, roughly")
    print(f"{tests_blocks * A3.ALPHA:.1f} significant results would be expected by chance")
    print("alone across a family of this size, so an individual p-value is descriptive of")
    print("ONE comparison and is not a family-wise error rate. The cross-check re-ran a")
    print(f"further {tests_check} tests to reproduce published values; those are")
    print("verifications, not inferences, and are excluded. The count also excludes the")
    print("refits inside analyze_3e.loo_diagnostics, matching that script's convention.")

    we = ("No worked example available." if not ex else
          f"Worked example, `{ex['dir']}` (cell `{ex['cell']}`, {ex['n']} "
          f"requests). The split is exact per request: "
          f"max |ttft + decode - e2el| = {ex['worst']:.2e} ms. The percentiles "
          f"are not: p95(ttft) = {ex['t95']:.1f} ms and p95(decode) = "
          f"{ex['d95']:.1f} ms sum to {ex['t95'] + ex['d95']:.1f} ms, while "
          f"p95(e2el) = {ex['e95']:.1f} ms -- a residual of {ex['resid']:+.1f} "
          f"ms. The request 95th slowest to first token is not the request 95th "
          f"slowest to decode, so the sum is a percentile of nothing.")

    ctx = {
        "worked_example": we,
        "warmup_removed": wu_removed,
        "warmup_removed_pct": (wu_removed / wu_total) if wu_total else 0.0,
        "dispatch_agree": agree, "dispatch_total": total,
        "checks_total": checks_total, "checks_ok": checks_ok,
        "by_source": by_source,
        "wp_runs": wp["runs"], "raw_eq_prompts": wp["raw_eq_prompts"],
        "raw_eq_prompts_plus_warmup": wp["raw_eq_prompts_plus_warmup"],
        "warmup_values": set(wp["warmups"]), "thr_ok":
            wp["thr_matches_completed_over_duration"],
        "tests_blocks": tests_blocks, "tests_check": tests_check,
        "n_cleared": n_cleared, "n_tested": n_tested,
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
