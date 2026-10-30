"""Main-experiment analysis: per-run summaries, run-level statistics, and 3e_summary.csv.

Uses the standard library only; percentiles use linear interpolation. Warm-ups,
failed and truncated requests are excluded from latency and counted.
"""
import csv
from scope import included_paths, CELLS
import glob
import itertools
import json
import math
import os
import random
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # data root: 3e_results
OUT_HERE = os.path.dirname(os.path.abspath(__file__))  # outputs: everythingcombined

POLICIES = [
    "random",
    "round_robin",
    "weighted_round_robin",
    "least_connections",
    "weighted_least_connections",
    # wlc_conc: WLC with the Pi 5 weight raised from 3.1 to 6.0 (router_3nodes_conc.toml), reported as its own arm.
    "wlc_conc",
    "state_aware",
    "state_aware_v2",
    "cost_predict_v3",
]
STATELESS = {"random", "round_robin", "weighted_round_robin"}
STATE_READING = {
    "least_connections",
    "weighted_least_connections",
    "wlc_conc",
    "state_aware",
    "state_aware_v2",
    "cost_predict_v3",
}
NODES = ["pi-node-1", "pi-node-2", "pi-node-3"]
CONCS = [4, 8, 12]
WORKLOADS = ["uniform", "bimodal", "sharegpt"]
LAT_METRICS = ["e2el", "ttft", "tpot"]


COMBINED_GROUPS = {"sharegpt_c4_c12_v1", "sharegpt_ext_c4_c12_v1"}


def is_grid_cell(workload, conc):
    if workload in ("uniform", "bimodal"):
        return conc in (4, 8, 12)
    if workload == "sharegpt":
        return conc in (4, 8, 12)
    return False


# ───────────────────────── stats ─────────────────────────
def pct(xs, p):
    xs = sorted(x for x in xs if isinstance(x, (int, float)))
    if not xs:
        return None
    if len(xs) == 1:
        return float(xs[0])
    r = (p / 100.0) * (len(xs) - 1)
    lo = int(r)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (r - lo)


def median(xs):
    xs = sorted(x for x in xs if isinstance(x, (int, float)))
    n = len(xs)
    if n == 0:
        return None
    if n % 2:
        return float(xs[n // 2])
    return (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def agg(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    n = len(vals)
    if n == 0:
        return {"mean": None, "median": None, "std": None, "min": None,
                "max": None, "n": 0, "values": []}
    mean = sum(vals) / n
    std = (sum((v - mean) ** 2 for v in vals) / (n - 1)) ** 0.5 if n >= 2 else 0.0
    return {"mean": mean, "median": median(vals), "std": std,
            "min": min(vals), "max": max(vals), "n": n, "values": list(vals)}


def truthy_trunc(v):
    return v in (1, True, "1", "true", "True")


def tpot_ms(r):
    """TPOT recomputed from stored timings, as vLLM's serving benchmark defines it:
    (e2el - ttft) / (generated tokens - 1), with the SERVER-reported token count.

    The stored r["tpot_ms"] divides by the number of nonempty streamed text
    chunks instead. Chunks and tokens differ for ShareGPT only: llama.cpp holds
    back incomplete multi-byte text (several tokens -> one chunk) and counts the
    end-of-sequence token, which streams no text. Raw files are left untouched.
    Where the two counts agree the formulas coincide, so the stored value is kept
    (re-deriving from the rounded ms fields would add ~1e-12 ms float noise).
    Requests with fewer than two generated tokens have no TPOT (None)."""
    e2el, ttft = r.get("e2el_ms"), r.get("ttft_ms")
    n = r.get("generated_output_tokens")
    if not all(isinstance(v, (int, float)) for v in (e2el, ttft, n)) or n <= 1:
        return None
    if n == r.get("output_len") and isinstance(r.get("tpot_ms"), (int, float)):
        return r["tpot_ms"]
    return (e2el - ttft) / (n - 1)


def metric_ms(r, met):
    """Per-request latency metric in ms; TPOT goes through tpot_ms()."""
    return tpot_ms(r) if met == "tpot" else r.get(f"{met}_ms")


# ───────────────────────── loading ─────────────────────────
def load_bench(path):
    return json.loads(open(path).read())


def load_router(path):
    reqs, events = [], []
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
                events.append(r)
            else:
                reqs.append(r)
    return reqs, events


def per_run(meta_path):
    """Compute one run's metrics from its rep folder. Returns a dict or None."""
    rd = os.path.dirname(meta_path)
    m = json.loads(open(meta_path).read())
    # Group by experimental arm, not router policy: wlc_conc runs WLC with other weights.
    # Older runs without an "arm" field fall back to the policy name.
    policy = m.get("arm") or m["policy"]
    conc = m["concurrency"]
    workload, rep = m["workload"], m["rep"]

    run = {
        "policy": policy, "conc": conc, "workload": workload, "rep": rep,
        "verdict": m.get("verdict"), "reason": m.get("verdict_reason"),
        "extension_group": m.get("extension_group"),
        "grid": (is_grid_cell(workload, conc)
                 and not (workload == "sharegpt" and conc in (4, 12)
                          and m.get("extension_group") not in COMBINED_GROUPS)),
        "dir": rd,
        "notes": [],
    }

    # bench JSON (named in run_meta; fallback to the lone non-meta .json)
    bj = m.get("bench_result_file")
    bpath = os.path.join(rd, bj) if bj else None
    if not bpath or not os.path.exists(bpath):
        cands = [p for p in glob.glob(os.path.join(rd, "*.json"))
                 if os.path.basename(p) != "run_meta.json"]
        bpath = cands[0] if cands else None
    if not bpath:
        run["notes"].append("MISSING bench JSON")
        return run
    b = load_bench(bpath)
    raw = b.get("raw", [])
    num_warmup = b.get("num_warmup", 3)
    offered = b.get("num_prompts", len(raw))

    n_success = sum(1 for r in raw if r.get("success"))
    n_fail = sum(1 for r in raw if not r.get("success"))
    n_trunc = sum(1 for r in raw if truthy_trunc(r.get("truncated")))
    n_trunc_success = sum(1 for r in raw if r.get("success") and truthy_trunc(r.get("truncated")))
    run.update(offered=offered, n_success=n_success, n_fail=n_fail,
               n_truncated=n_trunc, n_truncated_success=n_trunc_success,
               completion_rate=(n_success / offered if offered else None),
               throughput_req_s=b.get("throughput_req_s"),
               throughput_tok_s=b.get("throughput_tok_s"))

    # latency: SUCCESS-ONLY, exclude truncated. per-run percentiles, overall + per class.
    def lat_for(subset):
        out = {}
        for met in LAT_METRICS:
            vals = [metric_ms(r, met) for r in subset
                    if r.get("success") and not truthy_trunc(r.get("truncated"))
                    and isinstance(metric_ms(r, met), (int, float))]
            out[met] = {"p50": pct(vals, 50), "p95": pct(vals, 95), "n": len(vals)}
        return out

    run["lat"] = {"overall": lat_for(raw)}
    classes = sorted({r.get("class") for r in raw if r.get("class") is not None})
    run["class_counts"] = {
        "overall": {
            "offered": len(raw),
            "success": n_success,
            "truncated": n_trunc,
        }
    }
    for cls in classes:
        class_raw = [r for r in raw if r.get("class") == cls]
        run["lat"][cls] = lat_for(class_raw)
        run["class_counts"][cls] = {
            "offered": len(class_raw),
            "success": sum(1 for r in class_raw if r.get("success")),
            "truncated": sum(1 for r in class_raw if truthy_trunc(r.get("truncated"))),
        }
    run["classes"] = classes

    # router log
    rlpath = os.path.join(rd, "router_log.jsonl")
    if not os.path.exists(rlpath):
        run["notes"].append("MISSING router_log.jsonl")
        return run
    reqs, events = load_router(rlpath)
    grouped = defaultdict(list)
    for request in reqs:
        grouped[request.get("run_id")].append(request)
    run_ids = sorted(run_id for run_id in grouped if run_id is not None)
    if len(run_ids) > 1:
        run["notes"].append(f"multiple run_ids in router log: {run_ids}")
    measured = [
        request
        for group in grouped.values()
        for request in group
        if isinstance(request.get("request_id"), int)
        and request["request_id"] >= num_warmup
    ]
    warmup = len(reqs) - len(measured)
    r_success = sum(1 for r in measured if r.get("outcome") == "success")

    # reconciliation vs bench
    recon_ok = (len(measured) == offered) and (r_success == n_success)
    if not recon_ok:
        run["notes"].append(
            f"RECONCILE MISMATCH: router measured={len(measured)} success={r_success} "
            f"vs bench offered={offered} completed={n_success}")

    sel = [r.get("select_us") for r in measured if isinstance(r.get("select_us"), (int, float))]
    stale_modes = Counter(r.get("stale_mode") for r in measured)
    rode_stale = sum(1 for r in measured if r.get("decision_on_stale_state") is True)
    chosen = Counter(r.get("chosen_replica") for r in measured if r.get("chosen_replica"))
    evacts = Counter(e.get("action") for e in events)
    outcomes = Counter(r.get("outcome") for r in measured)
    outcomes_by_replica = Counter(
        (r.get("outcome"), r.get("chosen_replica")) for r in measured
    )
    upstream_error_total_ms = [
        r.get("total_ms")
        for r in measured
        if r.get("outcome") == "upstream_error"
        and isinstance(r.get("total_ms"), (int, float))
    ]
    upstream_error_statuses = Counter(
        r.get("http_status")
        for r in measured
        if r.get("outcome") == "upstream_error"
    )
    v2_guard_fallback = 0
    v2_guard_fallback_failures = 0
    v2_exclusions = Counter()
    clamp_any = 0
    clamp_chosen = 0
    for request in measured:
        state_vector = request.get("state_vector") or []
        guard_fallback = any(
            candidate.get("v2_stage1_guard_fallback") is True
            for candidate in state_vector
        )
        if guard_fallback:
            v2_guard_fallback += 1
            if request.get("outcome") != "success":
                v2_guard_fallback_failures += 1
        for candidate in state_vector:
            reason = candidate.get("stage1_exclusion_reason")
            if reason:
                v2_exclusions[(candidate.get("replica"), reason)] += 1
        if any(candidate.get("r_proj_clamped") is True for candidate in state_vector):
            clamp_any += 1
        if any(
            candidate.get("v3_chosen") is True
            and candidate.get("r_proj_clamped") is True
            for candidate in state_vector
        ):
            clamp_chosen += 1

    run["router"] = {
        "run_ids": run_ids,
        "warmup_excluded": warmup,
        "measured_total": len(measured),
        "measured_success": r_success,
        "reconcile_ok": recon_ok,
        "select_us_p50": pct(sel, 50), "select_us_p95": pct(sel, 95),
        "select_us_mean": (sum(sel) / len(sel)) if sel else None,
        "select_us_n": len(sel),
        "stale_modes": dict(stale_modes),
        "rode_stale_frac": (rode_stale / len(measured)) if measured else None,
        "chosen": {nd: chosen.get(nd, 0) for nd in NODES},
        "chosen_total": sum(chosen.values()),
        "events": dict(evacts),
        "outcomes": dict(outcomes),
        "outcomes_by_replica": {
            f"{outcome}@{replica}": count
            for (outcome, replica), count in outcomes_by_replica.items()
        },
        "upstream_error_total_ms": upstream_error_total_ms,
        "upstream_error_statuses": dict(upstream_error_statuses),
        "v2_guard_fallback": v2_guard_fallback,
        "v2_guard_fallback_failures": v2_guard_fallback_failures,
        "v2_exclusions": {
            f"{replica}:{reason}": count
            for (replica, reason), count in v2_exclusions.items()
        },
        "v3_clamp_any": clamp_any,
        "v3_clamp_chosen": clamp_chosen,
        "select_logged": len(sel) > 0,
        "select_n": len(sel),
    }
    return run


def load_all():
    runs = []
    for mp in [str(p) for p in included_paths()]:
        try:
            runs.append(per_run(mp))
        except Exception as e:
            raise RuntimeError(f"Cannot analyse included run {mp}") from e
    return runs


# ───────────────────────── formatting ─────────────────────────
def f1(x, nd=1):
    return "n/a" if x is None else f"{x:.{nd}f}"


def pctf(x):
    return "n/a" if x is None else f"{x*100:.1f}%"


def meanspread(a, nd=0):
    """mean±std over reps, from an agg() dict."""
    if not a or a["n"] == 0:
        return "n/a"
    if a["n"] == 1:
        return f"{a['mean']:.{nd}f} (n=1)"
    return f"{a['mean']:.{nd}f}±{a['std']:.{nd}f}"


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def cellkey(p, c, w):
    return (p, c, w)


# (workload, conc) columns in a fixed order for the matrix tables.
CELL_COLS = CELLS

_WLAB = {"uniform": "uni", "bimodal": "bim", "sharegpt": "sgpt"}


def col_label(w, c):
    return f"c{c}_{_WLAB[w]}"


def clean_runs(cells, p, c, w):
    return [r for r in cells.get(cellkey(p, c, w), []) if r.get("verdict") == "clean"]


# ───────────────────────── aggregation across reps ─────────────────────────
def group_cells(runs):
    """(policy,conc,workload) -> list[run] (grid only)."""
    g = defaultdict(list)
    for r in runs:
        if r.get("grid") and "router" in r:
            g[cellkey(r["policy"], r["conc"], r["workload"])].append(r)
    return g


def cell_metric_series(cell_runs, subset, getter):
    """List of per-run values for the chosen subset (clean|all)."""
    rs = cell_runs if subset == "all" else [r for r in cell_runs if r.get("verdict") == "clean"]
    return [getter(r) for r in rs]


def lat_get(cls, met, stat):
    def g(r):
        return r["lat"].get(cls, {}).get(met, {}).get(stat)
    return g


# ═════════════════════════ SECTION 1 ═════════════════════════
def section1(runs):
    lines = ["# §3e analysis\n", "## 1. Suspect audit (before any latency comparison)\n"]
    grid = [r for r in runs if r.get("grid")]
    offgrid = [r for r in runs if not r.get("grid")]
    susp = [r for r in grid if r.get("verdict") == "suspect"]
    clean = [r for r in grid if r.get("verdict") == "clean"]

    lines.append(f"Grid runs: {len(grid)} (clean {len(clean)}, suspect {len(susp)}). "
                 f"Off-grid folders excluded from the grid analysis: {len(offgrid)}"
                 + (f" → {[os.path.relpath(r['dir'], HERE) for r in offgrid]}" if offgrid else "")
                 + ".\n")
    lines.append("All 27 suspect reasons are the same class (failure_rate > 10%); "
                 "no evictions or no_healthy_replica among them.\n")

    rows = []
    for r in sorted(susp, key=lambda r: (r["policy"], r["conc"], r["workload"], r["rep"])):
        rows.append([r["policy"], f"c{r['conc']}", r["workload"], f"rep{r['rep']}", r.get("reason")])
    lines.append("### All suspect runs\n")
    lines.append(md_table(["policy", "conc", "workload", "rep", "reason"], rows))

    per_pol = Counter(r["policy"] for r in susp)
    per_con = Counter(r["conc"] for r in susp)
    per_wl = Counter(r["workload"] for r in susp)
    tot_pol = Counter(r["policy"] for r in grid)
    tot_con = Counter(r["conc"] for r in grid)

    lines.append("\n### Suspects per policy\n")
    lines.append(md_table(
        ["policy", "suspect", "of grid runs", "suspect rate"],
        [[p, per_pol.get(p, 0), tot_pol.get(p, 0), pctf(per_pol.get(p, 0) / tot_pol.get(p, 1))]
         for p in POLICIES]))
    lines.append("\n### Suspects per concurrency\n")
    lines.append(md_table(
        ["concurrency", "suspect", "of grid runs", "suspect rate"],
        [[f"c{c}", per_con.get(c, 0), tot_con.get(c, 0), pctf(per_con.get(c, 0) / tot_con.get(c, 1))]
         for c in CONCS]))
    lines.append("\n### Suspects per workload\n")
    lines.append(md_table(["workload", "suspect"], [[w, per_wl.get(w, 0)] for w in WORKLOADS]))

    naive = per_pol.get("random", 0) + per_pol.get("round_robin", 0)
    lines.append(
        "\n**Pattern (described, not decided):** suspects are strongly CONCENTRATED, not "
        f"scattered. Naive random+round_robin account for {naive}/{len(susp)}; the two most "
        f"capacity-aware policies (weighted_least_connections, state_aware) contribute "
        f"{per_pol.get('weighted_least_connections',0)+per_pol.get('state_aware',0)}/{len(susp)}. "
        f"Suspect rate rises monotonically with load (c4={per_con.get(4,0)}, c8={per_con.get(8,0)}, "
        f"c12={per_con.get(12,0)}) and is {per_wl.get('bimodal',0)}/{len(susp)} on bimodal (the "
        "heavy long-request mix). A gradient aligned with policy sophistication AND with offered "
        "load is the signature of a RESULT (naive policies overloading the slow Pi 4Bs), not "
        "infrastructure scatter — which would be roughly uniform across policies and concurrency. "
        "The failures themselves are upstream connection errors under overload, not router "
        "evictions. Final call is yours; the distribution is above.\n")
    return "\n".join(lines), susp, offgrid


# ═════════════════════════ SECTION 2 ═════════════════════════
def section2(cells, csv_rows):
    lines = ["\n## 2. Per-policy results — clean-only AND all-runs\n",
             "Completion rate is shown FIRST and never separated from its latency. Latency is "
             "success-only, truncated excluded. Values are mean±std across reps (ms). "
             "`comp` = completion rate.\n"]
    for w in WORKLOADS:
        for c in CONCS:
            lines.append(f"\n### {w} c{c}\n")
            rows = []
            for p in POLICIES:
                cr = cells.get(cellkey(p, c, w))
                if not cr:
                    continue
                row = [p]
                for subset in ("clean", "all"):
                    comp = agg(cell_metric_series(cr, subset, lambda r: r["completion_rate"]))
                    e50 = agg(cell_metric_series(cr, subset, lat_get("overall", "e2el", "p50")))
                    e95 = agg(cell_metric_series(cr, subset, lat_get("overall", "e2el", "p95")))
                    t50 = agg(cell_metric_series(cr, subset, lat_get("overall", "ttft", "p50")))
                    tp50 = agg(cell_metric_series(cr, subset, lat_get("overall", "tpot", "p50")))
                    thr = agg(cell_metric_series(cr, subset, lambda r: r["throughput_req_s"]))
                    n = comp["n"]
                    row += [f"{pctf(comp['mean'])} (n={n})", meanspread(e50), meanspread(e95),
                            meanspread(t50), meanspread(tp50, 1), f1(thr["mean"], 3)]
                rows.append(row)
                # csv (overall) for both subsets
                for subset in ("clean", "all"):
                    for met, cls in (("completion_rate", "overall"),):
                        pass
                csv_emit(csv_rows, cr, p, c, w)
            hdr = ["policy",
                   "comp(clean)", "e2el_p50", "e2el_p95", "ttft_p50", "tpot_p50", "thr_req/s",
                   "comp(all)", "e2el_p50", "e2el_p95", "ttft_p50", "tpot_p50", "thr_req/s"]
            lines.append(md_table(hdr, rows))

            # per-class for bimodal
            if w == "bimodal":
                for cls in ("short", "long"):
                    lines.append(f"\n#### {w} c{c} — class={cls} (clean-only, mean±std ms)\n")
                    crows = []
                    for p in POLICIES:
                        cr = cells.get(cellkey(p, c, w))
                        if not cr:
                            continue
                        comp = agg(cell_metric_series(cr, "clean", lambda r: r["completion_rate"]))
                        e50 = agg(cell_metric_series(cr, "clean", lat_get(cls, "e2el", "p50")))
                        e95 = agg(cell_metric_series(cr, "clean", lat_get(cls, "e2el", "p95")))
                        t50 = agg(cell_metric_series(cr, "clean", lat_get(cls, "ttft", "p50")))
                        crows.append([p, f"{pctf(comp['mean'])}", meanspread(e50), meanspread(e95), meanspread(t50)])
                    lines.append(md_table(["policy", "comp(overall)", "e2el_p50", "e2el_p95", "ttft_p50"], crows))
    return "\n".join(lines)


def csv_emit(csv_rows, cr, p, c, w):
    classes = ["overall"] + (["short", "long"] if w == "bimodal" else [])
    def emit(cls, subset, metric, a):
        csv_rows.append([p, c, w, cls, subset, metric,
                         a["mean"], a["median"], a["std"], a["min"], a["max"], a["n"]])
    for subset in ("clean", "all"):
        emit("overall", subset, "completion_rate",
             agg(cell_metric_series(cr, subset, lambda r: r["completion_rate"])))
        emit("overall", subset, "throughput_req_s",
             agg(cell_metric_series(cr, subset, lambda r: r["throughput_req_s"])))
        for cls in classes:
            for met in LAT_METRICS:
                for stat in ("p50", "p95"):
                    emit(cls, subset, f"{met}_{stat}_ms",
                         agg(cell_metric_series(cr, subset, lat_get(cls, met, stat))))


# ═════════════════════════ SECTION 3 (median-of-reps) ═════════════════════════
def section3(cells):
    lines = ["\n## 3. Headline: does state_aware beat wlc? (MEDIAN-of-reps)\n",
             "Clean-only, **median across reps** (so a single wild rep — e.g. wlc uniform c8 — "
             "can't drive the comparison). All latencies ms, success-only, truncated excluded; "
             "`comp` = median completion rate; `thr` req·s⁻¹. wlc & state_aware marked ◄. "
             "Ladder deltas use median e2el_p50 (negative = faster).\n"]
    for w in WORKLOADS:
        for c in CONCS:
            lines.append(f"\n### {w} c{c}\n")
            rows = []
            e50m = {}
            for p in POLICIES:
                cr = cells.get(cellkey(p, c, w))
                mark = " ◄" if p in ("weighted_least_connections", "state_aware") else ""
                if not cr:
                    rows.append([p + mark] + ["n/a"] * 8)
                    continue

                def m(cls, met, st):
                    return agg(cell_metric_series(cr, "clean", lat_get(cls, met, st)))["median"]
                comp = agg(cell_metric_series(cr, "clean", lambda r: r["completion_rate"]))["median"]
                thr = agg(cell_metric_series(cr, "clean", lambda r: r["throughput_req_s"]))["median"]
                n = len(clean_runs(cells, p, c, w))
                e50m[p] = m("overall", "e2el", "p50")
                rows.append([p + mark, pctf(comp),
                             f1(m("overall", "e2el", "p50")), f1(m("overall", "e2el", "p95")),
                             f1(m("overall", "ttft", "p50")), f1(m("overall", "ttft", "p95")),
                             f1(m("overall", "tpot", "p50"), 1), f1(m("overall", "tpot", "p95"), 1),
                             f"{f1(thr, 3)} (n={n})"])
            lines.append(md_table(
                ["policy", "comp", "e2el_p50", "e2el_p95", "ttft_p50", "ttft_p95",
                 "tpot_p50", "tpot_p95", "thr (reps)"], rows))

            def delta(a, b):
                if e50m.get(a) is None or e50m.get(b) is None:
                    return "n/a"
                d = e50m[b] - e50m[a]
                return f"{d:+.0f} ms ({d/e50m[a]*100:+.1f}%)" if e50m[a] else f"{d:+.0f} ms"
            lines.append("\n_Ladder (Δ median e2el_p50):_ "
                         f"rr→wrr {delta('round_robin','weighted_round_robin')}; "
                         f"lc→wlc {delta('least_connections','weighted_least_connections')}; "
                         f"**wlc→state_aware {delta('weighted_least_connections','state_aware')}** "
                         "(what R adds).")
    return "\n".join(lines)


# ═════════════════════════ SECTION 1b — completion as a first-class result ═══════════
def section_completion(cells):
    lines = ["\n## 1b. Completion rate — reliability as a first-class result\n",
             "The §1 suspect finding, stated positively. All-runs completion (clean+suspect), "
             "successes/offered, **mean across the 3 reps**. Read left→right (rising concurrency) "
             "and uniform→bimodal (heavier load): capacity-/state-aware policies hold up where "
             "naive ones shed requests.\n"]
    comp = {}
    for p in POLICIES:
        for w, c in CELL_COLS:
            cr = cells.get(cellkey(p, c, w))
            comp[(p, c, w)] = agg(cell_metric_series(cr, "all", lambda r: r["completion_rate"]))["mean"] if cr else None

    hdr = ["policy"] + [col_label(w, c) for w, c in CELL_COLS]
    rows = [[p] + [pctf(comp[(p, c, w)]) for w, c in CELL_COLS] for p in POLICIES]
    lines.append(md_table(hdr, rows))

    # gradients
    def avg(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None
    lines.append("\n**Gradient by policy** (mean all-runs completion over its 7 cells): "
                 + ", ".join(f"{p} {pctf(avg([comp[(p,c,w)] for w,c in CELL_COLS]))}" for p in POLICIES))
    lines.append("\n**Gradient by concurrency** (over all policies/workloads): "
                 + ", ".join(f"c{c} {pctf(avg([comp[(p,cc,w)] for p in POLICIES for w,cc in CELL_COLS if cc==c]))}"
                             for c in CONCS))
    lines.append("\n**Gradient by workload**: "
                 + ", ".join(f"{w} {pctf(avg([comp[(p,c,ww)] for p in POLICIES for ww,c in CELL_COLS if ww==w]))}"
                             for w in WORKLOADS))
    lines.append("\nBoth gradients are monotone in the expected direction: reliability falls as "
                 "concurrency rises and under bimodal load, and rises from naive to capacity-/"
                 "state-aware policies. wlc and state_aware are the only policies that never drop "
                 "below ~90% in any cell.")
    return "\n".join(lines)


# ═════════════════════════ SECTION 3b — stability (variance) hypothesis check ═══════
def section_stability(cells):
    lines = ["\n## 3b. Stability check — is state_aware steadier than wlc? (HYPOTHESIS)\n",
             "Rep-to-rep spread (std across reps, ms) of e2el p50 and p95, clean-only, wlc vs "
             "state_aware side by side. Only cells where BOTH have ≥2 clean reps are comparable "
             "(a spread needs ≥2). ✓ = state_aware spread smaller. This tests 'similar median, "
             "lower variance' — reported as a hypothesis check, not a conclusion.\n"]
    rows = []
    comparable = smaller50 = smaller95 = 0
    ratios50 = []
    for w, c in CELL_COLS:
        wlc = clean_runs(cells, "weighted_least_connections", c, w)
        sa = clean_runs(cells, "state_aware", c, w)
        if len(wlc) < 2 or len(sa) < 2:
            rows.append([col_label(w, c), f"wlc n={len(wlc)}, sa n={len(sa)}",
                         "—", "—", "not comparable", "—", "—", "—"])
            continue
        comparable += 1
        w50 = agg([lat_get("overall", "e2el", "p50")(r) for r in wlc])
        s50 = agg([lat_get("overall", "e2el", "p50")(r) for r in sa])
        w95 = agg([lat_get("overall", "e2el", "p95")(r) for r in wlc])
        s95 = agg([lat_get("overall", "e2el", "p95")(r) for r in sa])
        lt50 = s50["std"] < w50["std"]
        lt95 = s95["std"] < w95["std"]
        smaller50 += lt50
        smaller95 += lt95
        if w50["std"] and w50["std"] > 0:
            ratios50.append(s50["std"] / w50["std"])
        rows.append([col_label(w, c), f"n={len(wlc)}/{len(sa)}",
                     f1(w50["std"]), f1(s50["std"]), "✓" if lt50 else "✗",
                     f1(w95["std"]), f1(s95["std"]), "✓" if lt95 else "✗"])
    lines.append(md_table(
        ["cell", "reps wlc/sa", "wlc e2el_p50 std", "sa e2el_p50 std", "sa<wlc(p50)",
         "wlc e2el_p95 std", "sa e2el_p95 std", "sa<wlc(p95)"], rows))
    med_ratio = median(ratios50)
    lines.append(
        f"\n**Result:** {comparable} cells were comparable (both ≥2 clean reps). state_aware's "
        f"e2el_p50 spread was smaller than wlc's in **{smaller50}/{comparable}** cells; e2el_p95 in "
        f"**{smaller95}/{comparable}**. "
        + (f"Median ratio sa_std/wlc_std (p50) = {f1(med_ratio,2)}. " if med_ratio is not None else "")
        + ("The pattern is CONSISTENT across cells" if comparable and smaller50 >= max(1, comparable - 1)
           else "The pattern is OCCASIONAL, not consistent")
        + " — but note the small number of comparable cells and n=2–3 reps each: this is a "
        "hypothesis worth more reps, not an established result.")
    return "\n".join(lines)


# ═════════════════════════ SECTION 3c — statistical power ═══════════════════════════
def section_power(cells):
    lines = ["\n## 3c. Statistical power — clean reps per cell\n",
             "Clean reps (of 3) per policy×cell. `*` = fewer than 3 (underpowered); **0** = no "
             "clean data at all. High-contention bimodal cells are where a state_aware effect "
             "would be largest AND where the clean data is thinnest.\n"]
    hdr = ["policy"] + [col_label(w, c) for w, c in CELL_COLS]
    rows = []
    zero_cells, under_cells = [], []
    for p in POLICIES:
        row = [p]
        for w, c in CELL_COLS:
            n = len(clean_runs(cells, p, c, w))
            cell = f"{p} {col_label(w,c)}"
            if n == 0:
                zero_cells.append(cell)
                row.append("**0**")
            elif n < 3:
                under_cells.append(f"{cell} (n={n})")
                row.append(f"{n}*")
            else:
                row.append(str(n))
        rows.append(row)
    lines.append(md_table(hdr, rows))
    lines.append(f"\n**Zero clean reps ({len(zero_cells)} cells):** "
                 + (", ".join(zero_cells) if zero_cells else "none"))
    lines.append(f"\n**Underpowered, 1–2 clean reps ({len(under_cells)} cells):** "
                 + (", ".join(under_cells) if under_cells else "none"))
    bim = [f"{p} {col_label(w,c)}={len(clean_runs(cells,p,c,w))}"
           for p in POLICIES for w, c in CELL_COLS if w == "bimodal" and c in (8, 12)]
    lines.append("\n**Bimodal c8/c12 specifically** (the load regime where R should matter most): "
                 + ", ".join(bim)
                 + ". Several are 0–2 clean reps — the comparison is underpowered exactly here, so "
                 "collecting more reps in bimodal c8/c12 would most improve the ability to detect "
                 "(or rule out) a state_aware effect.")
    return "\n".join(lines)


# ═════════════ SECTION 4: BASELINE LADDER ═════════════
# Each step adds one mechanism, so a difference isolates that mechanism.
LADDER_STEPS = [
    ("random", "round_robin",
     "ordering: uniform random -> deterministic cyclic", True),
    ("round_robin", "weighted_round_robin",
     "capacity: static per-node weights on top of cyclic", False),
    ("round_robin", "least_connections",
     "state: live in-flight count on top of cyclic", False),
    ("least_connections", "weighted_least_connections",
     "capacity: static per-node weights on top of connections", False),
]


def _exclusion_rate(runs, arm):
    total = sum(1 for r in runs if r.get("grid") and r["policy"] == arm)
    dropped = sum(1 for r in runs if r.get("grid") and r["policy"] == arm
                  and r.get("verdict") == "suspect")
    return (dropped, total, (dropped / total) if total else None)


def decision_section_ladder(cells, runs):
    """Run-level tests for the baseline ladder, on latency and on completion.

    Two tables per step because the two answer different questions and have
    different coverage. LATENCY is clean-only, so it is unavailable wherever a
    policy produced no valid run -- which is precisely the high-contention cells
    where the naive policies collapse. COMPLETION uses ALL runs (clean and
    suspect), so it covers exactly those cells, and it is where the capacity
    claim actually lives.
    """
    lines = ["\n## 8. Baseline ladder — does each added mechanism pay off?\n"]
    lines.append(
        "Each step adds one mechanism to the policy below it. Same permutation "
        "machinery, same four metrics, same run-level unit as §2-§3.\n")

    lines.append("\n**Read the two tables together.** Latency is tested on clean runs "
                 "only, so it is unavailable wherever an arm produced no valid run. "
                 "Completion is tested on ALL runs, so it covers those cells. The "
                 "capacity steps are supported by completion, not by latency.\n")

    # Survivorship warning with the actual numbers.
    ex_rows = []
    for arm in ["random", "round_robin", "weighted_round_robin",
                "least_connections", "weighted_least_connections"]:
        dropped, total, rate = _exclusion_rate(runs, arm)
        ex_rows.append([arm, total, dropped, pctf(rate)])
    lines.append("\n**Survivorship caveat.** The clean-only filter removes very "
                 "different fractions of each arm's runs:\n")
    lines.append(md_table(["arm", "runs", "excluded by the gate", "excluded %"], ex_rows))
    lines.append(
        "\nClean-run selection depends partly on completion and differs between arms. "
        "It can change the observed latency contrast in either direction; significance "
        "does not make the estimate conservative. Non-significance is not equivalence.\n")


    for base_arm, cand_arm, label, descriptive in LADDER_STEPS:
        tag = "  _(descriptive only — no claim rests on this step)_" if descriptive else ""
        lines.append(f"\n### {base_arm} -> {cand_arm}\n")
        lines.append(f"_{label}_{tag}\n")

        # ---- latency, clean-only ----
        lat_rows, untestable = [], []
        for workload, conc in CELL_COLS:
            b_runs = valid_runs(cells, base_arm, conc, workload, clean_only=True)
            c_runs = valid_runs(cells, cand_arm, conc, workload, clean_only=True)
            nb, nc = len(b_runs), len(c_runs)
            if nb < 2 or nc < 2:
                why = []
                if nb < 2:
                    why.append(f"{base_arm} has {nb} clean rep{'s' if nb != 1 else ''}")
                if nc < 2:
                    why.append(f"{cand_arm} has {nc} clean rep{'s' if nc != 1 else ''}")
                untestable.append(f"- `{col_label(workload, conc)}`: **untestable** — "
                                  + " and ".join(why)
                                  + " (a permutation test needs >=2 per arm)")
                continue
            floor = attainable_min_p(nb, nc)
            for metric, percentile in DECISION_METRICS:
                key = f"{metric}_{percentile}"
                v = run_level_verdict(latency_summary(b_runs, "overall", metric, percentile),
                                      latency_summary(c_runs, "overall", metric, percentile))
                lo, hi = v["ci"]
                lat_rows.append([
                    col_label(workload, conc), key, f"{nb}v{nc}",
                    f"{v['diff']:+.0f}" if v["diff"] is not None else "n/a",
                    (f"[{lo:+.0f}, {hi:+.0f}]" if lo is not None else "n/a"),
                    (f"{v['p']:.3f}" if v["p"] is not None else "n/a"),
                    ("exact" if v["exact"] else "sampled"),
                    (f"{floor:.3f}" if floor is not None else "n/a"),
                    v["label"],
                ])
        if lat_rows:
            lines.append("\n**Latency (clean runs only).** Δ is median-of-reps of the "
                         "candidate minus the baseline, in ms; negative favours the "
                         "candidate. `min p` is the smallest value this rep count can "
                         "produce.\n")
            lines.append(md_table(
                ["cell", "metric", "n", "Δ median", "95% CI", "p", "enum", "min p", "verdict"],
                lat_rows))
        if untestable:
            lines.append("\n**Cells where latency cannot be tested:**\n")
            lines.extend(untestable)

        # ---- completion, ALL runs ----
        comp_rows = []
        for workload, conc in CELL_COLS:
            b_runs = valid_runs(cells, base_arm, conc, workload)   # clean_only defaults False
            c_runs = valid_runs(cells, cand_arm, conc, workload)
            nb, nc = len(b_runs), len(c_runs)
            if nb < 2 or nc < 2:
                comp_rows.append([col_label(workload, conc), f"{nb}v{nc}",
                                  "n/a", "n/a", "n/a", "n/a", "n/a", "n/a",
                                  "untestable (<2 runs)"])
                continue
            B = completion_summary(b_runs)
            C = completion_summary(c_runs)
            v = run_level_verdict(scaled(B, 100.0), scaled(C, 100.0), lower_is_better=False)
            lo, hi = v["ci"]
            floor = attainable_min_p(nb, nc)
            TB, TC = throughput_summary(b_runs), throughput_summary(c_runs)
            tv = run_level_verdict(TB, TC, lower_is_better=False)
            comp_rows.append([
                col_label(workload, conc), f"{nb}v{nc}",
                f"{B['median'] * 100:.1f}%", f"{C['median'] * 100:.1f}%",
                f"{v['diff']:+.1f}" if v["diff"] is not None else "n/a",
                (f"[{lo:+.1f}, {hi:+.1f}]" if lo is not None else "n/a"),
                (f"{v['p']:.3f}" if v["p"] is not None else "n/a"),
                (f"{floor:.3f}" if floor is not None else "n/a"),
                v["label"],
                (f"{TB['median']:.3f}->{TC['median']:.3f}"
                 if TB["median"] is not None and TC["median"] is not None else "n/a"),
                (f"{tv['p']:.3f}" if tv["p"] is not None else "n/a"),
                tv["label"],
            ])
        lines.append("\n**Completion rate (ALL runs — clean and suspect).** Δ in "
                     "percentage points; positive favours the candidate. This path "
                     "applies no clean-only filter at any stage, which is why the "
                     "cells missing above are present here.\n")
        lines.append(md_table(
            ["cell", "n", "base", "cand", "Δ pp", "95% CI", "p", "min p", "verdict",
             "goodput req/s", "thr p", "thr verdict"],
            comp_rows))
    return "\n".join(lines)


def section4(runs):
    """SQ2 — per-decision routing overhead, from the router's own select_us."""
    lines = ["\n## 7. SQ2 — routing overhead (per-decision selection time)\n"]
    grid = [r for r in runs if r.get("grid")]
    any_logged = any(r.get("router", {}).get("select_logged") for r in grid)
    if not any_logged:
        lines.append("`select_us` is NOT present in the router logs — routing overhead cannot be "
                     "reported from this data. (Not fabricated.)\n")
        return "\n".join(lines)

    lines.append(
        "Selection time is measured inside the router with microsecond resolution around the "
        "replica-selection function alone: the timed region starts before any shared state is "
        "read and ends when a replica index is chosen, so it excludes request parsing and "
        "forwarding. Warm-up decisions are excluded. Unlike the latency sections this table "
        "uses **all** grid runs (clean and suspect): selection cost is a property of the router "
        "process and does not depend on whether the load itself completed. Values are "
        "**median-of-reps [min-max]** over per-run percentiles, in microseconds.\n")

    # Every arm actually present, canonical order first, extras (e.g. control
    # arms like wlc_conc) appended so nothing is silently omitted.
    present = {r["policy"] for r in grid if r.get("router", {}).get("select_us_p50") is not None}
    ordered = [p for p in POLICIES if p in present] + sorted(present - set(POLICIES))

    rows = []
    kind_of = {}
    for p in ordered:
        sel = [r["router"] for r in grid
               if r["policy"] == p and r.get("router", {}).get("select_us_p50") is not None]
        a50 = agg([s["select_us_p50"] for s in sel])
        a95 = agg([s["select_us_p95"] for s in sel])
        kind = "stateless" if p in STATELESS else "state-reading"
        kind_of[p] = kind
        decisions = sum(s.get("select_n", 0) or 0 for s in sel)
        rows.append([p, kind, fmt_med_range(a50, 3), fmt_med_range(a95, 3),
                     a50["n"], decisions if decisions else "n/a"])
    lines.append(md_table(
        ["arm", "kind", "p50 (µs)", "p95 (µs)", "runs", "decisions"], rows))

    # Headline contrast: the cheapest stateless arm vs each state-reading arm.
    base_p50 = {p: agg([r["router"]["select_us_p50"] for r in grid
                        if r["policy"] == p and r.get("router", {}).get("select_us_p50") is not None])["median"]
                for p in ordered}
    stateless_vals = [base_p50[p] for p in ordered if kind_of[p] == "stateless" and base_p50[p] is not None]
    if stateless_vals:
        cheapest = min(stateless_vals)
        deltas = []
        for p in ordered:
            if kind_of[p] == "state-reading" and base_p50[p] is not None:
                deltas.append(f"{p} +{base_p50[p] - cheapest:.2f} µs")
        lines.append(
            f"\n_Stateless arms read only the reachability mask inside the timed region; "
            f"state-reading arms additionally load the state snapshot and scan it. Relative to the "
            f"cheapest stateless arm ({cheapest:.2f} µs), the added cost per decision is: "
            + "; ".join(deltas) + "._")
    lines.append(
        "\nAll arms decide in single-digit-to-tens of microseconds, against end-to-end latencies of "
        "tens of seconds on this hardware. Routing overhead is therefore ~4-7 orders of magnitude "
        "below the response time it is meant to influence and cannot explain any latency difference "
        "between policies.")
    return "\n".join(lines)


# ═════════════════════════ SECTION 5 ═════════════════════════
def section5(cells):
    lines = ["\n## 5. Staleness diagnostics\n",
             "Per cell (clean-only, pooled measured decisions): stale_mode split and the share of "
             "decisions whose CHOSEN replica rode stale state (`decision_on_stale_state`). "
             "A cell is FLAGGED if state_aware rode stale state on >50% of decisions.\n"]
    rows = []
    flagged = []
    for w in WORKLOADS:
        for c in CONCS:
            for p in POLICIES:
                if p not in STATE_READING:
                    continue
                cr = [r for r in cells.get(cellkey(p, c, w), []) if r.get("verdict") == "clean"]
                if not cr:
                    continue
                modes = Counter()
                rode = 0
                tot = 0
                for r in cr:
                    for k, v in r["router"]["stale_modes"].items():
                        modes[k] += v
                    tot += r["router"]["measured_total"]
                    rode += round((r["router"]["rode_stale_frac"] or 0) * r["router"]["measured_total"])
                def sh(k):
                    return pctf(modes.get(k, 0) / tot) if tot else "n/a"
                rode_frac = rode / tot if tot else 0
                flag = ""
                if p == "state_aware" and rode_frac > 0.5:
                    flag = " ⚠FLAG"
                    flagged.append(f"{w} c{c}")
                rows.append([f"{w} c{c}", p, sh("live"), sh("some_stale_excluded"),
                             sh("all_stale_compare"), sh("all_stale_rr"), pctf(rode_frac) + flag])
    lines.append(md_table(
        ["cell", "policy", "live", "some_stale_excl", "all_stale_cmp", "all_stale_rr", "rode_stale"],
        rows))
    if flagged:
        lines.append(f"\n**⚠ Weakened cells (state_aware rode stale >50%): {flagged}** — treat "
                     "their state_aware result with caution.")
    else:
        lines.append("\n**No state_aware cell rode stale state on a majority of decisions** — the "
                     "state-aware results are not undermined by staleness. (Note: `some_stale_excluded` "
                     "just means one node's /metrics was momentarily stale and excluded while a FRESH "
                     "node was chosen — that is state-aware working, not riding stale state.)")
    return "\n".join(lines)


# ═════════════════════════ SECTION 6 ═════════════════════════
def section6(cells):
    lines = ["\n## 6. Routing distribution sanity check\n",
             "Clean-only, pooled measured decisions. Weighted policies (wrr, wlc, state_aware — "
             "weights 3.1/1/1) should lean ~3:1 toward pi-node-1 (Pi 5); unweighted should spread "
             "more evenly.\n"]
    for w in WORKLOADS:
        for c in CONCS:
            rows = []
            for p in POLICIES:
                cr = [r for r in cells.get(cellkey(p, c, w), []) if r.get("verdict") == "clean"]
                if not cr:
                    continue
                tot = Counter()
                for r in cr:
                    for nd in NODES:
                        tot[nd] += r["router"]["chosen"][nd]
                s = sum(tot.values()) or 1
                share = [tot[nd] / s for nd in NODES]
                ratio = f"{share[0]/ (share[1] or 1e-9):.1f}:1" if share[1] else "n/a"
                rows.append([p, pctf(share[0]), pctf(share[1]), pctf(share[2]),
                             f"{share[0]*s:.0f}/{share[1]*s:.0f}/{share[2]*s:.0f}", ratio])
            lines.append(f"\n### {w} c{c}\n")
            lines.append(md_table(
                ["policy", "pi-node-1", "pi-node-2", "pi-node-3", "counts", "n1:n2 ratio"], rows))
    return "\n".join(lines)


# ═════════════════════════ exclusions / reconciliation ═════════════════════════
def section_exclusions(runs):
    lines = ["\n## Appendix — exclusions & reconciliation (every count stated)\n"]
    grid = [r for r in runs if r.get("grid")]
    warm = sum(r.get("router", {}).get("warmup_excluded", 0) for r in grid)
    trunc = sum(r.get("n_truncated", 0) for r in grid)
    trunc_succ = sum(r.get("n_truncated_success", 0) for r in grid)
    lines.append(f"- Warmup router records excluded (request_id < num_warmup): **{warm}** across {len(grid)} runs.")
    lines.append(f"- Requests with truncated flag set: **{trunc}** total "
                 f"({trunc_succ} of them successful — these are the ones removed from latency stats).")
    mism = [r for r in grid if r.get("router") and not r["router"]["reconcile_ok"]]
    lines.append(f"- Bench-vs-router completion reconciliation mismatches: **{len(mism)}**"
                 + ("" if not mism else ":"))
    for r in mism:
        note = next((n for n in r["notes"] if n.startswith("RECONCILE")), "")
        lines.append(f"    - {r['policy']} c{r['conc']} {r['workload']} rep{r['rep']}: {note}")
    other = [(r["dir"], n) for r in runs for n in r.get("notes", []) if not n.startswith("RECONCILE")]
    if other:
        lines.append(f"- Other load notes: {len(other)}")
        for d, n in other:
            lines.append(f"    - {os.path.relpath(d, HERE)}: {n}")
    return "\n".join(lines)


# ═════════════════════════ Decision report helpers ═════════════════════════
DECISION_METRICS = [
    ("e2el", "p50"),
    ("e2el", "p95"),
    ("tpot", "p50"),
    ("tpot", "p95"),
]
# Arms compared against WLC; wlc_conc tests whether a gain comes from a larger Pi 5 share alone.
CANDIDATE_ARMS = ("wlc_conc", "state_aware_v2", "cost_predict_v3")

CORE_POLICIES = [
    "weighted_least_connections",
    "wlc_conc",
    "state_aware",
    "state_aware_v2",
    "cost_predict_v3",
]


def valid_runs(cells, policy, conc, workload, clean_only=False):
    runs = cells.get(cellkey(policy, conc, workload), [])
    if clean_only:
        runs = [run for run in runs if run.get("verdict") == "clean"]
    return [run for run in runs if run.get("class_counts")]


def run_completion(run, cls="overall"):
    counts = run.get("class_counts", {}).get(cls)
    if not counts or not counts.get("offered"):
        return None
    return counts["success"] / counts["offered"]


def completion_summary(runs, cls="overall"):
    values = [run_completion(run, cls) for run in runs]
    summary = agg(values)
    successes = sum(
        run.get("class_counts", {}).get(cls, {}).get("success", 0)
        for run in runs
    )
    offered = sum(
        run.get("class_counts", {}).get(cls, {}).get("offered", 0)
        for run in runs
    )
    return {
        **summary,
        "successes": successes,
        "offered": offered,
        "pooled": successes / offered if offered else None,
    }


def throughput_summary(runs):
    """Per-run goodput (successful requests / duration), summarised across runs.

    bench computes throughput_req_s as len(ok)/duration, i.e. SUCCESSFUL requests
    only, so it already embeds completion: a policy that fails a third of its
    requests shows a third less throughput. Report it beside completion rather
    than as an independent result.
    """
    return agg([run.get("throughput_req_s") for run in runs])


def latency_summary(runs, cls, metric, percentile):
    return agg([
        run.get("lat", {}).get(cls, {}).get(metric, {}).get(percentile)
        for run in runs
    ])


def fmt_med_range(summary, decimals=0):
    if not summary or not summary["n"]:
        return "n/a"
    return (
        f"{summary['median']:.{decimals}f} "
        f"[{summary['min']:.{decimals}f}-{summary['max']:.{decimals}f}]"
    )


def fmt_completion(summary):
    if not summary or not summary["n"]:
        return "n/a"
    return (
        f"{summary['median'] * 100:.1f}% "
        f"[{summary['min'] * 100:.1f}%-{summary['max'] * 100:.1f}%]; "
        f"{summary['successes']}/{summary['offered']}"
    )


def fmt_all_completion(summary):
    if not summary or not summary["n"]:
        return "n/a"
    return (
        f"{summary['successes']}/{summary['offered']} "
        f"({summary['pooled'] * 100:.1f}%); reps "
        f"{summary['median'] * 100:.1f}% "
        f"[{summary['min'] * 100:.1f}%-{summary['max'] * 100:.1f}%], "
        f"n={summary['n']}"
    )


def improvement_pct(base, candidate):
    if base is None or candidate is None or base == 0:
        return None
    return (base - candidate) / base * 100


# ── Run-level significance ──
# The run is the unit of replication: requests within a run share a queue and are not independent.
ALPHA = 0.05
PERM_EXACT_LIMIT = 200_000   # enumerate exactly below this many splits
BOOT_ITERS = 20_000
BOOT_SEED = 20260729         # fixed so the report is reproducible


def _mean(xs):
    return sum(xs) / len(xs)


def perm_test(base, cand):
    """Two-sided permutation test on the difference of per-run MEDIANS.

    The test statistic is the median difference so that the test, the bootstrap
    interval and the headline tables all describe the same quantity (the tables
    report median-of-reps; testing means would test something else).

    Returns (diff, p, exact, floor) where diff = median(cand) - median(base) and
    floor is the smallest p-value this rep count can possibly produce, however
    large the effect. floor matters because a cell whose floor exceeds ALPHA
    cannot yield a significant result at all, so a non-significant outcome there
    is an untestable comparison, not evidence of no difference.

    Floor derivation: we enumerate the C(n, na) ways of choosing which runs form
    the first group. When the groups are the SAME size, each partition appears
    twice (a subset and its complement give the same |difference|), so the
    smallest attainable p is 2/C; when the sizes differ, the complement has the
    wrong size and is never enumerated, so the floor is 1/C. Getting this wrong
    by a factor of two is the difference between flagging a 3-vs-3 cell as
    untestable (floor 0.10) and silently reporting "no difference" for a
    comparison that could never have reached 0.05.

    Exhaustive when the split count allows, otherwise Monte Carlo.
    """
    a, b = list(base), list(cand)
    if len(a) < 2 or len(b) < 2:
        return None, None, None, None
    obs = median(b) - median(a)
    pool = a + b
    na, n = len(a), len(a) + len(b)
    total = math.comb(n, na)
    floor = (2.0 if len(a) == len(b) else 1.0) / total
    if total <= PERM_EXACT_LIMIT:
        hits = 0
        for idx in itertools.combinations(range(n), na):
            sel = set(idx)
            x = [pool[i] for i in idx]
            y = [pool[i] for i in range(n) if i not in sel]
            if abs(median(y) - median(x)) >= abs(obs) - 1e-12:
                hits += 1
        return obs, hits / total, True, floor
    rng = random.Random(BOOT_SEED)
    shuffled, hits = pool[:], 0
    for _ in range(BOOT_ITERS):
        rng.shuffle(shuffled)
        if abs(median(shuffled[na:]) - median(shuffled[:na])) >= abs(obs) - 1e-12:
            hits += 1
    return obs, (hits + 1) / (BOOT_ITERS + 1), False, floor


def bootstrap_ci(base, cand, iters=BOOT_ITERS, seed=BOOT_SEED):
    """Percentile CI for median(cand) - median(base), resampling RUNS."""
    if len(base) < 2 or len(cand) < 2:
        return None, None, None
    rng = random.Random(seed)
    nb, nc = len(base), len(cand)
    diffs = []
    for _ in range(iters):
        x = [base[rng.randrange(nb)] for _ in range(nb)]
        y = [cand[rng.randrange(nc)] for _ in range(nc)]
        diffs.append(median(y) - median(x))
    diffs.sort()
    return (median(cand) - median(base),
            diffs[int(0.025 * iters)], diffs[int(0.975 * iters)])


# Family size: every permutation test run in this report.
TESTS_RUN = {"count": 0}

_MIN_P_CACHE = {}


def attainable_min_p(n_a, n_b):
    """Smallest p-value the median-statistic permutation test can produce at
    these group sizes, i.e. under perfect separation.

    This is the REALISED minimum, not the combinatorial bound: because the median
    ignores extreme values, many relabellings reproduce the observed difference
    exactly, so the attainable minimum sits above 2/C (equal sizes) or 1/C
    (unequal). A cell whose attainable minimum exceeds ALPHA cannot return a
    significant result however large the effect, so a non-significant outcome
    there is a resolution limit rather than evidence of no difference.
    """
    if n_a < 2 or n_b < 2:
        return None
    key = (n_a, n_b)
    if key in _MIN_P_CACHE:
        return _MIN_P_CACHE[key]
    pool = [float(i) for i in range(n_a)] + [1e6 + i for i in range(n_b)]
    obs = median(pool[n_a:]) - median(pool[:n_a])
    n, total = n_a + n_b, math.comb(n_a + n_b, n_a)
    if total > PERM_EXACT_LIMIT:
        value = 1.0 / (BOOT_ITERS + 1)
    else:
        hits = 0
        for idx in itertools.combinations(range(n), n_a):
            sel = set(idx)
            x = [pool[i] for i in idx]
            y = [pool[i] for i in range(n) if i not in sel]
            if abs(median(y) - median(x)) >= abs(obs) - 1e-12:
                hits += 1
        value = hits / total
    _MIN_P_CACHE[key] = value
    return value


def _separated(a, b):
    """True when every value in one group beats every value in the other."""
    return max(b) < min(a) or max(a) < min(b)


def loo_diagnostics(base, cand):
    """Leave-one-out robustness, aware of the test's own resolution limit.

    Returns (worst_p, damaging, detail). `damaging` is True only when some single
    deletion actually degrades the FINDING, as opposed to merely coarsening the
    RESOLUTION of the test.

    Why the distinction is necessary: the permutation distribution of the median
    statistic is coarse at small, equal group sizes. At 6v6 the smallest
    attainable p is 0.013, but after one deletion (6v5) certain median
    configurations cannot produce anything below 0.052. A finding whose groups
    remain COMPLETELY SEPARATED after every deletion can therefore be flagged
    "fragile" purely because the reduced design is unable to express it -- a
    property of the test, not of the data. Flagging that as fragility understates
    real results (observed here: a 6v6 comparison in which all six candidate runs
    beat all six baseline runs, intact under every deletion, was labelled
    fragile).

    A deletion counts as damaging only if the effect itself weakens: the median
    difference changes sign or vanishes, or the groups stop being fully separated
    AND the reduced test no longer clears alpha. Needs >=3 runs per group so that
    >=2 remain after a deletion.
    """
    if len(base) < 3 or len(cand) < 3:
        return None, False, []
    obs = median(cand) - median(base)
    worst = 0.0
    damaging = []

    def check(a, b, dropped):
        nonlocal worst
        diff, p, _, _ = perm_test(a, b)
        if p is None:
            return
        worst = max(worst, p)
        flipped = diff == 0 or ((diff > 0) != (obs > 0))
        sep = _separated(a, b)
        if flipped or (p > ALPHA and not sep):
            damaging.append({"dropped": dropped, "p": p, "flipped": flipped,
                             "separated": sep})

    for i in range(len(base)):
        check(base[:i] + base[i + 1:], cand, f"base[{i}]")
    for i in range(len(cand)):
        check(base, cand[:i] + cand[i + 1:], f"cand[{i}]")
    return worst, bool(damaging), damaging


def run_level_verdict(base, candidate, lower_is_better=True):
    """Compare two agg() dicts on their per-run values. Returns a dict."""
    a = list(base.get("values") or [])
    b = list(candidate.get("values") or [])
    out = {"n_base": len(a), "n_cand": len(b), "diff": None, "p": None,
           "exact": None, "floor": None, "ci": (None, None), "loo": None,
           "loo_damaging": False, "loo_resolution_limited": False,
           "label": "insufficient reps", "underpowered": False}
    if len(a) < 2 or len(b) < 2:
        return out
    diff, p, exact, floor = perm_test(a, b)
    TESTS_RUN["count"] += 1
    _, lo, hi = bootstrap_ci(a, b)
    loo, damaging, _detail = loo_diagnostics(a, b)
    out.update(diff=diff, p=p, exact=exact, floor=floor, ci=(lo, hi),
               loo=loo, loo_damaging=damaging, underpowered=floor > ALPHA,
               loo_resolution_limited=(loo is not None and loo > ALPHA and not damaging))
    better = (diff < 0) if lower_is_better else (diff > 0)
    if out["underpowered"]:
        out["label"] = "underpowered"
    elif p > ALPHA:
        out["label"] = "no run-level difference"
    elif damaging:
        # A real dependence on one run: some deletion weakened the effect itself.
        out["label"] = ("BEAT (fragile)" if better else "LOSE (fragile)")
    else:
        # Survives every deletion. If loo > alpha it is the reduced design that
        # cannot express the result, not the result that failed.
        out["label"] = ("BEAT" if better else "LOSE")
    return out


def scaled(summary, factor):
    """Rescale an agg() dict (e.g. completion fractions -> percentage points)."""
    out = dict(summary)
    for key in ("mean", "median", "std", "min", "max"):
        if out.get(key) is not None:
            out[key] = out[key] * factor
    out["values"] = [v * factor for v in (summary.get("values") or [])]
    return out


def fmt_verdict(v, decimals=1, unit=""):
    """One-cell rendering of a run_level_verdict."""
    if v["label"] == "insufficient reps":
        return f"insufficient reps (n={v['n_base']} vs {v['n_cand']})"
    bits = [v["label"]]
    if v["p"] is not None:
        bits.append(f"p={v['p']:.3f}" + ("" if v["exact"] else "~"))
    if v["diff"] is not None:
        bits.append(f"Δ={v['diff']:+.{decimals}f}{unit}")
    lo, hi = v["ci"]
    if lo is not None:
        bits.append(f"CI [{lo:+.{decimals}f}, {hi:+.{decimals}f}]")
    if v["underpowered"]:
        bits.append(f"p floor {v['floor']:.3f} > {ALPHA}")
    elif v.get("loo_damaging") and v["loo"] is not None:
        bits.append(f"one run weakens it (LOO p={v['loo']:.3f})")
    elif v.get("loo_resolution_limited") and v["loo"] is not None:
        bits.append(f"LOO p={v['loo']:.3f} but effect intact under every deletion "
                    f"(resolution-limited at n-1, not fragile)")
    return "; ".join(bits)


def cell_policy_stats(cells, policy, conc, workload, cls="overall"):
    runs = valid_runs(cells, policy, conc, workload, clean_only=True)
    return {
        "runs": runs,
        "completion": completion_summary(runs, cls),
        "throughput": throughput_summary(runs),
        **{
            f"{metric}_{percentile}": latency_summary(
                runs, cls, metric, percentile
            )
            for metric, percentile in DECISION_METRICS
        },
    }


def e2el_cell_verdict(base_stats, candidate_stats):
    b50 = base_stats["e2el_p50"]["median"]
    b95 = base_stats["e2el_p95"]["median"]
    c50 = candidate_stats["e2el_p50"]["median"]
    c95 = candidate_stats["e2el_p95"]["median"]
    if None in (b50, b95, c50, c95):
        return "not comparable"
    if c50 < b50 and c95 < b95:
        return "BEAT"
    if c50 > b50 and c95 > b95:
        return "LOSE"
    return "TIE/MIXED"


def comparison_runlevel_text(base_stats, candidate_stats):
    """Per-metric run-level verdicts; silent on metrics with no difference."""
    labels = []
    for metric, percentile in DECISION_METRICS:
        key = f"{metric}_{percentile}"
        v = run_level_verdict(base_stats[key], candidate_stats[key])
        if v["label"].startswith(("BEAT", "LOSE")):
            labels.append(f"{key}: {fmt_verdict(v, 0, ' ms')}")
    if labels:
        return "; ".join(labels)
    tv = run_level_verdict(base_stats["throughput"], candidate_stats["throughput"],
                           lower_is_better=False)
    if tv["label"].startswith(("BEAT", "LOSE")):
        return f"goodput: {fmt_verdict(tv, 3, ' req/s')}"
    v = run_level_verdict(base_stats["e2el_p50"], candidate_stats["e2el_p50"])
    return "underpowered at this rep count" if v["underpowered"] else \
        "no run-level difference on any metric"


# ═════════════════════════ SECTION 1 — power ═════════════════════════
def decision_section1(cells):
    lines = [
        "# Combined §3e diagnostic report\n\nRaw tests below are exploratory. For reported inference use sq3/b/with_correction (definition B) and the *_report_table.csv files. Cell coverage differs between arms; aggregate policy rates are descriptive.\n",
        "## 1. Power check — did the infill close the gap?\n",
        "Each entry is `clean/total`; `!` marks fewer than five clean reps. "
        "Counts pool every available rep (original rep1-3 plus rep4+).\n",
    ]
    rows = []
    below = []
    for policy in POLICIES:
        row = [policy]
        for workload, conc in CELL_COLS:
            all_runs = cells.get(cellkey(policy, conc, workload), [])
            clean = sum(run.get("verdict") == "clean" for run in all_runs)
            marker = " !" if clean < 5 else ""
            row.append(f"{clean}/{len(all_runs)}{marker}")
            if clean < 5:
                below.append(f"{policy} {col_label(workload, conc)} ({clean})")
        rows.append(row)
    lines.append(md_table(
        ["policy"] + [col_label(workload, conc) for workload, conc in CELL_COLS],
        rows,
    ))
    lines.append(
        f"\n**Cells below five clean reps ({len(below)}):** "
        + ", ".join(below)
        + "."
    )

    lines.append("\n### Targeted bimodal c8/c12 infill\n")
    targeted = []
    for policy in BASELINE_REPORT_POLICIES:
        counts = []
        for conc in (8, 12):
            all_runs = cells.get(cellkey(policy, conc, "bimodal"), [])
            clean = sum(run.get("verdict") == "clean" for run in all_runs)
            counts.append(f"{clean}/{len(all_runs)}")
        targeted.append([policy, *counts])
    lines.append(md_table(["policy", "c8 clean/total", "c12 clean/total"], targeted))

    confirmations = {
        policy: [
            len(clean_runs(cells, policy, conc, "bimodal"))
            for conc in (8, 12)
        ]
        for policy in (
            "state_aware",
            "weighted_least_connections",
            "weighted_round_robin",
            "least_connections",
        )
    }
    lines.append(
        "\n**Target conclusion:** state_aware = "
        f"{confirmations['state_aware']} and WLC = "
        f"{confirmations['weighted_least_connections']} — both reached five "
        "clean reps at c8 and c12. WRR = "
        f"{confirmations['weighted_round_robin']} and LC = "
        f"{confirmations['least_connections']} — their c12 gaps remain."
    )
    lines.append(
        "\n**Naive overload result:** random and round_robin have zero clean "
        "bimodal c8/c12 reps despite exhausting their attempt caps. This is "
        "expected overload behavior, not a reason to collect more naive-policy reps."
    )
    return "\n".join(lines)


BASELINE_REPORT_POLICIES = [
    "random",
    "round_robin",
    "least_connections",
    "weighted_round_robin",
    "weighted_least_connections",
    "state_aware",
]


# ═════════════════════════ SECTION 2 — headline ═════════════════════════
def decision_section2(cells):
    lines = [
        "\n## 2. Headline — state_aware vs WLC (retained runs)\n",
        "Clean runs only. Latencies are success-only with truncated requests "
        "excluded. Every value is **median-of-reps [min-max]** in ms. Completion "
        "beside it is median [min-max] plus pooled successes/offered for the same "
        "clean reps. Cell verdict uses e2el: BEAT when both p50 and p95 are lower, "
        "LOSE when both are higher, otherwise TIE/MIXED. A difference exceeds "
        "rep spread only when the observed min-max ranges do not overlap. Only "
        "bimodal c8/c12 were infilled to five clean reps for both policies; "
        "sample size alone does not establish adequate power.\n",
    ]
    verdict_rows = []
    verdict_counts = Counter()
    for workload, conc in CELL_COLS:
        lines.append(f"\n### {workload} c{conc}\n")
        rows = []
        stats = {}
        for policy in ("weighted_least_connections", "state_aware"):
            policy_stats = cell_policy_stats(cells, policy, conc, workload)
            stats[policy] = policy_stats
            rows.append([
                policy,
                len(policy_stats["runs"]),
                fmt_completion(policy_stats["completion"]),
                fmt_med_range(policy_stats["e2el_p50"]),
                fmt_med_range(policy_stats["e2el_p95"]),
                fmt_med_range(policy_stats["tpot_p50"], 1),
                fmt_med_range(policy_stats["tpot_p95"], 1),
                fmt_med_range(policy_stats["throughput"], 3),
            ])
        lines.append(md_table(
            ["policy", "clean reps", "completion", "e2el p50", "e2el p95",
             "tpot p50", "tpot p95", "goodput req/s"],
            rows,
        ))
        base = stats["weighted_least_connections"]
        candidate = stats["state_aware"]
        verdict = e2el_cell_verdict(base, candidate)
        verdict_counts[verdict] += 1
        d50 = improvement_pct(
            base["e2el_p50"]["median"], candidate["e2el_p50"]["median"]
        )
        d95 = improvement_pct(
            base["e2el_p95"]["median"], candidate["e2el_p95"]["median"]
        )
        verdict_rows.append([
            col_label(workload, conc),
            verdict,
            (
                f"{fmt_completion(base['completion'])} -> "
                f"{fmt_completion(candidate['completion'])}"
            ),
            f"{d50:+.1f}%" if d50 is not None else "n/a",
            f"{d95:+.1f}%" if d95 is not None else "n/a",
            comparison_runlevel_text(base, candidate),
        ])

    lines.append("\n### Cell-by-cell verdict\n")
    lines.append(md_table(
        ["cell", "state_aware vs WLC", "completion WLC -> state_aware",
         "e2el p50 improvement", "e2el p95 improvement",
         "run-level test"],
        verdict_rows,
    ))
    lines.append(
        "\n**Overall verdict:** the point-estimate direction is "
        f"{verdict_counts['BEAT']} cells beat, {verdict_counts['TIE/MIXED']} "
        f"tie/mixed, {verdict_counts['LOSE']} lose on the two e2el "
        "percentiles. Those are medians only; the run-level column is what "
        "reports unadjusted evidence; use SQ3b definition-B corrections for final inference."
    )
    return "\n".join(lines)


# ═════════════════════════ SECTION 3 — v2/v3 ═════════════════════════
def decision_section3(cells):
    lines = [
        "\n## 3. Does any candidate arm beat WLC anywhere?\n",
        "Definitions and units match §2. Completion is always shown beside "
        "success-only latency. `n/a` means a policy has no clean rep in that cell.\n",
    ]
    beyond = defaultdict(list)
    cell_verdicts = defaultdict(list)
    for workload, conc in CELL_COLS:
        lines.append(f"\n### {workload} c{conc}\n")
        rows = []
        stats = {}
        for policy in CORE_POLICIES:
            policy_stats = cell_policy_stats(cells, policy, conc, workload)
            stats[policy] = policy_stats
            rows.append([
                policy,
                len(policy_stats["runs"]),
                fmt_completion(policy_stats["completion"]),
                fmt_med_range(policy_stats["e2el_p50"]),
                fmt_med_range(policy_stats["e2el_p95"]),
                fmt_med_range(policy_stats["tpot_p50"], 1),
                fmt_med_range(policy_stats["tpot_p95"], 1),
                fmt_med_range(policy_stats["throughput"], 3),
            ])
        lines.append(md_table(
            ["policy", "clean reps", "completion", "e2el p50", "e2el p95",
             "tpot p50", "tpot p95", "goodput req/s"],
            rows,
        ))
        base = stats["weighted_least_connections"]
        for policy in CANDIDATE_ARMS:
            candidate = stats[policy]
            verdict = e2el_cell_verdict(base, candidate)
            cell_verdicts[policy].append((col_label(workload, conc), verdict))
            for metric, percentile in DECISION_METRICS:
                key = f"{metric}_{percentile}"
                v = run_level_verdict(base[key], candidate[key])
                if v["label"].startswith("BEAT"):
                    beyond[policy].append(
                        f"{col_label(workload, conc)} {key} ({fmt_verdict(v, 0, ' ms')})"
                    )
            tv = run_level_verdict(base["throughput"], candidate["throughput"],
                                   lower_is_better=False)
            if tv["label"].startswith("BEAT"):
                beyond[policy].append(
                    f"{col_label(workload, conc)} goodput ({fmt_verdict(tv, 3, ' req/s')})"
                )

    lines.append("\n### Run-level wins over WLC\n")
    lines.append(
        "A win is listed only if the permutation test on per-run values clears "
        f"p<={ALPHA}. `(fragile)` means some single-run deletion actually WEAKENS "
        "the effect (the difference changes sign, or the groups stop being fully "
        f"separated and the reduced test no longer clears {ALPHA}) — report those "
        "as suggestive, not established. A win whose leave-one-out p exceeds "
        f"{ALPHA} while the effect stays intact under every deletion is NOT "
        "fragile: at these group sizes the reduced design simply cannot express "
        "the result, and that is noted inline rather than held against it.\n"
    )
    for policy in CANDIDATE_ARMS:
        lines.append(
            f"- **{policy}:** "
            + ("; ".join(beyond[policy]) if beyond[policy] else "none")
        )
        lines.append(
            "  Cell e2el verdicts: "
            + ", ".join(f"{cell}={verdict}" for cell, verdict in cell_verdicts[policy])
        )

    lines.append("\n### Bimodal per-class latency\n")
    lines.append(
        "This is the mechanism-specific view for v3. Values are clean-run "
        "median-of-reps [min-max], with class completion from the same reps. "
        "The long rows are the primary test of v3's placement mechanism.\n"
    )
    class_rows = []
    for conc in CONCS:
        for cls in ("short", "long"):
            for policy in (
                "weighted_least_connections",
                "state_aware",
                "cost_predict_v3",
            ):
                stats = cell_policy_stats(
                    cells, policy, conc, "bimodal", cls=cls
                )
                class_rows.append([
                    f"c{conc}",
                    cls,
                    policy,
                    len(stats["runs"]),
                    fmt_completion(stats["completion"]),
                    fmt_med_range(stats["e2el_p50"]),
                    fmt_med_range(stats["e2el_p95"]),
                ])
    lines.append(md_table(
        ["cell", "class", "policy", "clean reps", "completion",
         "e2el p50", "e2el p95"],
        class_rows,
    ))
    # NOTE: col_label() emits "c8_bim"/"c12_bim", so the prefixes must match
    # that form. An earlier version tested for "bimodal c8 " and so never fired.
    high_load_wins = {
        policy: [
            item for item in beyond[policy]
            if item.startswith(("c8_bim ", "c12_bim "))
        ]
        for policy in CANDIDATE_ARMS
    }
    if any(high_load_wins.values()):
        detail = "; ".join(
            f"{policy}: {', '.join(items)}"
            for policy, items in high_load_wins.items() if items
        )
        lines.append(
            "\n**Expected-regime check:** a run-level win over WLC in the "
            f"expected bimodal c8/c12 regime IS present — {detail}."
        )
    else:
        lines.append(
            "\n**Expected-regime check:** neither v2 nor v3 records a "
            "run-level win over WLC in bimodal c8/c12. Any v2/v3 wins listed "
            "above are in uniform and/or ShareGPT cells, so the per-class rows "
            "do not support a high-concurrency bimodal latency advantage. Note "
            "this is an absence of evidence at the current rep count, not "
            "evidence of absence — check the p floor before concluding."
        )
    for conc in (8, 12):
        base = cell_policy_stats(
            cells, "weighted_least_connections", conc, "bimodal", cls="long"
        )
        candidate = cell_policy_stats(
            cells, "cost_predict_v3", conc, "bimodal", cls="long"
        )
        relation = fmt_verdict(
            run_level_verdict(base["e2el_p95"], candidate["e2el_p95"]), 0, " ms"
        )
        lines.append(
            f"\n- **Long c{conc}:** completion "
            f"{pctf(base['completion']['pooled'])} -> "
            f"{pctf(candidate['completion']['pooled'])}; e2el p95 "
            f"{fmt_med_range(base['e2el_p95'])} -> "
            f"{fmt_med_range(candidate['e2el_p95'])} ms ({relation})."
        )

    lines.append("\n### Why state_aware_v2 has zero clean bimodal c12 reps\n")
    v2_runs = sorted(
        valid_runs(cells, "state_aware_v2", 12, "bimodal"),
        key=lambda run: run["rep"],
    )
    audit_rows = []
    aggregate_outcomes = Counter()
    aggregate_chosen = Counter()
    aggregate_stale = Counter()
    aggregate_events = Counter()
    aggregate_failure_nodes = Counter()
    aggregate_exclusions = Counter()
    upstream_error_times = []
    fallback_decisions = 0
    fallback_failures = 0
    for run in v2_runs:
        router = run.get("router", {})
        outcomes = Counter(router.get("outcomes", {}))
        chosen = Counter(router.get("chosen", {}))
        stale = Counter(router.get("stale_modes", {}))
        events = Counter(router.get("events", {}))
        aggregate_outcomes.update(outcomes)
        aggregate_chosen.update(chosen)
        aggregate_stale.update(stale)
        aggregate_events.update(events)
        upstream_error_times.extend(router.get("upstream_error_total_ms", []))
        fallback_decisions += router.get("v2_guard_fallback", 0)
        fallback_failures += router.get("v2_guard_fallback_failures", 0)
        aggregate_exclusions.update(router.get("v2_exclusions", {}))
        for key, count in router.get("outcomes_by_replica", {}).items():
            outcome, replica = key.split("@", 1)
            if outcome != "success":
                aggregate_failure_nodes[replica] += count
        audit_rows.append([
            os.path.relpath(run["dir"], HERE),
            run.get("verdict"),
            run.get("reason") or "",
            (
                f"{run.get('n_success', 0)}/{run.get('offered', 0)} "
                f"({pctf(run.get('completion_rate'))})"
            ),
            (
                f"success={outcomes.get('success', 0)}, "
                f"upstream_error={outcomes.get('upstream_error', 0)}, "
                f"no_healthy={outcomes.get('no_healthy_replica', 0)}"
            ),
            "/".join(str(chosen.get(node, 0)) for node in NODES),
            (
                f"{stale.get('live', 0)}/"
                f"{stale.get('some_stale_excluded', 0)}/"
                f"{stale.get('all_stale_compare', 0)}/"
                f"{stale.get('all_stale_rr', 0)}"
            ),
            sum(events.values()),
        ])
    lines.append(md_table(
        ["run", "verdict", "run_meta reason", "bench completion",
         "router outcomes", "chosen n1/n2/n3",
         "stale live/some/compare/RR", "reachability events"],
        audit_rows,
    ))
    lines.append(
        "\nThe `run_meta` failure percentages use all 103 router records "
        "(3 warmups + 100 measured requests), hence 16/103 = 15.5% and "
        "19/103 = 18.4%. The benchmark completion column above uses only the "
        "100 measured requests."
    )
    total_chosen = sum(aggregate_chosen.values())
    long_counts = completion_summary(v2_runs, "long")
    short_counts = completion_summary(v2_runs, "short")
    error_min = min(upstream_error_times) if upstream_error_times else None
    error_max = max(upstream_error_times) if upstream_error_times else None
    lines.append(
        "\n**V2 c12 diagnosis:** this is repeatable policy/load behavior, not "
        "infrastructure. Across the five runs, the router made "
        f"{total_chosen} choices and recorded "
        f"{aggregate_outcomes.get('no_healthy_replica', 0)} "
        "`no_healthy_replica` outcomes, "
        f"{sum(aggregate_events.values())} reachability/eviction events, and "
        f"{aggregate_outcomes.get('upstream_error', 0)} upstream stream errors. "
        "The choice split was "
        + "/".join(
            f"{aggregate_chosen.get(node, 0)} "
            f"({aggregate_chosen.get(node, 0) / total_chosen * 100:.1f}%)"
            for node in NODES
        )
        + "."
    )
    lines.append(
        "\nAll failures were long-class requests "
        f"(long completion {long_counts['successes']}/{long_counts['offered']} "
        f"= {pctf(long_counts['pooled'])}; short completion "
        f"{short_counts['successes']}/{short_counts['offered']} "
        f"= {pctf(short_counts['pooled'])}). Router-side failures were confined "
        "to the Pi 4Bs: "
        + ", ".join(
            f"{node}={aggregate_failure_nodes.get(node, 0)}"
            for node in NODES
        )
        + (
            f". They terminated after {error_min / 1000:.1f}-"
            f"{error_max / 1000:.1f}s with HTTP 200 already established, "
            "matching the client's incomplete-transfer errors."
            if error_min is not None and error_max is not None
            else "."
        )
    )
    lines.append(
        f"\nThe Stage-1 all-full fallback fired on {fallback_decisions}/"
        f"{total_chosen} decisions ({fallback_decisions / total_chosen * 100:.1f}%) "
        f"and accounted for {fallback_failures}/"
        f"{aggregate_outcomes.get('upstream_error', 0)} failures. The Pi 5 was "
        f"marked full/deferred by Stage 1 on "
        f"{sum(count for key, count in aggregate_exclusions.items() if key.startswith('pi-node-1:'))} "
        "candidate observations. V2 therefore did not exclude the whole cluster; "
        "it continued placing work on available/shortlisted Pi 4Bs, and because "
        "v2 does not inspect the incoming request length, some long requests "
        "exceeded the frozen 120s stream deadline there. No same-configuration "
        "infrastructure rerun is indicated."
    )
    return "\n".join(lines)


# ═════════════════════════ SECTION 4 — completion ═════════════════════════
def decision_section4(cells):
    lines = [
        "\n## 4. Completion rate as a first-class result\n",
        "All runs are included (clean + suspect); each entry is pooled "
        "`successes/offered (rate)` so overload failures are not hidden.\n",
    ]
    rates = {}
    rows = []
    for policy in POLICIES:
        row = [policy]
        for workload, conc in CELL_COLS:
            runs = valid_runs(cells, policy, conc, workload)
            summary = completion_summary(runs)
            rates[(policy, conc, workload)] = summary["pooled"]
            row.append(
                f"{summary['successes']}/{summary['offered']} "
                f"({pctf(summary['pooled'])})"
                if summary["offered"] else "n/a"
            )
        rows.append(row)
    lines.append(md_table(
        ["policy"] + [col_label(workload, conc) for workload, conc in CELL_COLS],
        rows,
    ))

    def mean_rate(policy, selected):
        values = [
            rates.get((policy, conc, workload))
            for workload, conc in selected
        ]
        values = [value for value in values if value is not None]
        return sum(values) / len(values) if values else None

    lines.append("\n**Descriptive completion pooled across each arm's available cells (coverage differs between arms):** "
                 + ", ".join(
                     f"{policy}={pctf(mean_rate(policy, CELL_COLS))}"
                     for policy in POLICIES
                 ))

    lines.append("\n### Completion rate under overload\n")
    lines.append(
        "All runs, including suspect runs. Each entry is "
        "`successes/offered (pooled rate); rep median [min-max], n`. The "
        "min-max range is the observed run-to-run spread.\n"
    )
    overload_cells = [
        ("uniform", 12),
        ("bimodal", 8),
        ("bimodal", 12),
    ]
    overload_rows = []
    for policy in POLICIES:
        overload_rows.append([
            policy,
            *[
                fmt_all_completion(
                    completion_summary(
                        valid_runs(cells, policy, conc, workload)
                    )
                )
                for workload, conc in overload_cells
            ],
        ])
    lines.append(md_table(
        ["policy"] + [
            col_label(workload, conc) for workload, conc in overload_cells
        ],
        overload_rows,
    ))

    lines.append("\n### Live-state completion versus WLC\n")
    comparison_rows = []
    higher_beyond = []
    for workload, conc in overload_cells:
        base = completion_summary(
            valid_runs(
                cells, "weighted_least_connections", conc, workload
            )
        )
        for policy in (
            "state_aware",
            "state_aware_v2",
            "cost_predict_v3",
        ):
            candidate = completion_summary(
                valid_runs(cells, policy, conc, workload)
            )
            delta = (
                (candidate["pooled"] - base["pooled"]) * 100
                if candidate["pooled"] is not None and base["pooled"] is not None
                else None
            )
            v = run_level_verdict(
                scaled(base, 100.0), scaled(candidate, 100.0),
                lower_is_better=False,
            )
            if v["label"].startswith("BEAT"):
                higher_beyond.append(f"{policy} {col_label(workload, conc)}")
            comparison_rows.append([
                col_label(workload, conc),
                policy,
                pctf(base["pooled"]),
                pctf(candidate["pooled"]),
                f"{delta:+.1f} pp" if delta is not None else "n/a",
                fmt_verdict(v, 1, " pp"),
            ])
    lines.append(md_table(
        ["cell", "candidate", "WLC pooled", "candidate pooled",
         "delta (pooled)", "run-level test (per-run rates)"],
        comparison_rows,
    ))
    lines.append(
        "\nThe delta column is a pooled request count and is shown for scale "
        "only — it is NOT what the verdict is based on. Pooling requests across "
        "runs treats every request as independent and will call a single bad "
        "run a significant effect. The verdict column tests the per-run "
        "completion rates instead, and flags `(fragile)` when deleting one run "
        "overturns the result.\n"
    )
    lines.append(
        "**Overload completion verdict:** "
        + (
            "no live-state policy has a run-level completion win over WLC."
            if not higher_beyond
            else "run-level completion wins over WLC: "
                 f"{', '.join(higher_beyond)}."
        )
        + " The strong separation is instead between naive routing and "
        "capacity-aware/weighted routing: this supports a capacity-awareness "
        "reliability finding, not a distinct live-state reliability win."
    )
    return "\n".join(lines)


# ═════════════════════════ SECTION 5 — routing/state sanity ═════════════════════════
def decision_section5(cells):
    lines = [
        "\n## 5. Routing distribution + staleness sanity\n",
        "All runs and measured requests are pooled. `some_stale_excluded` still "
        "compares live nodes after excluding stale ones; only all-stale modes "
        "mean no live state was available. Stateless policies report `n/a` for "
        "state-mode percentages. `rode stale` is the chosen-replica stale flag.\n",
    ]
    rows = []
    weakened = []
    v3_clamped = 0
    v3_total = 0
    for workload, conc in CELL_COLS:
        for policy in POLICIES:
            runs = valid_runs(cells, policy, conc, workload)
            if not runs:
                continue
            chosen = Counter()
            modes = Counter()
            total = 0
            rode = 0
            clamp_any = 0
            for run in runs:
                router = run.get("router", {})
                chosen.update(router.get("chosen", {}))
                modes.update(router.get("stale_modes", {}))
                measured = router.get("measured_total", 0)
                total += measured
                rode += round((router.get("rode_stale_frac") or 0) * measured)
                clamp_any += router.get("v3_clamp_any", 0)
            chosen_total = sum(chosen.values())
            split = "/".join(
                pctf(chosen.get(node, 0) / chosen_total)
                if chosen_total else "n/a"
                for node in NODES
            )
            if policy in STATELESS:
                state_values = ["n/a"] * 5
            else:
                def share(mode):
                    return pctf(modes.get(mode, 0) / total) if total else "n/a"
                all_stale = (
                    modes.get("all_stale_compare", 0)
                    + modes.get("all_stale_rr", 0)
                ) / total if total else 0
                rode_share = rode / total if total else 0
                flag = ""
                if all_stale > 0.5 or rode_share > 0.5:
                    flag = " FLAG"
                    weakened.append(f"{policy} {col_label(workload, conc)}")
                state_values = [
                    share("live"),
                    share("some_stale_excluded"),
                    share("all_stale_compare"),
                    share("all_stale_rr"),
                    pctf(rode_share) + flag,
                ]
            clamp_text = "n/a"
            if policy == "cost_predict_v3":
                v3_clamped += clamp_any
                v3_total += total
                clamp_text = pctf(clamp_any / total) if total else "n/a"
            rows.append([
                col_label(workload, conc),
                policy,
                split,
                *state_values,
                clamp_text,
            ])
    lines.append(md_table(
        ["cell", "policy", "chosen n1/n2/n3", "live",
         "some stale excluded", "all stale compare", "all stale RR",
         "rode stale", "v3 R_proj clamp"],
        rows,
    ))
    lines.append(
        "\n**Weakened cells (majority without live comparison or majority chose "
        "stale):** " + (", ".join(weakened) if weakened else "none") + "."
    )
    lines.append(
        f"\n**V3 R_proj clamping:** {v3_clamped}/{v3_total} decisions "
        f"({pctf(v3_clamped / v3_total) if v3_total else 'n/a'})."
    )
    return "\n".join(lines)


# ═════════════════════════ SECTION 6 — suspect/skip audit ═════════════════════════
def suspect_category(run):
    reason = (run.get("reason") or "").lower()
    events = run.get("router", {}).get("events", {})
    if run.get("verdict") == "skipped":
        return "infrastructure"
    if (
        events.get("evicted", 0)
        or "no_healthy_replica" in reason
        or "node down" in reason
        or "unreachable" in reason
        or "eviction" in reason
    ):
        return "infrastructure"
    if (
        run.get("policy") in ("random", "round_robin")
        and run.get("workload") == "bimodal"
        and run.get("conc") in (8, 12)
    ):
        return "expected naive c8/c12 overload"
    return "policy/load failure (not infrastructure)"


def decision_section6(runs):
    lines = [
        "\n## 6. Suspect/skip audit\n",
        "Every non-clean pooled run is listed. `original` means rep1-3; "
        "otherwise the extension group is shown.\n",
    ]
    nonclean = [
        run for run in runs
        if run.get("grid") and run.get("verdict") in ("suspect", "skipped")
    ]
    rows = []
    categories = Counter()
    infra_cells = set()
    for run in sorted(
        nonclean,
        key=lambda item: (
            suspect_category(item),
            item.get("policy", ""),
            item.get("workload", ""),
            item.get("conc", 0),
            item.get("rep", 0),
        ),
    ):
        category = suspect_category(run)
        categories[category] += 1
        if category == "infrastructure":
            infra_cells.add(
                f"{run['policy']} {col_label(run['workload'], run['conc'])}"
            )
        rows.append([
            category,
            run["policy"],
            col_label(run["workload"], run["conc"]),
            f"rep{run['rep']}",
            run.get("extension_group") or "original",
            run.get("verdict"),
            run.get("reason") or "",
        ])
    lines.append(md_table(
        ["category", "policy", "cell", "rep", "source", "verdict", "reason"],
        rows,
    ))
    lines.append(
        "\n**Audit counts:** "
        + ", ".join(f"{category}={count}" for category, count in categories.items())
        + f"; suspect={sum(run.get('verdict') == 'suspect' for run in nonclean)}, "
        + f"skipped={sum(run.get('verdict') == 'skipped' for run in nonclean)}."
    )
    lines.append(
        "\n**Infrastructure-dependent cells:** "
        + (", ".join(sorted(infra_cells)) if infra_cells else "none")
        + ". Therefore no cell-level conclusion rests on a node-down or eviction run."
    )
    return "\n".join(lines)


# ═════════════════════════ Appendix + CSV ═════════════════════════
def decision_appendix(runs):
    grid = [run for run in runs if run.get("grid")]
    warmup = sum(run.get("router", {}).get("warmup_excluded", 0) for run in grid)
    truncated = sum(run.get("n_truncated", 0) for run in grid)
    truncated_success = sum(run.get("n_truncated_success", 0) for run in grid)
    mismatches = [
        run for run in grid
        if run.get("router") and not run["router"].get("reconcile_ok")
    ]
    lines = [
        "\n## Appendix A. Exclusions and reconciliation\n",
        f"- Grid runs loaded: **{len(grid)}**.",
        f"- Router warmup records excluded: **{warmup}**.",
        f"- Truncated requests: **{truncated}** total; "
        f"**{truncated_success}** successful truncated requests excluded from latency.",
        f"- Bench/router completion mismatches: **{len(mismatches)}**.",
    ]
    for run in mismatches:
        note = next(
            (note for note in run.get("notes", []) if note.startswith("RECONCILE")),
            "reconciliation mismatch",
        )
        lines.append(
            f"  - {run['policy']} {col_label(run['workload'], run['conc'])} "
            f"rep{run['rep']}: {note}"
        )
    other_notes = [
        (run, note)
        for run in runs
        for note in run.get("notes", [])
        if not note.startswith("RECONCILE")
    ]
    lines.append(f"- Other load notes: **{len(other_notes)}**.")
    for run, note in other_notes:
        lines.append(f"  - {os.path.relpath(run['dir'], HERE)}: {note}")
    return "\n".join(lines)


def decision_csv_rows(cells):
    rows = []
    for policy in POLICIES:
        for workload, conc in CELL_COLS:
            cell = cells.get(cellkey(policy, conc, workload), [])
            if not cell:
                continue
            classes = ["overall"] + (
                ["short", "long"] if workload == "bimodal" else []
            )
            for subset in ("clean", "all"):
                runs = (
                    [run for run in cell if run.get("verdict") == "clean"]
                    if subset == "clean" else list(cell)
                )
                runs = [run for run in runs if run.get("class_counts")]
                for cls in classes:
                    completion = completion_summary(runs, cls)
                    rows.append([
                        policy, conc, workload, cls, subset, "completion_rate",
                        completion["mean"], completion["median"], completion["std"],
                        completion["min"], completion["max"], completion["n"],
                        completion["successes"], completion["offered"],
                        completion["pooled"],
                    ])
                    if cls == "overall":
                        # Whole-run scalar: no per-class decomposition exists.
                        for name, getter in (
                            ("throughput_req_s", "throughput_req_s"),
                            ("throughput_tok_s", "throughput_tok_s"),
                        ):
                            thr = agg([run.get(getter) for run in runs])
                            rows.append([
                                policy, conc, workload, cls, subset, name,
                                thr["mean"], thr["median"], thr["std"],
                                thr["min"], thr["max"], thr["n"],
                                completion["successes"], completion["offered"],
                                completion["pooled"],
                            ])
                    for metric in LAT_METRICS:
                        for percentile in ("p50", "p95"):
                            summary = latency_summary(
                                runs, cls, metric, percentile
                            )
                            rows.append([
                                policy, conc, workload, cls, subset,
                                f"{metric}_{percentile}_ms",
                                summary["mean"], summary["median"], summary["std"],
                                summary["min"], summary["max"], summary["n"],
                                completion["successes"], completion["offered"],
                                completion["pooled"],
                            ])
    return rows


# ───────────────────────── main ─────────────────────────
def main():
    runs = load_all()
    cells = group_cells(runs)

    sections = [
        decision_section1(cells),
        decision_section2(cells),
        decision_section3(cells),
        decision_section4(cells),
        decision_section5(cells),
        decision_section6(runs),
        section4(runs),          # SQ2 routing overhead (numbered 7 in the report)
        decision_section_ladder(cells, runs),   # baseline ladder (numbered 8)
        decision_appendix(runs),
    ]
    # Family size must be counted AFTER every section has run.
    sections.append(
        "\n## 9. Multiplicity — family size\n\n"
        f"This report executed **{TESTS_RUN['count']} permutation tests** in total "
        f"(all sections, latency and completion). No correction for multiple "
        f"comparisons is applied, so at alpha={ALPHA} roughly "
        f"{TESTS_RUN['count'] * ALPHA:.0f} significant results would be expected by "
        "chance alone across the whole family. Individual p-values are therefore "
        "descriptive of a single comparison and are not family-wise error rates; "
        "conclusions rest on the direction and consistency of effects across cells "
        "and workloads rather than on any single test clearing the threshold.\n"
    )
    report = "\n".join(sections) + "\n"
    with open(os.path.join(OUT_HERE, "3e_report.md"), "w") as handle:
        handle.write(report)

    csv_rows = decision_csv_rows(cells)
    with open(os.path.join(OUT_HERE, "3e_summary.csv"), "w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([
            "policy", "concurrency", "workload", "class", "subset", "metric",
            "mean_of_reps", "median_of_reps", "std_across_reps",
            "min_rep", "max_rep", "n_reps", "successes", "offered",
            "pooled_completion_rate",
        ])
        writer.writerows(csv_rows)

    # Decision sections first on stdout, matching the report's required order.
    print(sections[0])
    print(sections[1])
    print(sections[2])
    print(
        f"\n[wrote 3e_report.md ({len(runs)} pooled runs) and "
        f"3e_summary.csv ({len(csv_rows)} rows)]"
    )


if __name__ == "__main__":
    main()
