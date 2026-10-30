"""Decision replay: how often argmin(B/w) and argmin(R/w) pick different replicas.

Uses only the state recorded at each decision, so it describes disagreement on
observed snapshots, not the outcome of running the other policy.
"""
import csv
from scope import included_paths, CELLS
import glob
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # data root: 3e_results
OUT_HERE = os.path.dirname(os.path.abspath(__file__))  # outputs: everythingcombined

# Replica order as declared in router_3nodes.toml (also the argmin tie-break order).
NODES = ["pi-node-1", "pi-node-2", "pi-node-3"]

WORKLOADS = ["uniform", "bimodal", "sharegpt"]
CONCS = [4, 8, 12]
CELL_COLS = CELLS

_WLAB = {"uniform": "uni", "bimodal": "bim", "sharegpt": "sgpt"}


def col_label(w, c):
    return f"c{c}_{_WLAB[w]}"


# ───────────────────────── stats helpers (mirrors analyze_3e.py) ──────────
def median(xs):
    xs = sorted(x for x in xs if isinstance(x, (int, float)))
    n = len(xs)
    if n == 0:
        return None
    if n % 2:
        return float(xs[n // 2])
    return (xs[n // 2 - 1] + xs[n // 2]) / 2.0


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


def agg(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    n = len(vals)
    if n == 0:
        return {"mean": None, "median": None, "std": None, "n": 0}
    mean = sum(vals) / n
    std = (sum((v - mean) ** 2 for v in vals) / (n - 1)) ** 0.5 if n >= 2 else 0.0
    return {"mean": mean, "median": median(vals), "std": std, "n": n}


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def meanspread(a, nd=3):
    if not a or a["n"] == 0:
        return "n/a"
    if a["n"] == 1:
        return f"{a['mean']:.{nd}f}(n=1)"
    return f"{a['mean']:.{nd}f}±{a['std']:.{nd}f}"


# ───────────────────────── argmin_over, replicated exactly ────────────────
def argmin_over(candidates, score_fn):
    """candidates: iterable of canonical indices, ascending. Replicates
    policy.rs's argmin_over: strict `<` replace (first minimum at the lowest
    index wins ties), NaN scores skipped."""
    best_i, best_s = None, None
    for i in candidates:
        s = score_fn(i)
        if s is None or s != s:  # NaN check without importing math
            continue
        if best_s is None or s < best_s:
            best_i, best_s = i, s
    return best_i


# ───────────────────────── per-decision counterfactual ────────────────────
def analyze_run(meta, router_log_path):
    """Returns a dict of run-level aggregates, or None if unusable."""
    weights = (meta.get("frozen_config") or {}).get("weights")
    if not weights or len(weights) != 3:
        return None
    weight_by_node = dict(zip(NODES, weights))

    n_analyzed = 0
    n_skipped = 0
    div_router = []       # 1/0 per analyzed decision
    div_uncensored = []
    tie_router = []
    tie_uncensored = []
    mag_router = []       # magnitude only where div_router==1 and defined
    mag_uncensored = []

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
            sv = r.get("state_vector")
            if not sv or not isinstance(sv, list):
                n_skipped += 1
                continue

            by_name = {e.get("replica"): e for e in sv}
            if not all(n in by_name for n in NODES):
                n_skipped += 1
                continue

            # Strict: ALL 3 configured nodes must be simultaneously eligible
            # and not-stale, or this decision is skipped entirely (see
            # docstring -- no partial-cluster comparisons).
            if not all(
                by_name[n].get("eligible") is True and by_name[n].get("stale") is False
                for n in NODES
            ):
                n_skipped += 1
                continue

            b_score = {n: by_name[n].get("b_metrics") for n in NODES}
            deferred_score = {
                n: (by_name[n].get("b_metrics") or 0.0) + (by_name[n].get("requests_deferred") or 0.0)
                for n in NODES
            }
            u_score = {n: by_name[n].get("u") for n in NODES}
            rw_score = {}
            for n in NODES:
                u = u_score[n]
                w = weight_by_node.get(n)
                if u is None or w is None or w == 0:
                    rw_score[n] = None
                else:
                    rw_score[n] = u / w

            idx_of = {n: i for i, n in enumerate(NODES)}

            cand_b = argmin_over(range(3), lambda i: b_score[NODES[i]])
            cand_b_unc = argmin_over(range(3), lambda i: deferred_score[NODES[i]])
            cand_rw = argmin_over(range(3), lambda i: rw_score[NODES[i]])

            if cand_b is None or cand_b_unc is None or cand_rw is None:
                n_skipped += 1
                continue

            n_analyzed += 1

            # b_tie_rate: 2+ nodes share the minimum b-score.
            min_b = min(b_score[n] for n in NODES)
            tie_router.append(1 if sum(1 for n in NODES if b_score[n] == min_b) >= 2 else 0)
            min_bu = min(deferred_score[n] for n in NODES)
            tie_uncensored.append(
                1 if sum(1 for n in NODES if deferred_score[n] == min_bu) >= 2 else 0
            )

            d_router = 1 if cand_b != cand_rw else 0
            d_unc = 1 if cand_b_unc != cand_rw else 0
            div_router.append(d_router)
            div_uncensored.append(d_unc)

            if d_router:
                a, b = rw_score[NODES[cand_b]], rw_score[NODES[cand_rw]]
                if a is not None and b is not None and b > 0:
                    mag_router.append(a / b)
            if d_unc:
                a, b = rw_score[NODES[cand_b_unc]], rw_score[NODES[cand_rw]]
                if a is not None and b is not None and b > 0:
                    mag_uncensored.append(a / b)

    if n_analyzed == 0 and n_skipped == 0:
        return None

    return {
        "n_analyzed": n_analyzed,
        "n_skipped": n_skipped,
        "divergence_rate_router": (sum(div_router) / n_analyzed) if n_analyzed else None,
        "divergence_rate_uncensored": (sum(div_uncensored) / n_analyzed) if n_analyzed else None,
        "b_tie_rate_router": (sum(tie_router) / n_analyzed) if n_analyzed else None,
        "b_tie_rate_uncensored": (sum(tie_uncensored) / n_analyzed) if n_analyzed else None,
        "magnitude_median_router": median(mag_router),
        "magnitude_median_uncensored": median(mag_uncensored),
    }


# ───────────────────────── result-tree walk (mirrors analyze_3e.py) ───────
def load_all_runs():
    """One entry per rep folder: run_meta + resolved arm + run-level
    divergence aggregates, or a skip reason if unusable."""
    out = []
    for mp in [str(p) for p in included_paths()]:
        rd = os.path.dirname(mp)
        try:
            meta = json.loads(open(mp).read())
        except Exception as e:  # noqa: BLE001
            out.append({"dir": rd, "usable": False, "reason": f"run_meta load error: {e}"})
            continue

        arm = meta.get("arm") or meta.get("policy")
        workload = meta.get("workload")
        conc = meta.get("concurrency")
        verdict = meta.get("verdict")
        router_file = meta.get("router_log_file")

        if not arm or not workload or not conc or not router_file:
            out.append({"dir": rd, "usable": False, "reason": "incomplete run_meta"})
            continue

        router_path = os.path.join(rd, router_file)
        if not os.path.exists(router_path):
            out.append({"dir": rd, "usable": False, "reason": "missing router_log_file"})
            continue

        try:
            result = analyze_run(meta, router_path)
        except Exception as e:  # noqa: BLE001
            out.append({"dir": rd, "usable": False, "reason": f"analysis error: {e}"})
            continue

        if result is None:
            out.append({"dir": rd, "usable": False, "reason": "no usable decisions"})
            continue

        out.append({
            "dir": rd, "usable": True, "arm": arm, "workload": workload,
            "conc": conc, "verdict": verdict, **result,
        })
    return out


def cellkey(arm, c, w):
    return (arm, c, w)


def group_cells(runs, clean):
    g = defaultdict(list)
    for r in runs:
        if not r.get("usable"):
            continue
        is_clean = r.get("verdict") == "clean"
        if is_clean != clean:
            continue
        g[cellkey(r["arm"], r["conc"], r["workload"])].append(r)
    return g


def cell_row(arm, c, w, cell_runs):
    div_r = agg([r["divergence_rate_router"] for r in cell_runs
                 if r["divergence_rate_router"] is not None])
    div_u = agg([r["divergence_rate_uncensored"] for r in cell_runs
                 if r["divergence_rate_uncensored"] is not None])
    tie_r = agg([r["b_tie_rate_router"] for r in cell_runs
                 if r["b_tie_rate_router"] is not None])
    tie_u = agg([r["b_tie_rate_uncensored"] for r in cell_runs
                 if r["b_tie_rate_uncensored"] is not None])
    mag_r_vals = [r["magnitude_median_router"] for r in cell_runs
                  if r["magnitude_median_router"] is not None]
    mag_u_vals = [r["magnitude_median_uncensored"] for r in cell_runs
                  if r["magnitude_median_uncensored"] is not None]
    n_analyzed = sum(r["n_analyzed"] for r in cell_runs)
    n_skipped = sum(r["n_skipped"] for r in cell_runs)

    return {
        "arm": arm, "conc": c, "workload": w, "n_runs": len(cell_runs),
        "n_analyzed": n_analyzed, "n_skipped": n_skipped,
        "divergence_rate_router": div_r, "divergence_rate_uncensored": div_u,
        "b_tie_rate_router": tie_r, "b_tie_rate_uncensored": tie_u,
        "magnitude_median_router": median(mag_r_vals), "magnitude_p90_router": pct(mag_r_vals, 90),
        "magnitude_median_uncensored": median(mag_u_vals), "magnitude_p90_uncensored": pct(mag_u_vals, 90),
        "magnitude_n_runs_router": len(mag_r_vals), "magnitude_n_runs_uncensored": len(mag_u_vals),
    }


def all_cell_rows(runs, clean):
    grouped = group_cells(runs, clean)
    arms = sorted({k[0] for k in grouped})
    rows = []
    for arm in arms:
        for w, c in CELL_COLS:
            key = cellkey(arm, c, w)
            cell_runs = grouped.get(key, [])
            if cell_runs:
                rows.append(cell_row(arm, c, w, cell_runs))
            else:
                rows.append({"arm": arm, "conc": c, "workload": w, "n_runs": 0,
                             "n_analyzed": 0, "n_skipped": 0, "empty": True})
    return rows


def main():
    print("=" * 100)
    print("§3e per-decision counterfactual: argmin(B) vs argmin(R/w) on OBSERVED decision-time state.")
    print("NOT a prediction of alternative-policy outcomes -- see script docstring.")
    print("requests_deferred WAS available in every logged decision (proxy.rs always logs it);")
    print("'router' columns replicate actual b_metrics-only scoring, 'uncensored' columns add it.")
    print("=" * 100)

    runs = load_all_runs()
    usable = [r for r in runs if r.get("usable")]
    unusable = [r for r in runs if not r.get("usable")]
    print(f"\n{len(runs)} run directories found. {len(usable)} usable, {len(unusable)} not usable.")
    if unusable:
        reasons = defaultdict(int)
        for r in unusable:
            reasons[r["reason"]] += 1
        for reason, n in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"  {n:>4}  {reason}")

    n_clean = sum(1 for r in usable if r.get("verdict") == "clean")
    n_not_clean = sum(1 for r in usable if r.get("verdict") != "clean")
    print(f"\nOf usable runs: {n_clean} verdict==clean, {n_not_clean} verdict!=clean.")
    print("Reported below as two SEPARATE blocks -- never pooled.\n")

    headers = ["arm", "cell", "n_runs", "analyzed", "skipped",
               "div_rate(router)", "div_rate(uncens)",
               "b_tie(router)", "b_tie(uncens)",
               "mag_median(router)", "mag_p90(router)",
               "mag_median(uncens)", "mag_p90(uncens)"]

    csv_rows = []
    for clean, label in ((True, "VERDICT == CLEAN"), (False, "VERDICT != CLEAN (includes overload-collapse cells)")):
        print("#" * 100)
        print(f"# {label}")
        print("#" * 100)
        rows = all_cell_rows(runs, clean)
        md_rows = []
        for r in rows:
            cell = col_label(r["workload"], r["conc"])
            if r.get("empty"):
                md_rows.append([r["arm"], cell, 0, 0, 0, "empty", "empty", "empty", "empty",
                                 "empty", "empty", "empty", "empty"])
                csv_rows.append({
                    "clean": clean, "arm": r["arm"], "cell": cell, "n_runs": 0,
                    "n_analyzed": 0, "n_skipped": 0,
                    "divergence_rate_router_mean": "", "divergence_rate_router_std": "",
                    "divergence_rate_uncensored_mean": "", "divergence_rate_uncensored_std": "",
                    "b_tie_rate_router_mean": "", "b_tie_rate_router_std": "",
                    "b_tie_rate_uncensored_mean": "", "b_tie_rate_uncensored_std": "",
                    "magnitude_median_router": "", "magnitude_p90_router": "",
                    "magnitude_median_uncensored": "", "magnitude_p90_uncensored": "",
                })
                continue
            md_rows.append([
                r["arm"], cell, r["n_runs"], r["n_analyzed"], r["n_skipped"],
                meanspread(r["divergence_rate_router"]), meanspread(r["divergence_rate_uncensored"]),
                meanspread(r["b_tie_rate_router"]), meanspread(r["b_tie_rate_uncensored"]),
                f"{r['magnitude_median_router']:.2f}" if r["magnitude_median_router"] is not None else "n/a",
                f"{r['magnitude_p90_router']:.2f}" if r["magnitude_p90_router"] is not None else "n/a",
                f"{r['magnitude_median_uncensored']:.2f}" if r["magnitude_median_uncensored"] is not None else "n/a",
                f"{r['magnitude_p90_uncensored']:.2f}" if r["magnitude_p90_uncensored"] is not None else "n/a",
            ])
            dr = r["divergence_rate_router"]
            du = r["divergence_rate_uncensored"]
            tr = r["b_tie_rate_router"]
            tu = r["b_tie_rate_uncensored"]
            csv_rows.append({
                "clean": clean, "arm": r["arm"], "cell": cell, "n_runs": r["n_runs"],
                "n_analyzed": r["n_analyzed"], "n_skipped": r["n_skipped"],
                "divergence_rate_router_mean": dr["mean"], "divergence_rate_router_std": dr["std"],
                "divergence_rate_uncensored_mean": du["mean"], "divergence_rate_uncensored_std": du["std"],
                "b_tie_rate_router_mean": tr["mean"], "b_tie_rate_router_std": tr["std"],
                "b_tie_rate_uncensored_mean": tu["mean"], "b_tie_rate_uncensored_std": tu["std"],
                "magnitude_median_router": r["magnitude_median_router"],
                "magnitude_p90_router": r["magnitude_p90_router"],
                "magnitude_median_uncensored": r["magnitude_median_uncensored"],
                "magnitude_p90_uncensored": r["magnitude_p90_uncensored"],
            })
        print(md_table(headers, md_rows))
        print()

    out_csv = os.path.join(OUT_HERE, "divergence_analysis.csv")
    fieldnames = list(csv_rows[0].keys()) if csv_rows else []
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(csv_rows)
    print(f"CSV written: {out_csv}")


if __name__ == "__main__":
    main()
