"""Sensitivity checks: deadline failures counted at 120 s, and short/long class comparisons.

Run: python -B 3e_results/analysis/sq3/b/completion_and_class_checks.py
"""
import csv
import json
import math
import sys
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from scope import included_paths  # noqa: E402
from analyze_3e import col_label, metric_ms, pct, perm_test, truthy_trunc  # noqa: E402

SA, WLC, WRR = "state_aware", "weighted_least_connections", "weighted_round_robin"
BASES = (WLC, WRR)
DEADLINE_MS = 120_000.0
DEADLINE_FLOOR_MS = 119_000.0   # router total_ms for requests cut at the cap
CLASS_METRICS = ("e2el_p50", "e2el_p95", "tpot_p50", "tpot_p95", "completion")


def load_runs():
    """Per-run request records for the three central policies."""
    runs = []
    for p in included_paths():
        m = json.loads(p.read_text())
        arm = m.get("arm") or m["policy"]
        if arm not in (SA, WLC, WRR):
            continue
        bench = json.loads((p.parent / m["bench_result_file"]).read_text())
        raw = bench["raw"]
        num_warmup = (m.get("bench_summary") or {}).get("num_warmup", 3)
        router = [json.loads(line) for line in (p.parent / m["router_log_file"]).open()
                  if line.strip()]
        router = [r for r in router if r.get("type") == "request"
                  and isinstance(r.get("request_id"), int) and r["request_id"] >= num_warmup]
        cut_router = sum(1 for r in router if r.get("outcome") != "success"
                         and (r.get("total_ms") or 0) >= DEADLINE_FLOOR_MS)
        cut_bench = sum(1 for r in raw if not r["success"] and r.get("outcome") == "connection_error")
        # The two logs must agree on which failures were deadline cuts.
        assert cut_router == cut_bench, (p, cut_router, cut_bench)
        runs.append(dict(arm=arm, cell=col_label(m["workload"], m["concurrency"]),
                         workload=m["workload"], concurrency=int(m["concurrency"]),
                         clean=m["verdict"] == "clean", raw=raw, deadline_cut=cut_bench,
                         other_failed=sum(1 for r in raw if not r["success"]) - cut_bench))
    return runs


def ok_values(raw, stem, cls=None):
    return [metric_ms(r, stem) for r in raw
            if r["success"] and not truthy_trunc(r.get("truncated"))
            and (cls is None or r.get("class") == cls)
            and isinstance(metric_ms(r, stem), (int, float))]


def bh(ps):
    n = len(ps)
    order = sorted(range(n), key=lambda i: ps[i])
    out, prev = [0.0] * n, 1.0
    for rank in range(n, 0, -1):
        i = order[rank - 1]
        prev = min(prev, ps[i] * n / rank, 1.0)
        out[i] = prev
    return out


def write(name, rows):
    path = HERE / name
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{path.relative_to(ROOT)}: {len(rows)} rows")


def completion_bound(runs):
    corrected = [r for r in csv.DictReader((HERE / "with_correction/sq3b_bh_corrected.csv").open())
                 if r["arm"] == SA and r["base_arm"] in BASES
                 and r["metric"] in ("e2el_p50", "e2el_p95") and r["clears_alpha_B"] == "True"]
    published = {}
    for base, tag in ((WLC, "wlc"), (WRR, "wrr")):
        for r in csv.DictReader((HERE / f"sq3b_vs_{tag}.csv").open()):
            if r["arm"] == SA:
                published[base, r["cell"], r["metric"]] = float(r["diff"])
    rows = []
    for c in corrected:
        base, cell, metric = c["base_arm"], c["cell"], c["metric"]
        q = 50 if metric == "e2el_p50" else 95
        group = {arm: [r for r in runs if r["arm"] == arm and r["cell"] == cell and r["clean"]]
                 for arm in (base, SA)}
        vals = {}
        for arm, rs in group.items():
            reported = [pct(ok_values(r["raw"], "e2el"), q) for r in rs]
            bounded = [pct(ok_values(r["raw"], "e2el") + [DEADLINE_MS] * r["deadline_cut"], q)
                       for r in rs]
            vals[arm] = (reported, bounded)
        d_rep, p_rep, _, _ = perm_test(vals[base][0], vals[SA][0])
        d_bnd, p_bnd, _, _ = perm_test(vals[base][1], vals[SA][1])
        # The reported difference must reproduce the published one exactly.
        assert math.isclose(d_rep, published[base, cell, metric], abs_tol=1e-6)
        rows.append(dict(
            base_arm=base, cand_arm=SA, cell=cell, metric=metric,
            n_base=len(group[base]), n_cand=len(group[SA]),
            deadline_cut_base=sum(r["deadline_cut"] for r in group[base]),
            deadline_cut_cand=sum(r["deadline_cut"] for r in group[SA]),
            other_failed_base=sum(r["other_failed"] for r in group[base]),
            other_failed_cand=sum(r["other_failed"] for r in group[SA]),
            diff_reported_s=round(d_rep / 1000, 4), diff_bound_s=round(d_bnd / 1000, 4),
            p_raw_reported=p_rep, p_raw_bound=p_bnd, p_adj_B_reported=c["p_adj_B"],
            direction_preserved=(d_rep > 0) == (d_bnd > 0) and d_bnd != 0))
    rows.sort(key=lambda r: (r["base_arm"], r["metric"], r["cell"]))
    write("sq3b_completion_bound.csv", rows)
    return rows


def class_checks(runs):
    values = {}
    for r in runs:
        if r["workload"] != "bimodal":
            continue
        for cls in ("short", "long"):
            reqs = [x for x in r["raw"] if x.get("class") == cls]
            key = (r["arm"], r["concurrency"], cls)
            done = sum(bool(x["success"]) for x in reqs)
            values.setdefault(key + ("completion",), []).append(100 * done / len(reqs))
            values.setdefault(key + ("counts",), []).append((done, len(reqs)))
            if not r["clean"]:
                continue
            for stem in ("e2el", "tpot"):
                xs = ok_values(r["raw"], stem, cls)
                if xs:
                    for q in (50, 95):
                        values.setdefault(key + (f"{stem}_p{q}",), []).append(pct(xs, q))
    rows = []
    for base in BASES:
        for conc in (4, 8, 12):
            for cls in ("short", "long"):
                for metric in CLASS_METRICS:
                    a, b = values[base, conc, cls, metric], values[SA, conc, cls, metric]
                    diff, p, _, _ = perm_test(a, b)
                    unit = "pp" if metric == "completion" else "ms"
                    rows.append(dict(base_arm=base, cand_arm=SA, cell=f"c{conc}_bim",
                                     request_class=cls, metric=metric, unit=unit,
                                     n_base=len(a), n_cand=len(b),
                                     base_median=median(a), cand_median=median(b),
                                     diff=diff, p_raw=p))
    for row, adj in zip(rows, bh([r["p_raw"] for r in rows])):
        row["p_adj_bh60"] = adj
        row["survives_bh60"] = adj < 0.05
    assert len(rows) == 60
    write("sq3b_class_checks.csv", rows)
    return rows


def class_composition(runs):
    rows = []
    for arm in (SA, WLC, WRR):
        for conc in (4, 8, 12):
            rs = [r for r in runs if r["arm"] == arm and r["workload"] == "bimodal"
                  and r["concurrency"] == conc]
            row = dict(arm=arm, cell=f"c{conc}_bim", runs=len(rs),
                       clean_runs=sum(r["clean"] for r in rs))
            for cls in ("short", "long"):
                reqs = [x for r in rs for x in r["raw"] if x.get("class") == cls]
                row[f"{cls}_completed"] = sum(bool(x["success"]) for x in reqs)
                row[f"{cls}_offered"] = len(reqs)
            middle = {"short": 0, "long": 0, "mixed": 0}
            for r in rs:
                if not r["clean"]:
                    continue
                done = sorted((x for x in r["raw"] if x["success"] and not truthy_trunc(x.get("truncated"))),
                              key=lambda x: x["e2el_ms"])
                n = len(done)
                classes = {done[(n - 1) // 2].get("class"), done[n // 2].get("class")}
                middle[classes.pop() if len(classes) == 1 else "mixed"] += 1
            row.update({f"median_position_{k}_runs": v for k, v in middle.items()})
            rows.append(row)
    write("sq3b_class_composition.csv", rows)
    return rows


def main():
    runs = load_runs()
    bound = completion_bound(runs)
    classes = class_checks(runs)
    composition = class_composition(runs)
    print(f"completion bound: {len(bound)} significant latency results; "
          f"{sum(not r['direction_preserved'] for r in bound)} change direction")
    print(f"class checks: {sum(r['p_raw'] < 0.05 for r in classes)} raw p < 0.05; "
          f"{sum(r['survives_bh60'] for r in classes)} survive BH over 60; "
          f"smallest adjusted p = {min(r['p_adj_bh60'] for r in classes):.3f}")
    failures = {cls: sum(r[f"{cls}_offered"] - r[f"{cls}_completed"] for r in composition)
                for cls in ("short", "long")}
    print(f"bimodal failures by class (three central policies): {failures}")


if __name__ == "__main__":
    main()
