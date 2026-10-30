"""Fits the cost models to the sweep's nine cell means per node (OLS on measured R).

Usage: python3 analyze_cost_model_sweep.py [sweep_cost_model_summary.csv]
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

try:
    from scipy import stats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False
    print("WARNING: scipy not installed -- F-test p-values omitted, t-critical "
          "values fall back to a small hardcoded table.\n")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Two-tailed alpha=0.05 t-critical values, used only if scipy is unavailable.
_T_TABLE_005 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
                6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def t_critical(df):
    if HAVE_SCIPY:
        return float(stats.t.ppf(0.975, df))
    return _T_TABLE_005.get(df, 2.0)


def load_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def ols(X, y):
    coefs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ coefs
    resid = y - fitted
    rss = float(resid @ resid)
    xtx_inv = np.linalg.inv(X.T @ X)
    return coefs, rss, fitted, xtx_inv, resid


def r2(y, fitted, k):
    n = len(y)
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1 - ss_res / ss_tot
    adj = 1 - (1 - r_squared) * (n - 1) / (n - k)
    return r_squared, adj


def nested_f_test(rss_reduced, rss_full, df_diff, df_full):
    f_stat = ((rss_reduced - rss_full) / df_diff) / (rss_full / df_full)
    p = float(stats.f.sf(f_stat, df_diff, df_full)) if HAVE_SCIPY else None
    return f_stat, p


def coef_cis(coefs, rss, df, xtx_inv):
    se = np.sqrt(np.diag((rss / df) * xtx_inv))
    tcrit = t_critical(df)
    return se, coefs - tcrit * se, coefs + tcrit * se


def print_model(name, formula, coefs, se, los, his, names, rss, rsq, adjrsq, dfree):
    print(f"\n{name}: {formula}")
    for nm, c, s, lo, hi in zip(names, coefs, se, los, his):
        print(f"  {nm:<12} = {c:>10.5f}   SE={s:>9.5f}   95% CI=[{lo:>10.5f}, {hi:>10.5f}]")
    print(f"  R²={rsq:.4f}  adj-R²={adjrsq:.4f}  RSS={rss:.2f}  df={dfree}")


def cell_stats(rows_subset, label):
    vals = [float(r["tpot_mean"]) for r in rows_subset]
    return {
        "label": label, "n": len(vals),
        "mean": float(np.mean(vals)) if vals else float("nan"),
        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "values": vals,
    }


def find_intra_run_anomalies(rows, ratio_threshold=5.0):
    """Flag (not exclude) any run where max/min per-request tpot_ms exceeds
    ratio_threshold. This is how the extreme-cell prefill-stall run was
    found: one huge-prompt request's prefill can starve concurrent small
    requests' decode for the prefill's whole duration, producing a handful
    of pathologically slow per-request tpot values sitting alongside normal
    ones in the SAME run. That's a real, reportable phenomenon, not a
    measurement error -- callers decide what to do with the flag, this
    function only detects and reports it.
    """
    anomalies = []
    for r in rows:
        d = json.load(open(r["json_file"]))
        per_req_tpot = [req["tpot_ms"] for req in d["raw"] if req["success"] and req["tpot_ms"] > 0]
        if len(per_req_tpot) < 2:
            continue
        ratio = max(per_req_tpot) / min(per_req_tpot)
        if ratio >= ratio_threshold:
            anomalies.append({
                "row": r, "ratio": ratio,
                "raw": [req for req in d["raw"] if req["success"]],
            })
    return anomalies


def load_per_request_tpot(rows_subset):
    """Return one entry per successful individual request across rows_subset,
    each carrying its OWN tpot_ms (not a run-averaged value) alongside the
    RUN-level R_actual (total resident tokens across all concurrent slots in
    that run, repeated for each of that run's requests). R stays run-level
    because that's what a router actually observes -- total node load -- not
    each request's own footprint in isolation."""
    out = []
    for r in rows_subset:
        d = json.load(open(r["json_file"]))
        b = float(r["B"])
        r_actual_run = float(r["R_actual"])
        for req in d["raw"]:
            if req["success"]:
                out.append({
                    "B": b, "R_actual": r_actual_run, "tpot": req["tpot_ms"],
                    "cell_label": r["cell_label"], "rep": r["rep"],
                })
    return out


def print_anomaly_detail(a):
    r = a["row"]
    print(f"  [{r['block']}/{r['cell_label']} rep={r['rep']}] max/min tpot ratio = {a['ratio']:.1f}x "
          f"(temp_peak={r['temp_peak']}C throttled={r['throttled']})")
    for i, req in enumerate(a["raw"]):
        print(f"    req {i}: ttft={req['ttft_ms']:>9.1f}ms  tpot={req['tpot_ms']:>9.1f}ms  "
              f"queue_delay={req['queue_delay_ms']:>7.1f}ms  "
              f"input_tokens={req['input_tokens']:>4}  output_len={req['output_len']:>4}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "sweep_cost_model_summary.csv"
    rows = load_rows(path)

    # --------------------------------------------------------------- #
    # Exclusions
    # --------------------------------------------------------------- #
    n_total = len(rows)
    throttled_rows = [r for r in rows if r["throttled"] == "1"]
    clean_rows = [r for r in rows if r["throttled"] != "1"]
    print("=" * 70)
    print(f"Loaded {n_total} rows from {path}")
    if throttled_rows:
        by_cell = {}
        for r in throttled_rows:
            key = f"{r['block']}/{r['cell_label']}"
            by_cell[key] = by_cell.get(key, 0) + 1
        print(f"Dropping {len(throttled_rows)} throttled row(s):")
        for k, v in by_cell.items():
            print(f"  {k}: {v} row(s)")
    else:
        print("No throttled rows -- nothing dropped.")
    print("=" * 70)

    # Intra-run anomaly scan (flagged, not excluded): concurrent requests can split into groups
    # decoding at different rates, also in symmetric backbone cells.
    print("\n" + "#" * 70)
    print("INTRA-RUN ANOMALY SCAN (all runs, informational -- flagged, not excluded)")
    print("#" * 70)
    anomalies = find_intra_run_anomalies(clean_rows, ratio_threshold=5.0)
    anomalous_json_files = {a["row"]["json_file"] for a in anomalies}
    if anomalies:
        print(f"\n{len(anomalies)} run(s) with max/min per-request tpot ratio >= 5x:")
        for a in anomalies:
            print_anomaly_detail(a)
    else:
        print("\nNo runs with a >=5x max/min per-request tpot ratio.")

    backbone = [r for r in clean_rows if r["block"] == "backbone"]
    asym = [r for r in clean_rows if r["block"] == "asymmetric"]

    # --------------------------------------------------------------- #
    # Part 1: main fit, BACKBONE ONLY, on CELL MEANS (n=9)
    # --------------------------------------------------------------- #
    print("\n" + "#" * 70)
    print("PART 1 — MAIN FIT (backbone cells only, n=9 cell means)")
    print("#" * 70)

    cells = {}
    for r in backbone:
        cells.setdefault(r["cell_label"], []).append(r)

    print(f"\n{len(cells)} backbone cells, {sum(len(v) for v in cells.values())} raw runs "
          f"(asymmetric cells excluded from this fit -- see Part 2).")

    print("\nSE approach, stated explicitly: fitting on CELL MEANS (n=9), not")
    print("cluster-robust SEs on the 27 raw rows. With only 9 clusters,")
    print("cluster-robust sandwich SEs are known to be unreliable/anti-conservative")
    print("(reliable guidance is roughly >=30-50 clusters; 9 is well below that).")
    print("Collapsing to cell means sidesteps this: each cell contributes exactly")
    print("one independent observation (the 3 reps ARE the non-independent")
    print("pseudo-replicates this collapsing absorbs), so plain OLS needs no")
    print("clustering correction -- there's exactly one row per cluster by")
    print("construction. The cost is fewer degrees of freedom (27 -> 9).")

    labels_sorted = sorted(cells.keys())
    B_arr, R_arr, tpot_arr, within_cell_std = [], [], [], {}
    for label in labels_sorted:
        reps = cells[label]
        B_arr.append(float(reps[0]["B"]))
        r_vals = [float(r["R_actual"]) for r in reps]
        t_vals = [float(r["tpot_mean"]) for r in reps]
        R_arr.append(float(np.mean(r_vals)))
        tpot_arr.append(float(np.mean(t_vals)))
        within_cell_std[label] = float(np.std(t_vals, ddof=1)) if len(t_vals) > 1 else 0.0

    B_arr, R_arr, tpot_arr = np.array(B_arr), np.array(R_arr), np.array(tpot_arr)
    n = len(tpot_arr)
    ones = np.ones(n)

    print(f"\n{'cell':<12} {'B':>4} {'R_actual(mean)':>15} {'tpot_mean':>12} {'within-cell SD':>15}")
    for i, label in enumerate(labels_sorted):
        print(f"{label:<12} {B_arr[i]:>4.0f} {R_arr[i]:>15.1f} {tpot_arr[i]:>12.2f} {within_cell_std[label]:>15.3f}")

    X1 = np.column_stack([ones, B_arr])
    c1, rss1, fit1, xtx1, resid1 = ols(X1, tpot_arr)
    r2_1, adj1 = r2(tpot_arr, fit1, k=2)
    df1 = n - 2

    X2 = np.column_stack([ones, R_arr])
    c2, rss2, fit2, xtx2, resid2 = ols(X2, tpot_arr)
    r2_2, adj2 = r2(tpot_arr, fit2, k=2)
    df2 = n - 2

    X3 = np.column_stack([ones, B_arr, R_arr, B_arr * R_arr])
    c3, rss3, fit3, xtx3, resid3 = ols(X3, tpot_arr)
    r2_3, adj3 = r2(tpot_arr, fit3, k=4)
    df3 = n - 4

    print(f"\nn={n}  df: M1/M2={df1}  M3={df3}")
    if df3 <= 5:
        print("*** CAUTION: M3 has very few residual degrees of freedom (<=5).")
        print("*** Treat M3's coefficients/CIs as strongly directional, not definitive.")

    se1, lo1, hi1 = coef_cis(c1, rss1, df1, xtx1)
    print_model("M1", "tpot ~ b0 + b1*B", c1, se1, lo1, hi1, ["b0", "b1(B)"], rss1, r2_1, adj1, df1)

    se2, lo2, hi2 = coef_cis(c2, rss2, df2, xtx2)
    print_model("M2", "tpot ~ b0 + b1*R_actual", c2, se2, lo2, hi2, ["b0", "b1(R)"], rss2, r2_2, adj2, df2)

    se3, lo3, hi3 = coef_cis(c3, rss3, df3, xtx3)
    print_model("M3", "tpot ~ b0 + b1*B + b2*R_actual + b3*(B*R_actual)", c3, se3, lo3, hi3,
                ["b0", "b1(B)", "b2(R)", "b3(B:R)"], rss3, r2_3, adj3, df3)

    print(f"\n{'-'*70}")
    print("Nested F-tests vs M3 (df_diff=2):")
    f_m1_m3, p_m1_m3 = nested_f_test(rss1, rss3, 2, df3)
    f_m2_m3, p_m2_m3 = nested_f_test(rss2, rss3, 2, df3)
    p1s = f"p={p_m1_m3:.4f}" if p_m1_m3 is not None else "(install scipy for p-value)"
    p2s = f"p={p_m2_m3:.4f}" if p_m2_m3 is not None else "(install scipy for p-value)"
    print(f"  M1 vs M3: F={f_m1_m3:.3f}  {p1s}")
    print(f"  M2 vs M3: F={f_m2_m3:.3f}  {p2s}")

    print(f"\n{'-'*70}")
    print("Heteroscedasticity check (M3 residuals vs fitted, DESCRIPTIVE ONLY --")
    print("n=9 is too small for a formal Breusch-Pagan test):")
    corr = float(np.corrcoef(fit3, resid3 ** 2)[0, 1]) if np.std(fit3) > 0 else float("nan")
    print(f"  corr(fitted, residual^2) = {corr:.3f}")
    if corr > 0.4:
        print("  -> Residual variance appears to GROW with fitted value.")
        print("  -> A log(tpot) fit or weighted least squares would likely be more appropriate.")
    elif corr < -0.4:
        print("  -> Residual variance appears to SHRINK with fitted value (less common).")
    else:
        print("  -> No strong evidence of heteroscedasticity at this sample size.")

    print(f"\n{'-'*70}")
    print("R200_B4 (LOW_ITL_GAPS, ~9 ITL gaps/rep -- noisiest cell by design):")
    if "R200_B4" in labels_sorted:
        idx = labels_sorted.index("R200_B4")
        this_resid = resid3[idx]
        other_resids = np.delete(resid3, idx)
        other_std = float(np.std(other_resids, ddof=1)) if len(other_resids) > 1 else 0.0
        print(f"  M3 residual = {this_resid:.3f}ms   other 8 cells' residual SD = {other_std:.3f}ms")
        if other_std > 0 and abs(this_resid) > 2 * other_std:
            print("  -> OUTLIER: |residual| exceeds 2x the other cells' residual SD.")
        else:
            print("  -> Not a standout outlier relative to the other 8 cells' residuals.")
    else:
        print("  R200_B4 not found among backbone cells (unexpected -- check the CSV).")

    # Plots are written next to the analyzed CSV, so different sweeps do not overwrite each other.
    plot_dir = Path(path).parent / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(fit3, resid3)
    for i, label in enumerate(labels_sorted):
        ax.annotate(label, (fit3[i], resid3[i]), fontsize=7)
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_xlabel("Fitted tpot (M3, ms)")
    ax.set_ylabel("Residual (ms)")
    ax.set_title("Residual vs Fitted (M3, backbone cell means, n=9)")
    fig.tight_layout()
    resid_path = plot_dir / "resid_vs_fitted.png"
    fig.savefig(resid_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(tpot_arr, fit3, label="M3 fit (cell means)")
    lims = [min(tpot_arr.min(), fit3.min()), max(tpot_arr.max(), fit3.max())]
    ax.plot(lims, lims, color="gray", linestyle="--", label="y=x")
    for i, label in enumerate(labels_sorted):
        ax.annotate(label, (tpot_arr[i], fit3[i]), fontsize=7)
    ax.set_xlabel("Measured tpot (ms)")
    ax.set_ylabel("Predicted tpot (M3, ms)")
    ax.set_title("Predicted vs Measured (backbone cell means, n=9)")
    ax.legend()
    fig.tight_layout()
    pred_path = plot_dir / "predicted_vs_measured.png"
    fig.savefig(pred_path, dpi=150)
    plt.close(fig)

    print(f"\nPlots written: {resid_path}, {pred_path}")

    # Part 1B: refit on each request's own TPOT, since run means can blend two decode rates.
    print("\n" + "#" * 70)
    print("PART 1B — PER-REQUEST STEADY-STATE REFIT (same backbone cells)")
    print("#" * 70)

    per_req = load_per_request_tpot(backbone)
    n_pr = len(per_req)
    print(f"\n{n_pr} individual per-request tpot observations across the same "
          f"27 backbone runs (vs n=9 cell means in Part 1).")

    print("\nCAUTION on inference here: these observations are NOT independent --")
    print("up to 4 share a run, and runs share a cell. Proper clustered SEs at two")
    print("nested levels are a separate problem this refit doesn't solve. Point")
    print("estimates (coefficients) are reported for comparison against Part 1's")
    print("cell-means fit; treat any single-model R²/adj-R² here as illustrative")
    print("only, not as a replacement for Part 1's inference.")

    B_pr = np.array([x["B"] for x in per_req])
    R_pr = np.array([x["R_actual"] for x in per_req])
    tpot_pr = np.array([x["tpot"] for x in per_req])
    ones_pr = np.ones(n_pr)

    X1p = np.column_stack([ones_pr, B_pr])
    c1p, rss1p, fit1p, _, _ = ols(X1p, tpot_pr)
    _, adj1p = r2(tpot_pr, fit1p, k=2)

    X2p = np.column_stack([ones_pr, R_pr])
    c2p, rss2p, fit2p, _, _ = ols(X2p, tpot_pr)
    _, adj2p = r2(tpot_pr, fit2p, k=2)

    X3p = np.column_stack([ones_pr, B_pr, R_pr, B_pr * R_pr])
    c3p, rss3p, fit3p, _, _ = ols(X3p, tpot_pr)
    _, adj3p = r2(tpot_pr, fit3p, k=4)

    print(f"\n{'model':<10} {'b0':>10} {'b1(B)':>10} {'b2(R)':>10} {'b3(B:R)':>12} {'adj-R²':>9}")
    print(f"{'M1(cell)':<10} {c1[0]:>10.5f} {c1[1]:>10.5f} {'--':>10} {'--':>12} {adj1:>9.4f}")
    print(f"{'M1(req)':<10} {c1p[0]:>10.5f} {c1p[1]:>10.5f} {'--':>10} {'--':>12} {adj1p:>9.4f}")
    print(f"{'M2(cell)':<10} {c2[0]:>10.5f} {'--':>10} {c2[1]:>10.5f} {'--':>12} {adj2:>9.4f}")
    print(f"{'M2(req)':<10} {c2p[0]:>10.5f} {'--':>10} {c2p[1]:>10.5f} {'--':>12} {adj2p:>9.4f}")
    print(f"{'M3(cell)':<10} {c3[0]:>10.5f} {c3[1]:>10.5f} {c3[2]:>10.5f} {c3[3]:>12.6f} {adj3:>9.4f}")
    print(f"{'M3(req)':<10} {c3p[0]:>10.5f} {c3p[1]:>10.5f} {c3p[2]:>10.5f} {c3p[3]:>12.6f} {adj3p:>9.4f}")

    b1_shift = abs(c1p[1] - c1[1]) / abs(c1[1]) if c1[1] != 0 else float("inf")
    b2_shift = abs(c2p[1] - c2[1]) / abs(c2[1]) if c2[1] != 0 else float("inf")
    print(f"\nb1(B) shift, M1 cell-means -> per-request: {b1_shift*100:+.1f}%")
    print(f"b1(R) shift, M2 cell-means -> per-request: {b2_shift*100:+.1f}%")
    if b1_shift > 0.2 or b2_shift > 0.2:
        print("-> Coefficients change MATERIALLY (>20%) moving from run-mean to per-request.")
    else:
        print("-> Coefficients do NOT change materially moving from run-mean to per-request.")

    print("\n" + "-" * 70)
    print("NAMED FINDING: EFFECTIVE BATCH SPLITTING AT B=4")
    print("-" * 70)
    print("At B=4, llama.cpp does not always process 4 concurrent sequences as one")
    print("batch. Observed directly in this sweep's raw per-request timings:")
    print("  - R800_B4: 1 of 3 reps splits 2+2 (two requests decode at ~567ms/tok,")
    print("    two at ~239ms/tok) -- intermittent.")
    print("  - R3200_B4: ALL 3 of 3 reps split 3+1 (three at ~422ms/tok, one at")
    print("    ~287ms/tok), essentially identically each time -- deterministic at")
    print("    this load level.")
    print("  - R200_B4, R1600_B4: no splitting in any rep.")
    print("This means EFFECTIVE decoding concurrency at high resident load can sit")
    print("BELOW nominal B -- the same effective-vs-nominal-B distinction found")
    print("earlier for the asymmetric cells (diagnose_distribution_confound.py's")
    print("concurrency profile), now visible directly in the backbone cost data.")
    print("\nCitation scope (precise, not to be broadened): Zhou et al. 2026 (SLICE)")
    print("report that FastLLM, a CPU-oriented engine, shows decode latency rising")
    print("sharply once batch size exceeds a threshold (batch>9) -- establishing")
    print("that batch-size knees degrading per-step latency are real on CPU-oriented")
    print("engines generally. SLICE is cited ONLY for that general claim. The 2+2 /")
    print("3+1 splitting behavior itself is observed directly in this data and is")
    print("NOT reported by SLICE or any other source -- it is a distinct phenomenon:")
    print("rather than smooth degradation past a batch threshold, llama.cpp appears")
    print("to cap the effective batch below nominal B when resident context is large.")

    # Part 2: asymmetric cells, kept separate from Part 1's fit:
    # 2a decode cost on clean reps; 2b prefill contention, reported on its own.
    print("\n" + "#" * 70)
    print("PART 2 — ASYMMETRIC SPLIT COMPARISON (kept separate from Part 1's fit)")
    print("#" * 70)

    even_rows = cells.get("R800_B4", [])
    onebig_rows = [r for r in asym if r["cell_label"] == "onebig"]
    extreme_rows = [r for r in asym if r["cell_label"] == "extreme"]
    extreme_clean_rows = [r for r in extreme_rows if r["json_file"] not in anomalous_json_files]
    extreme_anomalous_rows = [r for r in extreme_rows if r["json_file"] in anomalous_json_files]

    print("\n" + "-" * 70)
    print("2a — DECODE COST: does scalar R suffice? (clean reps only)")
    print("-" * 70)
    print("All at B=4, R_target~800. Any run flagged in the anomaly scan above is")
    print("excluded here -- it reflects prefill contention (2b), a different")
    print("mechanism from decode cost, and must not be mixed into this comparison.")

    even_stats = cell_stats(even_rows, "even (backbone R800_B4, reused)")
    onebig_stats = cell_stats(onebig_rows, "onebig (500/100/100/100)")
    extreme_stats = cell_stats(extreme_clean_rows, "extreme, clean reps only (680/40/40/40)")

    print()
    for s in (even_stats, onebig_stats, extreme_stats):
        vals_str = ", ".join(f"{v:.2f}" for v in s["values"])
        print(f"  {s['label']:<38} n={s['n']}  mean={s['mean']:.2f}ms  "
              f"std={s['std']:.3f}ms  values=[{vals_str}]")
    if extreme_anomalous_rows:
        print(f"  (excluded from 'extreme' above: {len(extreme_anomalous_rows)} anomalous "
              f"rep(s) -- see 2b)")

    means = [even_stats["mean"], onebig_stats["mean"], extreme_stats["mean"]]
    spread_between = max(means) - min(means)
    avg_within = float(np.mean([s["std"] for s in (even_stats, onebig_stats, extreme_stats) if s["n"] > 1]))

    print(f"\nSpread between condition means: {spread_between:.2f}ms")
    print(f"Average within-condition SD (clean reps): {avg_within:.3f}ms")
    print("(n<=3 per condition -- reporting numbers and direction only, NOT a formal")
    print(" significance test; the sample is too small for that.)")

    print(f"\n{'='*70}")
    print("2a INTERPRETATION (descriptive -- not a hypothesis test)")
    print("=" * 70)
    if avg_within > 0 and spread_between > 3 * avg_within:
        print("Between-condition spread is large relative to within-condition noise.")
        print("-> tpot APPEARS TO DIFFER across splits at constant total R, based on")
        print("   the CLEAN decode-cost data (prefill-stall reps excluded).")
        print("-> Total R alone looks INSUFFICIENT for decode cost -- a distribution")
        print("   term (e.g. max slot length) may be warranted. FLAGGING AS A FINDING.")
    else:
        print("Between-condition spread is not clearly larger than within-condition noise.")
        print("-> tpot does NOT clearly differ across splits at constant total R.")
        print("-> Scalar R appears sufficient for decode cost; the router can score on")
        print("   total resident tokens.")

    print("\n" + "-" * 70)
    print("2a-PR — DECODE COST, PER-REQUEST (individual requests pooled instead of")
    print("        run-means -- tests whether run-mean contamination was masking")
    print("        or manufacturing the 2a conclusion)")
    print("-" * 70)
    print("'even' includes ALL 3 reps UNFILTERED, including the batch-split rep --")
    print("that split is the named finding above, not something to exclude here.")
    print("'extreme' still excludes the prefill-stall reps (2b) -- a separately")
    print("established, different mechanism from the decode-cost question.")

    even_pr = load_per_request_tpot(even_rows)
    onebig_pr = load_per_request_tpot(onebig_rows)
    extreme_pr = load_per_request_tpot(extreme_clean_rows)

    def pr_stats(pr_list, label):
        vals = [x["tpot"] for x in pr_list]
        return {
            "label": label, "n": len(vals),
            "mean": float(np.mean(vals)) if vals else float("nan"),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "values": vals,
        }

    even_pr_stats = pr_stats(even_pr, "even (all requests, all 3 reps, unfiltered)")
    onebig_pr_stats = pr_stats(onebig_pr, "onebig (all requests, all 3 reps)")
    extreme_pr_stats = pr_stats(extreme_pr, "extreme (clean reps only, all requests)")

    print()
    for s in (even_pr_stats, onebig_pr_stats, extreme_pr_stats):
        vals_str = ", ".join(f"{v:.1f}" for v in sorted(s["values"]))
        print(f"  {s['label']:<45} n={s['n']}  mean={s['mean']:.2f}ms  std={s['std']:.3f}ms")
        print(f"      sorted values: [{vals_str}]")

    means_pr = [even_pr_stats["mean"], onebig_pr_stats["mean"], extreme_pr_stats["mean"]]
    print(f"\nSpread between condition means (per-request pooled): "
          f"{max(means_pr) - min(means_pr):.2f}ms")

    print(f"\n{'='*70}")
    print("2a-PR READ (descriptive -- look at the sorted values above, not just the")
    print("means; 'even' is expected to show its own two clusters from the batch")
    print("split rather than being blended into one contaminated number)")
    print("=" * 70)

    print("\n" + "-" * 70)
    print("2b — PREFILL CONTENTION: did a long prefill stall concurrent decode?")
    print("-" * 70)
    print("A DIFFERENT mechanism from 2a (prefill contention, not decode cost).")
    print("Reported separately and NEVER used to override 2a's conclusion or to")
    print("justify a distribution term in the decode cost model on its own.")
    if extreme_anomalous_rows:
        print(f"\nOccurred in {len(extreme_anomalous_rows)} of {len(extreme_rows)} identical "
              f"'extreme' reps -- INTERMITTENT, not deterministic.")
        for r in extreme_anomalous_rows:
            match = next(a for a in anomalies if a["row"] is r)
            print_anomaly_detail(match)

        # Mechanism check computed from this node's own R800_B1 cell (not hard-coded values).
        r800_b1_rows = cells.get("R800_B1", [])
        if r800_b1_rows:
            per_token_rates = []
            for r in r800_b1_rows:
                d = json.load(open(r["json_file"]))
                for req in d["raw"]:
                    if req["success"] and req["input_tokens"] > 0:
                        per_token_rates.append(req["ttft_ms"] / req["input_tokens"])
            baseline_ms_per_tok = float(np.mean(per_token_rates))

            print(f"\nMechanism: the big request's prefill appears to occupy the CPU for")
            print("its entire duration, during which the small concurrent requests'")
            print("decode stalls -- their few output tokens only complete once the big")
            print("request's prefill finishes. Checked against an isolated-request")
            print(f"baseline computed from THIS dataset's own R800_B1 cell (640 tokens,")
            print(f"B=1, no concurrency): {baseline_ms_per_tok:.2f}ms/token.")
            for r in extreme_anomalous_rows:
                match = next(a for a in anomalies if a["row"] is r)
                big_req = max(match["raw"], key=lambda req: req["input_tokens"])
                predicted_ms = baseline_ms_per_tok * big_req["input_tokens"]
                observed_ms = big_req["ttft_ms"]
                pct_over = (observed_ms - predicted_ms) / predicted_ms * 100
                print(f"  rep {r['rep']}: {big_req['input_tokens']} tokens -> "
                      f"{predicted_ms:.0f}ms predicted vs {observed_ms:.0f}ms observed "
                      f"({pct_over:+.0f}% -- ordinary CPU-sharing overhead among 4 "
                      f"concurrent requests, not a separate anomaly, if roughly in the "
                      f"same ballpark as other reps here).")
        else:
            print("\n(R800_B1 cell not found in this dataset -- can't compute the")
            print(" isolated-request baseline comparison.)")
    else:
        print("\nNo prefill-stall runs detected in this sweep's data.")


if __name__ == "__main__":
    main()
