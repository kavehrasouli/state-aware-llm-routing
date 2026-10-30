"""Compares six cost-model forms for the Pi 5 (adjusted R2 and leave-one-out error).

Usage (from benchmarks/): python3 fit_pi5_candidates.py [sweep_cost_model_summary.csv]
"""
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

try:
    from scipy import stats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False
    print("WARNING: scipy not installed -- t-critical values fall back to a "
          "small hardcoded table, so CI columns may differ from the original.\n")

# Two-tailed alpha=0.05 t-critical values, used only if scipy is unavailable.
# Same fallback table as analyze_cost_model_sweep.py.
_T_TABLE_005 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
                6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}

MISMATCH_TOL = 5e-4

# Grouping tolerance for the observed batch sizes (a detection threshold, not a fitted value);
# the script reports the range of tolerances over which the grouping is stable.
SPLIT_TOL = 0.05


def t_critical(df):
    if HAVE_SCIPY:
        return float(stats.t.ppf(0.975, df))
    return _T_TABLE_005.get(df, 2.0)


def find_default_csv():
    hits = sorted(glob.glob("cost_model_results_pi-node-1_*/sweep_cost_model_summary.csv"))
    if not hits:
        sys.exit("No cost_model_results_pi-node-1_*/sweep_cost_model_summary.csv found. "
                 "Run from benchmarks/, or pass the CSV path explicitly.")
    if len(hits) > 1:
        sys.exit(f"Ambiguous: {len(hits)} pi-node-1 summary CSVs match {hits}. "
                 "Pass the one you mean explicitly.")
    return hits[0]


def resolve_json(raw_path, csv_path):
    """json_file paths in the summary CSV are relative to benchmarks/. Retry
    relative to the CSV's own directory and its parent so this works from any
    cwd instead of silently failing outside benchmarks/."""
    csv_path = Path(csv_path)
    for cand in (Path(raw_path),
                 csv_path.parent / raw_path,
                 csv_path.parent.parent / raw_path,
                 csv_path.parent / Path(raw_path).name):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"cannot locate raw result {raw_path!r} referenced by {csv_path}")


def load_backbone_cells(csv_path):
    """Same filtering as analyze_cost_model_sweep.py Part 1: throttled rows
    dropped, backbone block only, then collapsed to one row per cell."""
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    n_total = len(rows)
    throttled = [r for r in rows if r["throttled"] == "1"]
    clean = [r for r in rows if r["throttled"] != "1"]
    backbone = [r for r in clean if r["block"] == "backbone"]

    cells = {}
    for r in backbone:
        cells.setdefault(r["cell_label"], []).append(r)
    return rows, n_total, throttled, backbone, cells


def effective_batch_sizes(cells, labels, csv_path, tol=SPLIT_TOL):
    """Sequence-weighted effective batch size per cell, measured from the raw
    per-request decode rates.

    Within a run, sort the per-request tpot values and cut a new group wherever
    consecutive values differ by more than `tol` relatively. A run whose B
    requests all decode together forms one group of size B; a run that splits
    3+1 forms groups of size 3 and 1. Weighting by sequence (each sequence sees
    the size of ITS OWN group) gives B_eff = sum(g^2) / sum(g), so an unsplit
    run returns exactly B and a 3+1 split returns (9+1)/4 = 2.5. The cell's
    B_eff is the mean over its reps.
    """
    out, detail = [], []
    for label in labels:
        per_rep = []
        for r in sorted(cells[label], key=lambda z: z["rep"]):
            d = json.load(open(resolve_json(r["json_file"], csv_path)))
            t = sorted(req["tpot_ms"] for req in d["raw"]
                       if req["success"] and req["tpot_ms"] > 0)
            groups, cur = [], 1
            for i in range(1, len(t)):
                if t[i] > t[i - 1] * (1 + tol):
                    groups.append(cur)
                    cur = 1
                else:
                    cur += 1
            groups.append(cur)
            g = np.array(groups, dtype=float)
            per_rep.append(float((g ** 2).sum() / g.sum()))
            detail.append((label, r["rep"], groups, per_rep[-1]))
        out.append(float(np.mean(per_rep)))
    return np.array(out), detail


def split_tolerance_stability(cells, labels, csv_path):
    """Report the tolerance window over which the grouping above is unchanged.
    If that window is wide, SPLIT_TOL is not a judgement call in disguise."""
    ref, _ = effective_batch_sizes(cells, labels, csv_path, tol=SPLIT_TOL)
    lo = hi = SPLIT_TOL
    grid = np.concatenate([np.arange(0.001, 0.05, 0.001), np.arange(0.05, 0.60, 0.005)])
    for t in grid[grid <= SPLIT_TOL][::-1]:
        if np.allclose(effective_batch_sizes(cells, labels, csv_path, tol=t)[0], ref):
            lo = float(t)
        else:
            break
    for t in grid[grid >= SPLIT_TOL]:
        if np.allclose(effective_batch_sizes(cells, labels, csv_path, tol=t)[0], ref):
            hi = float(t)
        else:
            break
    return lo, hi


def ols(X, y):
    coefs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ coefs
    resid = y - fitted
    return coefs, fitted, resid


def fit_model(X, y):
    """OLS plus every reported diagnostic, all computed here at runtime.

    RMSE divides RSS by n (not by residual df) -- that is what reproduces the
    original CSV's rmse_ms column. LOOCV is leave-one-CELL-MEAN-out: the model
    is refit on the other 8 cells and scored on the held-out cell, which is the
    only honest cross-validation unit here because the 3 reps behind each cell
    mean are pseudo-replicates, not independent draws.
    """
    n, k = X.shape
    coefs, fitted, resid = ols(X, y)
    rss = float(resid @ resid)
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1 - rss / sst
    df_resid = n - k
    adj = 1 - (1 - r_squared) * (n - 1) / df_resid if df_resid > 0 else float("nan")

    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag((rss / df_resid) * xtx_inv)) if df_resid > 0 else np.full(k, np.nan)
    tcrit = t_critical(df_resid)
    ci_lo, ci_hi = coefs - tcrit * se, coefs + tcrit * se

    loo_err = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        ci, _, _ = ols(X[mask], y[mask])
        loo_err.append(y[i] - float(X[i] @ ci))
    loo_err = np.array(loo_err)

    return {
        "n_cells": n, "df_resid": df_resid, "coefs": coefs, "se": se,
        "ci_lo": ci_lo, "ci_hi": ci_hi, "r2": r_squared, "adj_r2": adj,
        "rmse": float(np.sqrt(rss / n)), "mae": float(np.mean(np.abs(resid))),
        "loocv_rmse": float(np.sqrt(np.mean(loo_err ** 2))),
        "loocv_mae": float(np.mean(np.abs(loo_err))),
        "resid": resid, "fitted": fitted,
    }


def build_specs(B, R, B_eff):
    """The six candidates.

    Each entry maps design-matrix columns onto the CSV's c0/c1/c2 slots. That
    mapping is by ROLE, not position: r_only has no B term, so its R
    coefficient lands in c2 and c1 stays empty, matching the original file.
    """
    ones = np.ones(len(B))
    return [
        ("constant", "tpot=c0",
         [("c0", ones)]),
        ("r_only", "tpot=c0+c2*R",
         [("c0", ones), ("c2", R)]),
        ("linear_b_r", "tpot=c0+c1*B+c2*R",
         [("c0", ones), ("c1", B), ("c2", R)]),
        ("cap2_b_r", "tpot=c0+c1*min(B,2)+c2*R",
         [("c0", ones), ("c1", np.minimum(B, 2.0)), ("c2", R)]),
        ("observed_split_beff_r", "tpot=c0+c1*B_eff+c2*R",
         [("c0", ones), ("c1", B_eff), ("c2", R)]),
        ("b2_indicator_r", "tpot=c0+d*I(B=2)+c2*R",
         [("c0", ones), ("c1", (B == 2).astype(float)), ("c2", R)]),
    ]


NUMERIC_COLS = ["n_cells", "df_resid",
                "c0", "c0_se", "c0_ci_low", "c0_ci_high",
                "c1", "c1_se", "c1_ci_low", "c1_ci_high",
                "c2", "c2_se", "c2_ci_low", "c2_ci_high",
                "r2", "adj_r2", "rmse_ms", "mae_ms", "loocv_rmse_ms", "loocv_mae_ms"]
OUT_COLS = ["model", "formula"] + NUMERIC_COLS + ["status", "notes"]


def row_for(name, formula, slots, fit):
    row = {c: "" for c in OUT_COLS}
    row["model"] = name
    row["formula"] = formula
    row["n_cells"] = fit["n_cells"]
    row["df_resid"] = fit["df_resid"]
    for j, (slot, _) in enumerate(slots):
        row[slot] = f"{fit['coefs'][j]:.6f}"
        row[f"{slot}_se"] = f"{fit['se'][j]:.6f}"
        row[f"{slot}_ci_low"] = f"{fit['ci_lo'][j]:.6f}"
        row[f"{slot}_ci_high"] = f"{fit['ci_hi'][j]:.6f}"
    row["r2"] = f"{fit['r2']:.6f}"
    row["adj_r2"] = f"{fit['adj_r2']:.6f}"
    row["rmse_ms"] = f"{fit['rmse']:.6f}"
    row["mae_ms"] = f"{fit['mae']:.6f}"
    row["loocv_rmse_ms"] = f"{fit['loocv_rmse']:.6f}"
    row["loocv_mae_ms"] = f"{fit['loocv_mae']:.6f}"
    # status/notes are editorial judgements about deployability, not fitted
    # quantities -- left empty rather than invented. See the ambiguity notes.
    return row


def compare(new_rows, old_path):
    """Diff every numeric cell against the existing CSV. Reports four outcomes
    per cell: match, mismatch above tolerance, and either side being empty
    where the other is not (a structural disagreement, not a numeric one)."""
    if not Path(old_path).exists():
        print(f"\nNo existing {old_path} to compare against -- skipping the diff.")
        return None
    with open(old_path) as f:
        old = {r["model"]: r for r in csv.DictReader(f)}

    n_ok = n_bad = n_struct = 0
    worst = (0.0, None)
    bad_lines, struct_lines = [], []

    only_new = [r["model"] for r in new_rows if r["model"] not in old]
    only_old = [m for m in old if m not in {r["model"] for r in new_rows}]

    print(f"\n{'model':<24}{'column':<16}{'recomputed':>15}{'existing':>15}{'abs diff':>13}  verdict")
    print("-" * 98)
    for r in new_rows:
        o = old.get(r["model"])
        if o is None:
            continue
        for col in NUMERIC_COLS:
            nv, ov = r[col], (o.get(col) or "").strip()
            if nv == "" and ov == "":
                continue
            if nv == "" or ov == "":
                n_struct += 1
                line = (f"{r['model']:<24}{col:<16}{(nv or '(empty)'):>15}"
                        f"{(ov or '(empty)'):>15}{'--':>13}  STRUCTURAL")
                struct_lines.append(line)
                print(line)
                continue
            diff = abs(float(nv) - float(ov))
            if diff > worst[0]:
                worst = (diff, f"{r['model']}.{col}")
            if diff > MISMATCH_TOL:
                n_bad += 1
                line = (f"{r['model']:<24}{col:<16}{float(nv):>15.6f}"
                        f"{float(ov):>15.6f}{diff:>13.2e}  MISMATCH")
                bad_lines.append(line)
                print(line)
            else:
                n_ok += 1
    print("-" * 98)
    print(f"{n_ok} numeric cells match within {MISMATCH_TOL:g}; "
          f"{n_bad} exceed it; {n_struct} structural (one side empty).")
    print(f"Largest absolute difference: {worst[0]:.3e}"
          + (f" at {worst[1]}" if worst[1] else ""))
    if only_new:
        print(f"Models only in the recomputed file: {only_new}")
    if only_old:
        print(f"Models only in the existing file:  {only_old}")

    # formula strings are labels, not results -- compared for drift, not correctness
    for r in new_rows:
        o = old.get(r["model"])
        if o and o["formula"].strip() != r["formula"]:
            print(f"NOTE: formula label differs for {r['model']}: "
                  f"recomputed {r['formula']!r} vs existing {o['formula']!r}")
    return n_bad + n_struct


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else find_default_csv()
    out_path = Path(csv_path).parent / "pi5_cost_model_fits_recomputed.csv"
    old_path = Path(csv_path).parent / "pi5_cost_model_fits.csv"
    if out_path.resolve() == old_path.resolve():
        sys.exit("Refusing to overwrite pi5_cost_model_fits.csv.")

    print("=" * 98)
    print(f"Source: {csv_path}")
    rows, n_total, throttled, backbone, cells = load_backbone_cells(csv_path)
    print(f"Loaded {n_total} rows; dropped {len(throttled)} throttled; "
          f"{len(backbone)} backbone runs in {len(cells)} cells.")
    if throttled:
        for r in throttled:
            print(f"  dropped (throttled): {r['block']}/{r['cell_label']} rep={r['rep']}")
    print("=" * 98)

    labels = sorted(cells)
    B = np.array([float(cells[l][0]["B"]) for l in labels])
    R = np.array([float(np.mean([float(r["R_actual"]) for r in cells[l]])) for l in labels])
    y = np.array([float(np.mean([float(r["tpot_mean"]) for r in cells[l]])) for l in labels])
    reps = np.array([len(cells[l]) for l in labels])

    B_eff, detail = effective_batch_sizes(cells, labels, csv_path)
    tol_lo, tol_hi = split_tolerance_stability(cells, labels, csv_path)

    print(f"\n{'cell':<12}{'reps':>5}{'B':>4}{'R_actual':>10}{'tpot_mean':>12}"
          f"{'B_eff':>8}{'min(B,2)':>10}{'I(B=2)':>8}")
    for i, l in enumerate(labels):
        print(f"{l:<12}{reps[i]:>5}{B[i]:>4.0f}{R[i]:>10.1f}{y[i]:>12.3f}"
              f"{B_eff[i]:>8.3f}{min(B[i], 2.0):>10.1f}{float(B[i] == 2):>8.0f}")

    print(f"\nObserved-split detail (groups of concurrently-decoding sequences per rep, "
          f"grouping tolerance {SPLIT_TOL:.0%}):")
    for label, rep, groups, beff in detail:
        if len(groups) > 1:
            print(f"  {label:<12} rep{rep}: groups {groups} -> B_eff={beff:.3f}  (SPLIT)")
    unsplit = [l for i, l in enumerate(labels) if abs(B_eff[i] - B[i]) < 1e-9]
    print(f"  cells where B_eff == nominal B (no split in any rep): {unsplit}")
    print(f"  grouping is unchanged for tolerances in [{tol_lo:.1%}, {tol_hi:.1%}]")

    specs = build_specs(B, R, B_eff)
    new_rows = []
    print(f"\n{'model':<24}{'c0':>12}{'c1':>12}{'c2':>12}{'R2':>10}{'adj-R2':>10}"
          f"{'RMSE':>10}{'LOOCV':>10}")
    print("-" * 98)
    for name, formula, slots in specs:
        X = np.column_stack([v for _, v in slots])
        fit = fit_model(X, y)
        new_rows.append(row_for(name, formula, slots, fit))
        got = dict(zip([s for s, _ in slots], fit["coefs"]))

        def cell(k, got=got):
            return f"{got[k]:>12.6f}" if k in got else f"{'--':>12}"

        print(f"{name:<24}{cell('c0')}{cell('c1')}{cell('c2')}"
              f"{fit['r2']:>10.4f}{fit['adj_r2']:>10.4f}"
              f"{fit['rmse']:>10.3f}{fit['loocv_rmse']:>10.3f}")

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLS)
        w.writeheader()
        w.writerows(new_rows)
    print(f"\nWrote {out_path}")

    print("\n" + "=" * 98)
    print(f"COMPARISON vs {old_path.name} (tolerance {MISMATCH_TOL:g})")
    print("=" * 98)
    n_problems = compare(new_rows, old_path)

    print("\n" + "=" * 98)
    print("AMBIGUITIES IN THE ORIGINAL CSV THAT THIS SCRIPT HAD TO RESOLVE")
    print("=" * 98)
    print("1. B_eff for observed_split_beff_r. The original records only the")
    print("   result ('sequence-weighted B_eff=2.5' for R3200_B4); the weighting")
    print("   rule is not stated. This script computes sum(g^2)/sum(g) over")
    print("   observed decode-rate groups, which reproduces 2.5 from a 3+1 split")
    print("   and returns nominal B when a run does not split. A different but")
    print("   equally defensible rule -- e.g. the unweighted mean group size,")
    print("   (3+1)/2 = 2.0 -- would give a different B_eff, so the agreement")
    print("   below is evidence FOR this reading, not proof of it.")
    print("2. RMSE denominator. rmse_ms reproduces only with RSS/n, not RSS/df.")
    print("3. status and notes are editorial judgements about deployability, not")
    print("   fitted quantities. They are left EMPTY here rather than invented,")
    print("   and are NOT part of the diff above -- only numeric columns are")
    print("   compared, so the original's wording is neither checked nor claimed.")
    print("4. The original reports p-values in the .md but not the .csv; they are")
    print("   not emitted here because the CSV has no column for them.")

    if n_problems:
        print(f"\nRESULT: {n_problems} cell(s) did not match cleanly -- see above.")
    else:
        print("\nRESULT: every numeric cell reproduces within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
