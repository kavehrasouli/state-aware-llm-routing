"""Fits the additive cost model (TPOT ~ B + R) next to the models of analyze_cost_model_sweep.py.

Usage: python3 analyze_cost_model_additive.py [summary.csv ...]
"""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Reuse the original's numerics verbatim -- not a re-implementation.
from analyze_cost_model_sweep import (  # noqa: E402
    HAVE_SCIPY,
    coef_cis,
    load_rows,
    nested_f_test,
    ols,
    r2,
)

ORIGINAL = HERE / "analyze_cost_model_sweep.py"

DEFAULT_CSVS = [
    "cost_model_results_pi-node-1_Pi5-8GB/sweep_cost_model_summary.csv",
    "cost_model_results_pi-node-2_4B-4g/sweep_cost_model_summary.csv",
    "cost_model_results_pi-node-3_4B-8g/sweep_cost_model_summary.csv",
]

# The original's print_model()/print formats. Comparison is done on these
# exact renderings, so "reproduces exactly" means exactly as printed.
F_COEF = "{:.5f}"
F_SE = "{:.5f}"
F_R2 = "{:.4f}"
F_RSS = "{:.2f}"


# --------------------------------------------------------------------- #
# Data prep -- mirrors analyze_cost_model_sweep.py main(), Part 1
# --------------------------------------------------------------------- #
def backbone_cell_means(csv_path):
    rows = load_rows(csv_path)
    clean_rows = [r for r in rows if r["throttled"] != "1"]
    backbone = [r for r in clean_rows if r["block"] == "backbone"]

    cells = {}
    for r in backbone:
        cells.setdefault(r["cell_label"], []).append(r)

    labels_sorted = sorted(cells.keys())
    B_arr, R_arr, tpot_arr = [], [], []
    for label in labels_sorted:
        reps = cells[label]
        B_arr.append(float(reps[0]["B"]))
        R_arr.append(float(np.mean([float(r["R_actual"]) for r in reps])))
        tpot_arr.append(float(np.mean([float(r["tpot_mean"]) for r in reps])))

    return (labels_sorted, np.array(B_arr), np.array(R_arr), np.array(tpot_arr),
            len(rows), len(rows) - len(clean_rows), sum(len(v) for v in cells.values()))


def fit_all(B, R, y):
    """Fit M1, M2, M_A, M3 on the cell means. Returns dict keyed by model."""
    n = len(y)
    ones = np.ones(n)

    designs = {
        "M1":  (np.column_stack([ones, B]),          ["b0", "b1(B)"],                       2),
        "M2":  (np.column_stack([ones, R]),          ["b0", "b1(R)"],                       2),
        "M_A": (np.column_stack([ones, B, R]),       ["b0", "b1(B)", "b2(R)"],              3),
        "M3":  (np.column_stack([ones, B, R, B * R]), ["b0", "b1(B)", "b2(R)", "b3(B:R)"],  4),
    }

    out = {}
    for name, (X, coef_names, k) in designs.items():
        coefs, rss, fitted, xtx_inv, resid = ols(X, y)
        rsq, adj = r2(y, fitted, k=k)
        dfree = n - k
        se, lo, hi = coef_cis(coefs, rss, dfree, xtx_inv)
        out[name] = {
            "names": coef_names, "coefs": coefs, "se": se, "lo": lo, "hi": hi,
            "rss": rss, "r2": rsq, "adj": adj, "df": dfree, "k": k,
            "fitted": fitted, "resid": resid,
        }
    return out


# --------------------------------------------------------------------- #
# Verification gate: parse the ORIGINAL script's real stdout
# --------------------------------------------------------------------- #
_MODEL_HDR = re.compile(r"^(M1|M2|M3): ")
_COEF_LINE = re.compile(r"^\s{2}(\S+)\s+=\s+(-?[\d.]+)\s+SE=\s*(-?[\d.]+)\s+95% CI=")
_STATS_LINE = re.compile(r"^\s{2}R²=(-?[\d.]+)\s+adj-R²=(-?[\d.]+)\s+RSS=([\d.]+)\s+df=(\d+)")
_FTEST_LINE = re.compile(r"^\s{2}(M1|M2) vs M3: F=([\d.]+)\s+(?:p=([\d.]+))?")


def run_original(csv_path):
    proc = subprocess.run(
        [sys.executable, str(ORIGINAL), csv_path],
        cwd=str(HERE), capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"original script failed on {csv_path} (exit {proc.returncode}):\n{proc.stderr}")
    return proc.stdout


def parse_original(stdout):
    """Extract M1/M2/M3 blocks and the M1/M2-vs-M3 F-tests from Part 1 only.

    Part 1 is delimited so Part 1B's per-request table (which reprints
    coefficients under labels like 'M1(cell)') can never be picked up here.
    """
    lines = stdout.splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if "PART 1 — MAIN FIT" in ln)
        end = next(i for i, ln in enumerate(lines) if "PART 1B" in ln)
    except StopIteration:
        raise RuntimeError("could not delimit Part 1 in the original script's output")
    part1 = lines[start:end]

    models, ftests, current = {}, {}, None
    for ln in part1:
        hdr = _MODEL_HDR.match(ln)
        if hdr:
            current = hdr.group(1)
            models[current] = {"coefs": [], "stats": None}
            continue
        if current:
            m = _COEF_LINE.match(ln)
            if m:
                models[current]["coefs"].append((m.group(1), m.group(2), m.group(3)))
                continue
            m = _STATS_LINE.match(ln)
            if m:
                models[current]["stats"] = (m.group(1), m.group(2), m.group(3), int(m.group(4)))
                current = None
                continue
        m = _FTEST_LINE.match(ln)
        if m:
            ftests[m.group(1)] = (m.group(2), m.group(3))

    for name in ("M1", "M2", "M3"):
        if name not in models or models[name]["stats"] is None:
            raise RuntimeError(f"could not parse {name} out of the original script's output")
    return models, ftests


def verify(csv_path, fits, ftests_here):
    """Abort unless every printed M1/M2/M3 number matches the original exactly."""
    ref_models, ref_ftests = parse_original(run_original(csv_path))
    checks, failures = 0, []

    def cmp(label, mine, theirs):
        nonlocal checks
        checks += 1
        if mine != theirs:
            failures.append(f"{label}: this script={mine!r}  original={theirs!r}")

    for name in ("M1", "M2", "M3"):
        f, ref = fits[name], ref_models[name]
        if len(ref["coefs"]) != len(f["coefs"]):
            failures.append(f"{name}: coefficient count "
                            f"{len(f['coefs'])} vs {len(ref['coefs'])}")
            continue
        for (rname, rcoef, rse), nm, c, s in zip(
                ref["coefs"], f["names"], f["coefs"], f["se"]):
            cmp(f"{name} {nm} name", nm, rname)
            cmp(f"{name} {nm} coef", F_COEF.format(c), rcoef)
            cmp(f"{name} {nm} SE", F_SE.format(s), rse)
        r_r2, r_adj, r_rss, r_df = ref["stats"]
        cmp(f"{name} R²", F_R2.format(f["r2"]), r_r2)
        cmp(f"{name} adj-R²", F_R2.format(f["adj"]), r_adj)
        cmp(f"{name} RSS", F_RSS.format(f["rss"]), r_rss)
        cmp(f"{name} df", f["df"], r_df)

    # The two pre-existing F-tests must also come out identical.
    for name in ("M1", "M2"):
        if name in ref_ftests:
            r_f, r_p = ref_ftests[name]
            f_here, p_here = ftests_here[name]
            cmp(f"{name} vs M3 F", f"{f_here:.3f}", r_f)
            if r_p is not None and p_here is not None:
                cmp(f"{name} vs M3 p", f"{p_here:.4f}", r_p)

    if failures:
        print(f"\n*** VERIFICATION FAILED for {csv_path} "
              f"({len(failures)} of {checks} checks) ***")
        for f in failures:
            print(f"    {f}")
        print("\nAborting: M_A is not reported unless M1/M2/M3 reproduce exactly.")
        sys.exit(1)
    return checks


# --------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------- #
def fmt_p(p):
    if p is None:
        return "p=(scipy missing)"
    return f"p={p:.4f}" if p >= 0.0001 else f"p={p:.2e}"


def print_model(name, formula, f):
    print(f"\n{name}: {formula}")
    for nm, c, s, lo, hi in zip(f["names"], f["coefs"], f["se"], f["lo"], f["hi"]):
        print(f"  {nm:<12} = {c:>10.5f}   SE={s:>9.5f}   "
              f"95% CI=[{lo:>10.5f}, {hi:>10.5f}]")
    print(f"  R²={f['r2']:.4f}  adj-R²={f['adj']:.4f}  RSS={f['rss']:.2f}  df={f['df']}")


def analyze(csv_path):
    node = Path(csv_path).parent.name.replace("cost_model_results_", "")
    print("\n" + "#" * 74)
    print(f"# {node}")
    print("#" * 74)

    labels, B, R, y, n_rows, n_dropped, n_runs = backbone_cell_means(csv_path)
    n = len(y)
    fits = fit_all(B, R, y)

    # Pre-existing tests, recomputed here for the verification gate.
    ftests_here = {
        "M1": nested_f_test(fits["M1"]["rss"], fits["M3"]["rss"], 2, fits["M3"]["df"]),
        "M2": nested_f_test(fits["M2"]["rss"], fits["M3"]["rss"], 2, fits["M3"]["df"]),
    }

    checks = verify(csv_path, fits, ftests_here)
    print(f"\nVERIFICATION: {checks}/{checks} checks passed -- M1, M2 and M3 "
          f"coefficients,\n  SEs, R², adj-R², RSS, df and both existing F-tests "
          f"reproduce the output of\n  analyze_cost_model_sweep.py exactly (compared "
          f"at its own printed precision).")

    print(f"\nData: {n_rows} rows, {n_dropped} throttled dropped, "
          f"{n_runs} backbone runs -> n={n} cell means.")
    print(f"\n{'cell':<12} {'B':>4} {'R_actual(mean)':>15} {'tpot_mean':>12}")
    for i, label in enumerate(labels):
        print(f"{label:<12} {B[i]:>4.0f} {R[i]:>15.1f} {y[i]:>12.2f}")

    print_model("M1 ", "tpot ~ b0 + b1*B", fits["M1"])
    print_model("M2 ", "tpot ~ b0 + b1*R_actual", fits["M2"])
    print_model("M_A", "tpot ~ b0 + b1*B + b2*R_actual   [additive, no interaction]",
                fits["M_A"])
    print_model("M3 ", "tpot ~ b0 + b1*B + b2*R_actual + b3*(B*R_actual)", fits["M3"])

    a, m3 = fits["M_A"], fits["M3"]
    tests = [
        ("M1 vs M_A", "does R add beyond B",
         nested_f_test(fits["M1"]["rss"], a["rss"], 1, a["df"]), 1, a["df"]),
        ("M2 vs M_A", "does B add beyond R",
         nested_f_test(fits["M2"]["rss"], a["rss"], 1, a["df"]), 1, a["df"]),
        ("M_A vs M3", "is the interaction needed",
         nested_f_test(a["rss"], m3["rss"], 1, m3["df"]), 1, m3["df"]),
        ("M1 vs M3", "existing test, R+interaction jointly",
         ftests_here["M1"], 2, m3["df"]),
        ("M2 vs M3", "existing test, B+interaction jointly",
         ftests_here["M2"], 2, m3["df"]),
    ]

    print(f"\n{'-' * 74}")
    print("NESTED F-TESTS")
    print(f"{'comparison':<12} {'question':<38} {'df':>5} {'F':>8}  p")
    for label, question, (fstat, p), d1, d2 in tests:
        print(f"{label:<12} {question:<38} {d1:>2},{d2:<2} {fstat:>8.3f}  {fmt_p(p)}")

    # t² on a single M_A coefficient must equal the corresponding 1-df F.
    # Internal consistency check on the two "does X add beyond Y" tests.
    t_R = a["coefs"][2] / a["se"][2]
    t_B = a["coefs"][1] / a["se"][1]
    f_m1_a = tests[0][2][0]
    f_m2_a = tests[1][2][0]
    print(f"\nConsistency: t²(b2(R) in M_A)={t_R ** 2:.3f} vs F(M1 vs M_A)={f_m1_a:.3f}; "
          f"t²(b1(B) in M_A)={t_B ** 2:.3f} vs F(M2 vs M_A)={f_m2_a:.3f}")

    return {
        "node": node, "fits": fits,
        "tests": {label: (fstat, p) for label, _, (fstat, p), _, _ in tests},
    }


def main():
    csvs = sys.argv[1:] or DEFAULT_CSVS
    if not HAVE_SCIPY:
        print("ERROR: scipy is required -- the whole point here is the p-values.")
        sys.exit(1)

    print("=" * 74)
    print("M_A (additive) added to the M1/M2/M3 cost-model comparison")
    print("Backbone cells only, cell means (n=9), R_actual not R_target.")
    print("=" * 74)

    results = [analyze(c) for c in csvs]

    print("\n" + "=" * 74)
    print("CROSS-NODE SUMMARY")
    print("=" * 74)

    print(f"\n{'node':<20} {'M1':>8} {'M2':>8} {'M_A':>8} {'M3':>8}   (adj-R²)")
    for r in results:
        f = r["fits"]
        print(f"{r['node']:<20} {f['M1']['adj']:>8.4f} {f['M2']['adj']:>8.4f} "
              f"{f['M_A']['adj']:>8.4f} {f['M3']['adj']:>8.4f}")

    print(f"\n{'node':<20} {'M1':>10} {'M2':>10} {'M_A':>10} {'M3':>10}   (RSS)")
    for r in results:
        f = r["fits"]
        print(f"{r['node']:<20} {f['M1']['rss']:>10.2f} {f['M2']['rss']:>10.2f} "
              f"{f['M_A']['rss']:>10.2f} {f['M3']['rss']:>10.2f}")

    order = ["M1 vs M_A", "M2 vs M_A", "M_A vs M3", "M1 vs M3", "M2 vs M3"]
    print(f"\n{'node':<20} " + " ".join(f"{o:>12}" for o in order) + "   (p-values)")
    for r in results:
        cells = " ".join(f"{r['tests'][o][1]:>12.4f}" for o in order)
        print(f"{r['node']:<20} {cells}")

    print(f"\n{'node':<20} " + " ".join(f"{o:>12}" for o in order) + "   (F)")
    for r in results:
        cells = " ".join(f"{r['tests'][o][0]:>12.3f}" for o in order)
        print(f"{r['node']:<20} {cells}")

    print("\nM_A coefficients across nodes:")
    print(f"{'node':<20} {'b0':>12} {'b1(B)':>12} {'b2(R)':>12}")
    for r in results:
        c = r["fits"]["M_A"]["coefs"]
        s = r["fits"]["M_A"]["se"]
        print(f"{r['node']:<20} {c[0]:>12.5f} {c[1]:>12.5f} {c[2]:>12.5f}")
        print(f"{'  (SE)':<20} {s[0]:>12.5f} {s[1]:>12.5f} {s[2]:>12.5f}")


if __name__ == "__main__":
    main()
