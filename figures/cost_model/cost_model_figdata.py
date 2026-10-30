import glob
import re
import sys
from pathlib import Path

import numpy as np


def _locate_benchmarks_dir():
    """
    Find the directory holding analyze_cost_model_sweep.py.
    """
    here = Path(__file__).resolve().parent
    candidates = [here.parent / "benchmarks", here, here / "benchmarks"]
    candidates += [p / "benchmarks" for p in here.parents]
    seen, ordered = set(), []
    for c in candidates:
        if str(c) not in seen:
            seen.add(str(c))
            ordered.append(c)
    for c in ordered:
        if (c / "analyze_cost_model_sweep.py").is_file():
            return c.resolve()
    raise SystemExit(
        "FATAL: cannot find analyze_cost_model_sweep.py, which these figures reuse "
        "for their loader and OLS.\n  Searched:\n    "
        + "\n    ".join(str(c) for c in ordered[:6])
        + "\n  Keep this file beside benchmarks/, or pass --search-dir."
    )


# benchmarks/ holds both the module imported below and the cost_model_results_*
# data directories, so it is resolved once here and reused as the default
# search path by the plotting scripts.
BENCH_DIR = _locate_benchmarks_dir()
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

# Reuse the sweep analysis's loader, OLS and R^2; the cell-mean step is mirrored from its main().
from analyze_cost_model_sweep import load_rows, ols, r2  # noqa: E402

# Node identity and panel order, fixed by the thesis text. These are labels, not
# measurements; every number attached to them is computed at runtime.
NODE_IDS = ("pi-node-1", "pi-node-2", "pi-node-3")

# The backbone grid is 9 cells x 3 reps by construction (see generate_grid_cells.py).
# Anything else means the CSV is not the dataset these figures describe.
EXPECTED_CELLS = 9

M3_FORMULA = "tpot ~ b0 + b1*B + b2*R_actual + b3*(B*R_actual)"
M3_K = 4  # parameters, for the adj-R^2 penalty


def hardware_label(dir_name):
    """Derive the hardware string from the results directory name, e.g.
    'cost_model_results_pi-node-1_Pi5-8GB' -> 'Pi 5, 8 GB'. Parsed rather than
    tabulated so a renamed or added node cannot silently keep a stale label."""
    suffix = dir_name.split("_", 1)[1] if "_" in dir_name else ""
    suffix = suffix.split("_")[-1]
    m = re.match(r"^(Pi ?5|4B)-(\d+)\s*(?:GB|G|g)$", suffix, re.IGNORECASE)
    if not m:
        return suffix or "unknown hardware"
    model = "Pi 5" if m.group(1).lower().replace(" ", "") == "pi5" else "Pi 4B"
    return f"{model}, {m.group(2)} GB"


def find_node_csv(node_id, search_dir):
    """Locate exactly one summary CSV for a node. Ambiguity is fatal: silently
    picking the first of several sweeps would put an unidentified dataset in a
    thesis figure."""
    pattern = str(Path(search_dir) / f"cost_model_results_{node_id}_*" /
                  "sweep_cost_model_summary.csv")
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise SystemExit(
            f"FATAL: no summary CSV for {node_id}.\n"
            f"  Looked for: {pattern}\n"
            f"  Run from benchmarks/, or pass --search-dir pointing at it."
        )
    if len(hits) > 1:
        raise SystemExit(
            f"FATAL: {len(hits)} summary CSVs match {node_id} -- ambiguous, refusing "
            f"to guess which sweep the figure should show:\n    " + "\n    ".join(hits)
        )
    return Path(hits[0])


def fit_node(node_id, csv_path):
    """Load one node, collapse to cell means, fit M3, and return everything the
    figures and the stdout report need."""
    rows = load_rows(csv_path)

    # --- analyze_cost_model_sweep.py Part 1 filtering, mirrored exactly ---
    clean_rows = [r for r in rows if r["throttled"] != "1"]
    backbone = [r for r in clean_rows if r["block"] == "backbone"]
    n_throttled = len(rows) - len(clean_rows)

    cells = {}
    for r in backbone:
        cells.setdefault(r["cell_label"], []).append(r)

    labels = sorted(cells.keys())
    if len(labels) != EXPECTED_CELLS:
        raise SystemExit(
            f"FATAL: {node_id} yielded {len(labels)} backbone cells, expected "
            f"{EXPECTED_CELLS}. Refusing to plot a partial design.\n"
            f"  CSV:    {csv_path}\n"
            f"  Rows:   {len(rows)} total, {n_throttled} throttled dropped, "
            f"{len(backbone)} backbone kept\n"
            f"  Cells:  {labels}"
        )

    # --- Part 1 cell collapse, mirrored exactly ---
    B_arr, R_arr, tpot_arr, reps = [], [], [], []
    for label in labels:
        cell_rows = cells[label]
        B_arr.append(float(cell_rows[0]["B"]))
        R_arr.append(float(np.mean([float(r["R_actual"]) for r in cell_rows])))
        tpot_arr.append(float(np.mean([float(r["tpot_mean"]) for r in cell_rows])))
        reps.append(len(cell_rows))
    B = np.array(B_arr)
    R = np.array(R_arr)
    measured = np.array(tpot_arr)

    # --- M3, fit on R_actual ---
    X3 = np.column_stack([np.ones(len(measured)), B, R, B * R])
    coefs, rss, fitted, xtx_inv, resid = ols(X3, measured)
    r_squared, adj_r2 = r2(measured, fitted, k=M3_K)

    # Leverage (hat diagonals). Computed, not asserted: the figures highlight
    # whichever cell actually has the highest leverage in this run.
    leverage = np.diag(X3 @ xtx_inv @ X3.T)
    high_lev_idx = int(np.argmax(leverage))

    return {
        "node_id": node_id,
        "csv_path": Path(csv_path),
        "hardware": hardware_label(Path(csv_path).parent.name),
        "labels": labels,
        "B": B,
        "R": R,
        "reps": np.array(reps),
        "measured": measured,
        "fitted": fitted,
        "resid": resid,
        "leverage": leverage,
        "high_lev_idx": high_lev_idx,
        "high_lev_label": labels[high_lev_idx],
        "coefs": coefs,
        "rss": rss,
        "r2": r_squared,
        "adj_r2": adj_r2,
        "df_resid": len(measured) - M3_K,
        "n_rows": len(rows),
        "n_throttled": n_throttled,
        "n_backbone": len(backbone),
    }


def load_all_nodes(search_dir):
    """Fit every node, in the fixed panel order."""
    return [fit_node(nid, find_node_csv(nid, search_dir)) for nid in NODE_IDS]


def print_node_report(fits, stream=sys.stdout):
    """Per-node coefficients, adj-R^2, and every cell's measured/fitted/residual
    -- the numbers to check the figure against before it goes in the thesis."""
    def p(*a):
        print(*a, file=stream)

    p("=" * 88)
    p(f"M3 fit: {M3_FORMULA}")
    p("OLS on 9 backbone cell means per node (throttled rows dropped, "
      "block == 'backbone'), fit on R_actual.")
    p("=" * 88)

    for f in fits:
        p(f"\n{f['node_id']}  ({f['hardware']})")
        p(f"  source: {f['csv_path']}")
        p(f"  rows: {f['n_rows']} total, {f['n_throttled']} throttled dropped, "
          f"{f['n_backbone']} backbone runs -> {len(f['labels'])} cells "
          f"(reps/cell: {sorted(set(f['reps'].tolist()))})")
        b0, b1, b2, b3 = f["coefs"]
        p(f"  b0 (intercept) = {b0:>12.6f}")
        p(f"  b1 (B)         = {b1:>12.6f}")
        p(f"  b2 (R_actual)  = {b2:>12.6f}")
        p(f"  b3 (B:R)       = {b3:>12.6f}")
        p(f"  R2 = {f['r2']:.4f}   adj-R2 = {f['adj_r2']:.4f}   "
          f"RSS = {f['rss']:.2f}   df_resid = {f['df_resid']}")
        p(f"  highest-leverage cell: {f['high_lev_label']} "
          f"(leverage {f['leverage'][f['high_lev_idx']]:.3f})")
        p(f"    {'cell':<11}{'B':>3}{'R_actual':>10}{'measured':>11}"
          f"{'fitted':>11}{'residual':>11}{'leverage':>10}")
        for i, label in enumerate(f["labels"]):
            p(f"    {label:<11}{f['B'][i]:>3.0f}{f['R'][i]:>10.1f}"
              f"{f['measured'][i]:>11.3f}{f['fitted'][i]:>11.3f}"
              f"{f['resid'][i]:>11.3f}{f['leverage'][i]:>10.3f}")

    # The figures highlight a per-node argmax; if the nodes disagree the panels
    # would be marking different cells, which the caption must not claim.
    marked = {f["high_lev_label"] for f in fits}
    if len(marked) > 1:
        p(f"\nNOTE: nodes do not share a highest-leverage cell ({sorted(marked)}). "
          f"Each panel marks its own, so do not describe the marked point as a "
          f"single cell in the caption.")


def resolve_outdir(raw, script_path):
    """Default to the directory the calling script lives in (figures/). Refuse
    to write into any existing results or plots directory -- these figures must
    never overwrite the sweep's own."""
    outdir = Path(raw).expanduser() if raw else Path(script_path).resolve().parent
    outdir = outdir.resolve()
    parts = set(outdir.parts)
    if any(p.startswith("cost_model_results") for p in parts) or "plots" in parts:
        raise SystemExit(
            f"FATAL: refusing to write figures into {outdir} -- that path is inside a "
            f"cost_model_results_* or plots/ directory, where it could overwrite the "
            f"sweep's own resid_vs_fitted.png / predicted_vs_measured.png. "
            f"Pick another --outdir."
        )
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def save(fig, outdir, stem, dpi=600):
    """Vector PDF for LaTeX plus a PNG for quick viewing."""
    written = []
    for ext in ("pdf", "png"):
        path = Path(outdir) / f"{stem}.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.012)
        written.append(path)
    return written
