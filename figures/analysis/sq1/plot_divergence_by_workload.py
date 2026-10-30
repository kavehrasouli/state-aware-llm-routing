import argparse
import csv
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "3e_results" / "analysis").is_dir())
ANALYSIS = REPO / "3e_results" / "analysis"
CSV_PATH = ANALYSIS / "sq1" / "sq1_divergence.csv"

ARM = "weighted_least_connections"

WORKLOAD_ORDER = [("uni", "uniform"), ("bim", "bimodal"), ("sgpt", "ShareGPT")]

SERIES = [
    ("divergence_rate_b_vs_bw", "least connections vs weighted least connections",
     "#1E90FF"),
    ("divergence_rate_bw_vs_rw", "weighted least connections vs state aware",
     "#CD1076"),
]

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
AXIS = "#bdbcb7"
ZERO_RULE = "#5c5b57"
BAND = "#f2f1ee"
SURFACE = "#ffffff"

RC = {
    "font.size": 6.8,
    "axes.titlesize": 7.0,
    "axes.labelsize": 6.8,
    "xtick.labelsize": 6.0,
    "ytick.labelsize": 6.0,
    "legend.fontsize": 5.9,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 0.0,
    "ytick.major.size": 2.0,
    "grid.linewidth": 0.4,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "text.color": TEXT_PRIMARY,
    "axes.labelcolor": TEXT_PRIMARY,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "legend.frameon": False,
    "axes.grid": False,
}


def conc_of(cell):
    """'c12_bim' -> 12. Used only to order cells within a workload group."""
    return int(cell.split("_")[0].lstrip("c"))


def load():
    if not CSV_PATH.exists():
        raise SystemExit(f"FATAL: {CSV_PATH} not found. Run analyze_sq1.py first.")
    with open(CSV_PATH) as f:
        rows = [r for r in csv.DictReader(f)
                if r["clean"] == "True" and r["arm"] == ARM]
    if not rows:
        raise SystemExit(f"FATAL: no rows with clean==True and arm=={ARM} in {CSV_PATH}.")

    groups = []
    for suffix, label in WORKLOAD_ORDER:
        cells = sorted((r for r in rows if r["cell"].endswith("_" + suffix)),
                       key=lambda r: conc_of(r["cell"]))
        cells = [r for r in cells if r[SERIES[0][0] + "_mean"] not in ("", None)]
        if cells:
            groups.append((label, cells))
    if not groups:
        raise SystemExit("FATAL: no plottable cells after filtering.")
    return groups


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=str(HERE))
    ap.add_argument("--width", type=float, default=3.33,
                    help="figure width in inches (default 3.33 = ACM single column)")
    ap.add_argument("--height", type=float, default=2.25)
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--stem", default="sq1_divergence_by_workload")
    args = ap.parse_args()

    groups = load()

    bar_w, cell_w, group_gap = 0.40, 1.0, 0.85
    x, cursor = [], 0.0
    group_spans = []
    for _, cells in groups:
        start = cursor
        for _ in cells:
            x.append(cursor + cell_w / 2)
            cursor += cell_w
        group_spans.append((start, cursor))
        cursor += group_gap
    x = np.array(x)

    flat = [r for _, cells in groups for r in cells]
    means = {k: np.array([float(r[k + "_mean"]) for r in flat]) for k, _, _ in SERIES}
    stds = {k: np.array([float(r[k + "_std"]) for r in flat]) for k, _, _ in SERIES}
    n_runs = [int(r["n_runs"]) for r in flat]
    labels = [r["cell"] for r in flat]

    print("=" * 88)
    print(f"Source: {CSV_PATH}")
    print(f"Filter: clean == True, arm == {ARM}")
    print("=" * 88)
    print(f"{'cell':<9}{'n_runs':>7}{'b_vs_bw':>10}{'±sd':>9}{'lo':>9}"
          f"{'bw_vs_rw':>11}{'±sd':>9}{'lo':>9}")
    k1, k2 = SERIES[0][0], SERIES[1][0]
    for i, cell in enumerate(labels):
        lo1 = means[k1][i] - stds[k1][i]
        lo2 = means[k2][i] - stds[k2][i]
        print(f"{cell:<9}{n_runs[i]:>7}{means[k1][i]:>10.4f}{stds[k1][i]:>9.4f}{lo1:>9.4f}"
              f"{means[k2][i]:>11.4f}{stds[k2][i]:>9.4f}{lo2:>9.4f}")

    below = [(labels[i], key, means[key][i] - stds[key][i])
             for key in (k1, k2) for i in range(len(labels))
             if means[key][i] - stds[key][i] < 0]
    print(f"\nBars whose lower whisker falls below zero: {len(below)}")
    for cell, key, lo in sorted(below, key=lambda t: t[2]):
        print(f"  {cell:<9} {key:<26} lower = {lo:+.4f}")

    lo_all = min((means[k] - stds[k]).min() for k in means)
    hi_all = max((means[k] + stds[k]).max() for k in means)
    pad = (hi_all - lo_all) * 0.10
    ylo, yhi = lo_all - pad, hi_all + pad
    print(f"\nLowest whisker {lo_all:+.4f}, highest whisker {hi_all:+.4f}, pad {pad:.4f}")
    print(f"Computed y limits: [{ylo:.4f}, {yhi:.4f}]  (not anchored at zero, not clipped)")

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(args.width, args.height))
        fig.subplots_adjust(left=0.175, right=0.995, top=0.98, bottom=0.32)


        for (label, cells), (s, e) in zip(groups, group_spans):
            ax.axvspan(s - 0.06, e + 0.06, color=BAND, lw=0, zorder=0)

        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=GRID, linewidth=0.4, linestyle="-", zorder=1)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

        for si, (key, series_label, colour) in enumerate(SERIES):
            off = (si - 0.5) * bar_w
            ax.bar(x + off, means[key], bar_w, yerr=stds[key],
                   color=colour, edgecolor=SURFACE, linewidth=0.35, zorder=3,
                   error_kw=dict(ecolor=TEXT_PRIMARY, elinewidth=0.6,
                                 capsize=1.4, capthick=0.6, zorder=4))

            for xi, m in zip(x + off, means[key]):
                if m == 0:
                    ax.plot([xi - bar_w / 2, xi + bar_w / 2], [0, 0],
                            color=colour, linewidth=1.3, solid_capstyle="butt",
                            zorder=5)


        ax.axhline(0, color=ZERO_RULE, linewidth=0.7, zorder=2)

        ax.set_ylim(ylo, yhi)
        ax.set_xlim(group_spans[0][0] - 0.18, group_spans[-1][1] + 0.18)
        ax.set_xticks(x)
        ax.set_xticklabels([f"c{conc_of(c)}" for c in labels])
        ax.tick_params(axis="x", pad=1.0)
        ax.set_ylabel("Divergence rate", labelpad=2)


        trans = ax.get_xaxis_transform()
        for xi, n in zip(x, n_runs):
            ax.text(xi, -0.085, f"n={n}", transform=trans, ha="center", va="top",
                    fontsize=5.2, color=TEXT_SECONDARY, clip_on=False)

        group_labels = []
        for (label, _), (s, e) in zip(groups, group_spans):
            group_labels.append(
                ax.text((s + e) / 2, -0.175, label, transform=trans, ha="center",
                        va="top", fontsize=6.4, color=TEXT_PRIMARY, clip_on=False))


        fig.canvas.draw()
        inv = ax.transData.inverted()
        need_lo, need_hi = ax.get_xlim()
        for t in group_labels:
            bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
            need_lo = min(need_lo, inv.transform((bb.x0, 0))[0])
            need_hi = max(need_hi, inv.transform((bb.x1, 0))[0])
        if (need_lo, need_hi) != ax.get_xlim():
            ax.set_xlim(need_lo - 0.08, need_hi + 0.08)
            print(f"x window widened to fit group labels: "
                  f"[{need_lo - 0.08:.2f}, {need_hi + 0.08:.2f}]")

        handles = [Patch(facecolor=c, edgecolor=SURFACE, linewidth=0.35, label=l)
                   for _, l, c in SERIES]

        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        lowest = min(t.get_window_extent(renderer=r)
                     .transformed(fig.transFigure.inverted()).y0
                     for t in group_labels)
        LEGEND_GAP = 0.055
        legend_x = ax.get_position().x0
        fig.legend(handles=handles, loc="upper left",
                   bbox_to_anchor=(legend_x, lowest - LEGEND_GAP),
                   bbox_transform=fig.transFigure, ncol=1,
                   handlelength=1.5, handletextpad=0.45, borderpad=0.0,
                   labelspacing=0.30)
        print(f"Workload labels end at figure y={lowest:.3f}; legend top placed at "
              f"y={lowest - LEGEND_GAP:.3f}, left-aligned to the axes at x={legend_x:.3f}")

        outdir = Path(args.outdir).expanduser().resolve()
        outdir.mkdir(parents=True, exist_ok=True)
        written = []
        for ext in ("pdf", "png"):
            p = outdir / f"{args.stem}.{ext}"
            fig.savefig(p, dpi=args.dpi, bbox_inches="tight", pad_inches=0.015)
            written.append(p)
        plt.close(fig)

    for p in written:
        print(f"Wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
