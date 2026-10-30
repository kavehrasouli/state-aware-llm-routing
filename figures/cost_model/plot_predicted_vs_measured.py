import argparse
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from cost_model_figdata import (load_all_nodes, print_node_report, resolve_outdir,
                                save, BENCH_DIR, M3_FORMULA)

# Two categorical colors (checked for color-vision deficiency); marker shape also differs.
SERIES_ORDINARY = "#2a78d6"   # slot 1, blue
SERIES_HIGHLEV = "#eb6834"    # slot 2, orange

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
AXIS = "#bdbcb7"
IDENTITY = "#8f8e89"
SURFACE = "#ffffff"

# Absolute point sizes, chosen to stay legible at ACM single-column scale (the
# figure is placed at 1:1, so these are the sizes that reach the page).
RC = {
    "font.size": 6.8,
    "axes.titlesize": 6.4,
    "axes.labelsize": 6.8,
    "xtick.labelsize": 6.0,
    "ytick.labelsize": 6.0,
    "legend.fontsize": 6.0,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.0,
    "ytick.major.size": 2.0,
    "grid.linewidth": 0.4,
    "lines.linewidth": 0.8,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "text.color": TEXT_PRIMARY,
    "axes.labelcolor": TEXT_PRIMARY,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "axes.titlecolor": TEXT_PRIMARY,
    "pdf.fonttype": 42,   # embed TrueType so LaTeX/print keeps text selectable
    "ps.fonttype": 42,
    "legend.frameon": False,
    "axes.grid": False,
}

MARKER_ORDINARY = "o"
MARKER_HIGHLEV = "D"


def shared_limits(fits, pad_frac=0.06):
    """One square window covering every measured and predicted value on every
    node. Square (identical x and y) so the identity line sits at a true 45deg
    and over/under-prediction reads as distance from it."""
    vals = np.concatenate([np.concatenate([f["measured"], f["fitted"]]) for f in fits])
    lo, hi = float(vals.min()), float(vals.max())
    pad = (hi - lo) * pad_frac if hi > lo else (abs(hi) or 1.0) * pad_frac
    return lo - pad, hi + pad


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--search-dir", default=None,
                    help=f"directory holding cost_model_results_* (default: {BENCH_DIR})")
    ap.add_argument("--outdir", default=None,
                    help="output directory (default: this script's directory)")
    ap.add_argument("--width", type=float, default=3.33,
                    help="figure width in inches (default 3.33 = ACM single column)")
    ap.add_argument("--height", type=float, default=1.72, help="figure height in inches")
    ap.add_argument("--dpi", type=int, default=600, help="PNG resolution (default 600)")
    ap.add_argument("--stem", default="cost_model_predicted_vs_measured",
                    help="output filename stem (no extension)")
    args = ap.parse_args()

    search_dir = args.search_dir or BENCH_DIR
    fits = load_all_nodes(search_dir)
    print_node_report(fits)

    outdir = resolve_outdir(args.outdir, __file__)
    lo, hi = shared_limits(fits)

    with plt.rc_context(RC):
        # Constrained layout places the shared labels and the outside legend
        # without them colliding with the panels or with each other.
        fig, axes = plt.subplots(1, len(fits), figsize=(args.width, args.height),
                                 sharex=True, sharey=True, layout="constrained")
        fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.02, hspace=0.0)
        axes = np.atleast_1d(axes)

        for ax, f in zip(axes, fits):
            ax.set_axisbelow(True)
            ax.grid(True, color=GRID, linewidth=0.4, linestyle="-")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)

            # Identity line first, so data sits on top of it.
            ax.plot([lo, hi], [lo, hi], color=IDENTITY, linewidth=0.6,
                    linestyle="-", zorder=1)

            hi_i = f["high_lev_idx"]
            ordinary = np.ones(len(f["labels"]), dtype=bool)
            ordinary[hi_i] = False

            ax.scatter(f["measured"][ordinary], f["fitted"][ordinary],
                       s=15, marker=MARKER_ORDINARY, c=SERIES_ORDINARY,
                       edgecolors=SURFACE, linewidths=0.35, zorder=3)
            ax.scatter(f["measured"][hi_i], f["fitted"][hi_i],
                       s=20, marker=MARKER_HIGHLEV, c=SERIES_HIGHLEV,
                       edgecolors=SURFACE, linewidths=0.35, zorder=4)

            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
            ax.tick_params(length=2.0, pad=1.5)

            ax.set_title(f"{f['node_id']}\n{f['hardware']}\n"
                         f"adj-$R^2$ = {f['adj_r2']:.3f}",
                         linespacing=1.15, pad=2)

        # One shared x-axis label on the middle panel (one per panel does not fit).
        axes[len(axes) // 2].set_xlabel("Measured TPOT (ms)", labelpad=1.5)
        fig.supylabel("Predicted TPOT (ms)", fontsize=RC["axes.labelsize"])

        # One figure-level legend: identity is never carried by colour alone, and
        # the marked cell is named with the leverage computed in this run.
        lev_labels = sorted({f["high_lev_label"] for f in fits})
        lev_name = lev_labels[0] if len(lev_labels) == 1 else "highest-leverage cell"
        lev_val = np.mean([f["leverage"][f["high_lev_idx"]] for f in fits])
        handles = [
            Line2D([], [], marker=MARKER_ORDINARY, color="none",
                   markerfacecolor=SERIES_ORDINARY, markeredgecolor=SURFACE,
                   markeredgewidth=0.35, markersize=3.4, label="backbone cell"),
            Line2D([], [], marker=MARKER_HIGHLEV, color="none",
                   markerfacecolor=SERIES_HIGHLEV, markeredgecolor=SURFACE,
                   markeredgewidth=0.35, markersize=3.4,
                   label=f"{lev_name} (h = {lev_val:.2f})"),
            Line2D([], [], color=IDENTITY, linewidth=0.6, label="y = x"),
        ]
        fig.legend(handles=handles, loc="outside lower center", ncol=3,
                   handlelength=1.2, handletextpad=0.35, columnspacing=0.9,
                   borderpad=0.0, borderaxespad=0.15)

        written = save(fig, outdir, args.stem, dpi=args.dpi)
        plt.close(fig)

    print(f"\nModel plotted: {M3_FORMULA}")
    print(f"Shared axis window (computed from pooled data): "
          f"[{lo:.2f}, {hi:.2f}] ms on both axes")
    for path in written:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
