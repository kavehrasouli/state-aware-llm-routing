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

# Ordinal blue ramp for B = 1, 2, 4; marker shape is a redundant encoding.
B_RAMP = ("#5598e7", "#256abf", "#0d366b")   # light -> dark, ascending B
B_MARKERS = ("o", "s", "^")
N_B_LEVELS = len(B_RAMP)

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
AXIS = "#bdbcb7"
ZERO_RULE = "#8f8e89"
SURFACE = "#ffffff"

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


def batch_levels(fits):
    """Discover the B levels present, ascending. The ramp above was validated
    for exactly three ordered levels, so anything else is fatal rather than
    quietly recycled -- a recycled hue would break the ordering the figure
    exists to show."""
    levels = sorted({float(b) for f in fits for b in f["B"]})
    if len(levels) != N_B_LEVELS:
        raise SystemExit(
            f"FATAL: found {len(levels)} distinct B levels {levels}, but the ordinal "
            f"palette was validated for exactly {N_B_LEVELS}. Refusing to cycle or "
            f"invent a hue -- re-validate a ramp of the right length first "
            f"(see the validator invocation quoted at the top of this file)."
        )
    return levels


def shared_ylim(fits, pad_frac=0.10):
    """One residual window covering every node, symmetric about zero so sign and
    magnitude read the same way in each panel."""
    m = float(np.max([np.max(np.abs(f["resid"])) for f in fits]))
    m = m if m > 0 else 1.0
    return -m * (1 + pad_frac), m * (1 + pad_frac)


def panel_xlim(f, pad_frac=0.10):
    lo, hi = float(f["fitted"].min()), float(f["fitted"].max())
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
    ap.add_argument("--height", type=float, default=1.55, help="figure height in inches")
    ap.add_argument("--dpi", type=int, default=600, help="PNG resolution (default 600)")
    ap.add_argument("--stem", default="cost_model_residuals_vs_fitted",
                    help="output filename stem (no extension)")
    args = ap.parse_args()

    search_dir = args.search_dir or BENCH_DIR
    fits = load_all_nodes(search_dir)
    print_node_report(fits)

    outdir = resolve_outdir(args.outdir, __file__)
    levels = batch_levels(fits)
    ylo, yhi = shared_ylim(fits)

    with plt.rc_context(RC):
        # Constrained layout places the shared labels and the outside legend
        # without them colliding with the panels or with each other.
        fig, axes = plt.subplots(1, len(fits), figsize=(args.width, args.height),
                                 sharey=True, layout="constrained")
        fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.03, hspace=0.0)
        axes = np.atleast_1d(axes)

        for ax, f in zip(axes, fits):
            ax.set_axisbelow(True)
            ax.grid(True, color=GRID, linewidth=0.4, linestyle="-")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)

            xlo, xhi = panel_xlim(f)
            # Zero rule first, so points sit on top of it.
            ax.plot([xlo, xhi], [0, 0], color=ZERO_RULE, linewidth=0.6,
                    linestyle="-", zorder=1)

            for j, lev in enumerate(levels):
                sel = f["B"] == lev
                if not np.any(sel):
                    continue
                ax.scatter(f["fitted"][sel], f["resid"][sel],
                           s=15, marker=B_MARKERS[j], c=B_RAMP[j],
                           edgecolors=SURFACE, linewidths=0.35, zorder=3)

            ax.set_xlim(xlo, xhi)
            ax.set_ylim(ylo, yhi)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.tick_params(length=2.0, pad=1.5)

            ax.set_title(f"{f['node_id']}\n{f['hardware']}", linespacing=1.15, pad=2)

        # One shared x-axis label on the middle panel (one per panel does not fit).
        axes[len(axes) // 2].set_xlabel("Fitted TPOT (ms)", labelpad=1.5)
        fig.supylabel("Residual (ms)", fontsize=RC["axes.labelsize"])

        handles = [
            Line2D([], [], marker=B_MARKERS[j], color="none",
                   markerfacecolor=B_RAMP[j], markeredgecolor=SURFACE,
                   markeredgewidth=0.35, markersize=3.4, label=f"B = {lev:g}")
            for j, lev in enumerate(levels)
        ]
        fig.legend(handles=handles, loc="outside lower center", ncol=len(levels),
                   handlelength=1.2, handletextpad=0.35, columnspacing=1.0,
                   borderpad=0.0, borderaxespad=0.15)

        written = save(fig, outdir, args.stem, dpi=args.dpi)
        plt.close(fig)

    print(f"\nModel plotted: {M3_FORMULA}")
    print(f"B levels encoded (ascending, light -> dark): "
          f"{[f'{lev:g}' for lev in levels]}")
    print(f"Shared residual window (computed from pooled residuals): "
          f"[{ylo:.2f}, {yhi:.2f}] ms")
    print("Per-panel fitted-value windows:")
    for f in fits:
        xlo, xhi = panel_xlim(f)
        print(f"  {f['node_id']:<12} [{xlo:.2f}, {xhi:.2f}] ms")
    for path in written:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
