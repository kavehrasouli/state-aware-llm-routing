"""Paper Fig. 1: latency differences from WLC, and decision and freshness shares."""

import plot_results_overview as base
import plot_results_overview_noselect as nos
import plot_thesis_figures as thesis
from matplotlib.lines import Line2D

plt, MUTED = base.plt, base.MUTED


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7,
                         "axes.labelsize": 7, "xtick.labelsize": 6.8,
                         "ytick.labelsize": 7.2, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.unicode_minus": True})
    lat, diag = base.read_latency(), base.read_diagnostics()
    fig = plt.figure(figsize=(7.16, 2.95))
    ax_l = fig.add_axes([0.115, 0.285, 0.40, 0.635])
    ax_r = fig.add_axes([0.575, 0.285, 0.395, 0.635])
    thesis.draw_one(ax_l, lat, "wlc", "(a) Service outcomes vs WLC", True)
    ax_l.set_xlabel("Latency difference (s): state-aware minus WLC", color=base.INK)
    nos.draw_diag_2d(ax_r, diag)
    left = [Line2D([], [], color=c, marker="o", markersize=4, linestyle="none", label=n)
            for _, n, c in thesis.METRICS]
    left += [Line2D([], [], color=MUTED, marker="o", markersize=4, linestyle="none",
                    label="Filled: adj. p < 0.05"),
             Line2D([], [], color=MUTED, marker="o", markersize=4, markerfacecolor="white",
                    linestyle="none", label="Open: not significant")]
    # No WLC comparison is leave-one-out sensitive, so no dagger entry.
    fig.legend(handles=left, loc="lower left", bbox_to_anchor=(0.02, 0.012), ncol=2,
               frameon=False, fontsize=6.3, handletextpad=0.3, columnspacing=0.9,
               borderaxespad=0)
    right = [Line2D([], [], color=c, marker=m, markersize=4, linestyle="none", label=n)
             for _, n, c, m in nos.DIAG]
    fig.legend(handles=right, loc="lower left", bbox_to_anchor=(0.55, 0.012), ncol=1,
               frameon=False, fontsize=6.3, handletextpad=0.3, borderaxespad=0,
               labelspacing=0.25)
    base.save(fig, "results_overview_wlc")


if __name__ == "__main__":
    main()
