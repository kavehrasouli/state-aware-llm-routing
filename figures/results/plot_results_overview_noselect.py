import plot_results_overview as base
from plot_results_overview import CELLS, DIAG, INK, plt

# Variant of results_overview_2d without the per-cell "Select (us)" column; reuses its loaders and styling.

# Panel (a) metric colours, overriding the original's blue/orange. Set on the
# base module so draw_latency and the legend both pick them up.
base.METRICS = (("e2el_p50", "Median", "#4F94CD"), ("e2el_p95", "P95", "#698B69"))

# Panel (b) series colours. Assigned to both names: draw_diag_2d below uses the
# local DIAG, while the shared legend reads base.DIAG.
DIAG = base.DIAG = (
    ("cap", "Capacity weighting changes choice", "#7B4FA0", "s"),
    ("ctx", "Resident context changes choice", "#009E73", "o"),
    ("stale", "Decisions with a stale replica", "#E69F00", "^"),
)


def draw_diag_2d(ax, diag):
    base.style_rows(ax, (-2.5, 92))
    offsets = {"cap": -0.24, "ctx": 0.0, "stale": 0.24}
    for key, _, color, marker in DIAG:
        for i, cell in enumerate(CELLS):
            y, v = i + offsets[key], diag[cell][key]
            ax.hlines(y, 0, v, color=color, linewidth=0.6, alpha=0.55, zorder=2)
            ax.plot(v, y, marker=marker, markersize=3.8, color=color,
                    markeredgecolor="white", markeredgewidth=0.4, linestyle="none", zorder=4)
    ax.set_xticks(range(0, 91, 20))
    ax.set_xticklabels([f"{t}%" for t in range(0, 91, 20)])
    ax.set_yticks(range(9), [""] * 9)
    ax.set_xlabel("Share of routing decisions", color=INK)
    ax.set_title("(b) Decisions and state freshness", loc="left",
                 fontsize=8.5, fontweight="bold", pad=6)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7,
                         "axes.labelsize": 7, "xtick.labelsize": 6.8,
                         "ytick.labelsize": 7.2, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.unicode_minus": True})
    lat, diag = base.read_latency(), base.read_diagnostics()
    fig = plt.figure(figsize=(7.16, 2.95))
    ax_l = fig.add_axes([0.115, 0.285, 0.40, 0.635])
    ax_r = fig.add_axes([0.575, 0.285, 0.395, 0.635])
    base.draw_latency(ax_l, lat)
    draw_diag_2d(ax_r, diag)
    base.legends(fig, 0.012)
    base.save(fig, "results_overview_2d_noselect")


if __name__ == "__main__":
    main()
