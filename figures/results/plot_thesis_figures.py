"""Thesis Fig. 2 (results overview split by baseline) and Fig. 4 (Pi 5 share vs p95 latency)."""
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import plot_results_overview as base  # noqa: E402
import plot_results_overview_noselect as nos  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

plt, INK, MUTED, CELLS = base.plt, base.INK, base.MUTED, base.CELLS
OUT = HERE  # artifact: written next to this script
METRICS = (("e2el_p50", "Median", "#4F94CD"), ("e2el_p95", "P95", "#698B69"))
OFFS = {"e2el_p50": -0.14, "e2el_p95": 0.14}


def draw_one(ax, lat, which, title, labels):
    base.style_rows(ax, (-32, 44))
    ax.axvline(0, color=MUTED, linewidth=0.8, zorder=2)
    for i, cell in enumerate(CELLS):
        for metric, _, color in METRICS:
            d, lo, hi, sig, loo = lat[which, cell, metric]
            y = i + OFFS[metric]
            ax.hlines(y, lo, hi, color=color, linewidth=0.9, zorder=3)
            ax.plot(d, y, marker="o", markersize=4.2, linestyle="none",
                    markerfacecolor=color if sig else "white",
                    markeredgecolor=color, markeredgewidth=0.9, zorder=4)
            if sig and loo:
                ax.annotate("†", (d, y), xytext=(3.5, -1.5), textcoords="offset points",
                            color=color, fontsize=6.5, fontweight="bold", zorder=5)
    ax.set_xticks(range(-30, 41, 15))
    ax.set_yticks(range(9), base.LABELS if labels else [""] * 9)
    ax.set_xlabel("State-aware minus baseline (s)", color=INK)
    ax.set_title(title, loc="left", fontsize=8.5, fontweight="bold", pad=6)


def overview():
    lat, diag = base.read_latency(), base.read_diagnostics()
    fig = plt.figure(figsize=(7.16, 3.35))
    a = fig.add_axes([0.125, 0.33, 0.245, 0.6])
    b = fig.add_axes([0.39, 0.33, 0.245, 0.6])
    c = fig.add_axes([0.675, 0.33, 0.305, 0.6])
    draw_one(a, lat, "wlc", "(a) vs WLC", True)
    draw_one(b, lat, "wrr", "(b) vs WRR", False)
    nos.draw_diag_2d(c, diag)
    c.set_title("(c) Decisions and state freshness", loc="left",
                fontsize=8.5, fontweight="bold", pad=6)
    left = [Line2D([], [], color=col, marker="o", markersize=4, linestyle="none", label=n)
            for _, n, col in METRICS]
    left += [Line2D([], [], color=MUTED, marker="o", markersize=4, linestyle="none",
                    label="Filled: adj. p < 0.05"),
             Line2D([], [], color=MUTED, marker="o", markersize=4, markerfacecolor="white",
                    linestyle="none", label="Open: not significant"),
             Line2D([], [], color="none", marker="$†$", markersize=5,
                    markerfacecolor=MUTED, markeredgecolor=MUTED, label="LOO-sensitive")]
    fig.legend(handles=left, loc="lower left", bbox_to_anchor=(0.02, 0.012), ncol=3,
               frameon=False, fontsize=6.3, handletextpad=0.3, columnspacing=0.9,
               borderaxespad=0)
    # Thesis-only labels naming the replayed rule pairs; the paper figure keeps nos.DIAG's.
    pairs = {"cap": "Capacity weighting changes choice\n(WLC vs least-connections)",
             "ctx": "Resident context changes choice\n(state-aware vs WLC)"}
    right = [Line2D([], [], color=col, marker=m, markersize=4, linestyle="none",
                    label=pairs.get(key, n))
             for key, n, col, m in nos.DIAG]
    fig.legend(handles=right, loc="lower left", bbox_to_anchor=(0.66, 0.012), ncol=1,
               frameon=False, fontsize=6.3, handletextpad=0.3, borderaxespad=0,
               labelspacing=0.25)
    save(fig, "results_overview_split")


ARMS = (("state_aware", "State-aware", "#009E73", "o"),
        ("state_aware_v2", "State-aware-v2", "#56B4E9", "s"),
        ("cost_predict_v3", "Cost-predict-v3", "#C2185B", "D"),
        ("wlc_conc", "Wlc-conc", "#7B4FA0", "^"))


def scatter():
    path = REPO / "3e_results/analysis/sq3/b/sq3b_routing_share_vs_tail.csv"
    rows = [r for r in csv.DictReader(path.open()) if r["metric"] == "e2el_p95"]
    fig, ax = plt.subplots(figsize=(3.5, 2.75))
    fig.subplots_adjust(left=0.15, right=0.97, top=0.97, bottom=0.36)
    ax.axhline(0, color=MUTED, linewidth=0.8, zorder=1)
    ax.axvline(0, color=MUTED, linewidth=0.8, zorder=1)
    # Expected quadrants: more Pi 5 share, lower p95 (and the reverse).
    ax.axvspan(-2, 2, color="#eeeeee", zorder=0)
    ax.fill_between([2, 60], -50, 0, color="#e8f4ee", zorder=0)
    ax.fill_between([-25, -2], 0, 35, color="#e8f4ee", zorder=0)
    for arm, _, color, marker in ARMS:
        for r in rows:
            if r["arm"] != arm:
                continue
            x, y, sig = float(r["pi5_share_diff_pp"]), float(r["latency_diff_s"]), r["significant"] == "True"
            ax.plot(x, y, marker=marker, markersize=4.6 if sig else 3.8, linestyle="none",
                    markerfacecolor=color if sig else "white", markeredgecolor=color,
                    markeredgewidth=0.9, zorder=3 if sig else 2)
    ax.set_xlim(-25, 58)
    ax.set_ylim(-50, 33)
    ax.grid(color=base.GRID, linewidth=0.45, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("Pi 5 dispatch share, policy minus WLC (pp)", color=INK)
    ax.set_ylabel("p95 latency, policy minus WLC (s)", color=INK)
    ax.text(56, -2.5, "more Pi 5,\nlower p95", ha="right", va="top", fontsize=6, color="#2e7d57")
    ax.text(-23.5, 1.5, "less Pi 5,\nhigher p95", ha="left", va="bottom", fontsize=6, color="#2e7d57")
    handles = [Line2D([], [], color=c, marker=m, markersize=4, linestyle="none", label=n)
               for _, n, c, m in ARMS]
    handles += [Line2D([], [], color=MUTED, marker="o", markersize=4, linestyle="none",
                       label="Filled: adj. p < 0.05"),
                Line2D([], [], color=MUTED, marker="o", markersize=4, markerfacecolor="white",
                       linestyle="none", label="Open: not significant"),
                Line2D([], [], color="#dddddd", marker="s", markersize=6, linestyle="none",
                       label="Share change < 2 pp (not counted)")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.03, 0.01), ncol=2,
               frameon=False, fontsize=6.2, handletextpad=0.3, columnspacing=0.9,
               borderaxespad=0, labelspacing=0.3)
    save(fig, "routing_share_tail")


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7,
                         "axes.labelsize": 7, "xtick.labelsize": 6.8,
                         "ytick.labelsize": 7.2, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.unicode_minus": True})
    overview()
    scatter()
