import csv
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "3e_results/analysis"
WORKLOADS = (("uni", "Uniform"), ("bim", "Bimodal"), ("sgpt", "ShareGPT"))
CONCS = (4, 8, 12)
CELLS = [f"c{c}_{w}" for w, _ in WORKLOADS for c in CONCS]
LABELS = [f"{name}  c{c}" for _, name in WORKLOADS for c in CONCS]

# Left panel: colour = metric, shape = baseline, fill = BH significance.
METRICS = (("e2el_p50", "Median", "#0072B2"), ("e2el_p95", "P95", "#D55E00"))
BASELINES = (("wlc", "vs WLC", "o", 4.0), ("wrr", "vs WRR", "D", 3.4))
OFFSETS = {("wlc", "e2el_p50"): -0.30, ("wlc", "e2el_p95"): -0.10,
           ("wrr", "e2el_p50"): 0.10, ("wrr", "e2el_p95"): 0.30}

# Right panel series; hues distinct from the left panel's metric colours.
DIAG = (("cap", "Capacity weighting changes choice", "#7A7A7A", "s"),
        ("ctx", "Resident context changes choice", "#009E73", "o"),
        ("stale", "Decisions with a stale replica", "#CC79A7", "^"))
INK, MUTED, GRID, BAND = "#222222", "#555555", "#e0e3e6", "#f0f2f4"


def rows(path, keep=lambda r: True):
    with path.open(newline="") as stream:
        return [r for r in csv.DictReader(stream) if keep(r)]


def read_latency():
    out = {}
    for base, *_ in BASELINES:
        table = {(r["cell"], r["metric"]): r for r in
                 rows(DATA / f"sq3/b/sq3b_vs_{base}_state_aware_report_table.csv")}
        loo = {(r["cell"], r["metric"]): r["loo_damaging"].lower() == "true" for r in
               rows(DATA / f"sq3/b/sq3b_vs_{base}.csv", lambda r: r["arm"] == "state_aware")}
        for cell in CELLS:
            for metric, *_ in METRICS:
                r = table[cell, metric]
                assert r["unit"] == "s" and r["run_scope"] == "clean"
                d, lo, hi, q = (float(r[k]) for k in
                                ("difference", "ci95_low", "ci95_high", "p_adj_B"))
                assert (r["significant_B"].lower() == "true") == (q < 0.05)
                out[base, cell, metric] = (d, lo, hi, q < 0.05, loo[cell, metric])
    return out


def read_diagnostics():
    div = {r["cell"]: r for r in rows(DATA / "sq1/sq1_divergence.csv", lambda r:
           r["arm"] == "weighted_least_connections" and r["clean"] == "True")}
    handling = rows(DATA / "sq2/sq2_stale_handling.csv", lambda r: r["scope"] == "clean")
    state_arms = {r["arm"] for r in handling
                  if int(r["mode_stateless_count"]) == 0 and int(r["n_requests"]) > 0}
    overhead = {(r["arm"], r["cell"]): float(r["select_us_median_pooled"]) for r in
                rows(DATA / "sq2/sq2_routing_overhead.csv",
                     lambda r: r["scope"] == "all" and r["select_us_median_pooled"])}
    out = {}
    for cell in CELLS:
        hs = [r for r in handling if r["cell"] == cell and r["arm"] in state_arms]
        n = sum(int(r["n_requests"]) for r in hs)
        stale = sum(int(r[f"mode_{m}_count"]) for r in hs
                    for m in ("some_stale_excluded", "all_stale_rr", "all_stale_compare"))
        out[cell] = {"cap": 100 * float(div[cell]["divergence_rate_b_vs_bw_mean"]),
                     "ctx": 100 * float(div[cell]["divergence_rate_bw_vs_rw_mean"]),
                     "stale": 100 * stale / n,
                     "sa_us": overhead["state_aware", cell],
                     "wrr_us": overhead["weighted_round_robin", cell]}
    return out


def style_rows(ax, xlim):
    ax.axhspan(2.5, 5.5, color=BAND, zorder=0)
    for y in (2.5, 5.5):
        ax.axhline(y, color="#d3d7dc", linewidth=0.5, zorder=1)
    ax.set_xlim(*xlim)
    ax.set_ylim(8.6, -0.6)
    ax.grid(axis="x", color=GRID, linewidth=0.45, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "left", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#aaaaaa")
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(axis="x", length=2.5, color="#777777")
    ax.tick_params(axis="y", length=0, pad=5)


def draw_latency(ax, lat):
    style_rows(ax, (-32, 44))
    ax.axvline(0, color=MUTED, linewidth=0.8, zorder=2)
    for i, cell in enumerate(CELLS):
        for base, _, marker, size in BASELINES:
            for metric, _, color in METRICS:
                d, lo, hi, sig, loo = lat[base, cell, metric]
                y = i + OFFSETS[base, metric]
                ax.hlines(y, lo, hi, color=color, linewidth=0.8, zorder=3)
                ax.plot(d, y, marker=marker, markersize=size, linestyle="none",
                        markerfacecolor=color if sig else "white",
                        markeredgecolor=color, markeredgewidth=0.8, zorder=4)
                if sig and loo:
                    ax.annotate("†", (d, y), xytext=(3.5, -1.5), textcoords="offset points",
                                color=color, fontsize=6.5, fontweight="bold", zorder=5)
    ax.set_xticks(range(-30, 41, 10))
    ax.set_yticks(range(9), LABELS)
    ax.set_xlabel("Latency difference (s): state-aware minus baseline", color=INK)
    ax.set_title("(a) Service outcomes", loc="left", fontsize=8.5, fontweight="bold", pad=6)


def draw_diag_2d(ax, diag):
    style_rows(ax, (-2.5, 92))
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
    # Selection overhead: a different unit, so a text column, not a second axis.
    x0 = 1.035
    ax.text(x0, -0.95, "Select (µs)", transform=ax.get_yaxis_transform(),
            fontsize=6.5, color=INK, fontweight="bold", ha="left", va="bottom")
    ax.text(x0, -0.55, "SA   WRR", transform=ax.get_yaxis_transform(),
            fontsize=6.2, color=MUTED, ha="left", va="bottom")
    for i, cell in enumerate(CELLS):
        ax.text(x0, i, f"{diag[cell]['sa_us']:.1f}   {diag[cell]['wrr_us']:.1f}",
                transform=ax.get_yaxis_transform(), fontsize=6.4, color=INK,
                ha="left", va="center")


def draw_diag_3d(fig, rect, diag):
    ax = fig.add_axes(rect, projection="3d")
    width, depth = 0.22, 0.5
    for k, (key, _, color, _) in enumerate(DIAG):
        for wi, (w, _) in enumerate(WORKLOADS):
            for ci, c in enumerate(CONCS):
                x = ci + (k - 1) * width - width / 2
                ax.bar3d(x, wi - depth / 2, 0, width, depth, diag[f"c{c}_{w}"][key],
                         color=color, edgecolor="white", linewidth=0.3, shade=True)
    ax.set_xticks(range(3), [f"c{c}" for c in CONCS])
    ax.set_yticks(range(3), [name for _, name in WORKLOADS])
    ax.set_zlim(0, 90)
    ax.set_zlabel("Share of decisions (%)", labelpad=2)
    ax.view_init(elev=24, azim=-58)
    ax.tick_params(pad=0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("#f7f7f7")
        axis.pane.set_edgecolor("#dddddd")
    ax.set_title("(b) Routing decisions and freshness", loc="left",
                 fontsize=8.5, fontweight="bold", pad=0)
    return ax


def legends(fig, y):
    left = [Line2D([], [], color=c, marker="s", linestyle="none", markersize=4, label=n)
            for _, n, c in METRICS]
    left += [Line2D([], [], color=MUTED, marker=m, markersize=s, markerfacecolor="white",
                    linestyle="none", label=n) for _, n, m, s in BASELINES]
    left += [Line2D([], [], color=MUTED, marker="o", markersize=4, linestyle="none",
                    label="Filled: adj. p < 0.05"),
             Line2D([], [], color="none", marker="$†$", markersize=5,
                    markerfacecolor=MUTED, markeredgecolor=MUTED, label="LOO-sensitive")]
    fig.legend(handles=left, loc="lower left", bbox_to_anchor=(0.02, y), ncol=3,
               frameon=False, fontsize=6.3, handletextpad=0.3, columnspacing=0.9,
               borderaxespad=0)
    right = [Line2D([], [], color=c, marker=m, markersize=4, linestyle="none", label=n)
             for _, n, c, m in DIAG]
    fig.legend(handles=right, loc="lower left", bbox_to_anchor=(0.55, y), ncol=1,
               frameon=False, fontsize=6.3, handletextpad=0.3, borderaxespad=0,
               labelspacing=0.25)


def save(fig, stem):
    for ext in ("pdf", "png"):
        path = HERE / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, facecolor="white")
        print(path)
    plt.close(fig)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7,
                         "axes.labelsize": 7, "xtick.labelsize": 6.8,
                         "ytick.labelsize": 7.2, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.unicode_minus": True})
    lat, diag = read_latency(), read_diagnostics()

    fig = plt.figure(figsize=(7.16, 2.95))
    ax_l = fig.add_axes([0.115, 0.285, 0.40, 0.635])
    ax_r = fig.add_axes([0.575, 0.285, 0.33, 0.635])
    draw_latency(ax_l, lat)
    draw_diag_2d(ax_r, diag)
    legends(fig, 0.012)
    save(fig, "results_overview_2d")

    fig = plt.figure(figsize=(7.16, 3.2))
    ax_l = fig.add_axes([0.115, 0.30, 0.40, 0.62])
    draw_latency(ax_l, lat)
    draw_diag_3d(fig, [0.53, 0.25, 0.46, 0.72], diag)
    legends(fig, 0.015)
    save(fig, "results_overview_3d")


if __name__ == "__main__":
    main()
