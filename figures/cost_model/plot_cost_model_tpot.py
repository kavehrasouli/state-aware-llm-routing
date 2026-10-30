"""Plots measured TPOT against total context per node, with the additive fits for the Pi 4Bs."""
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[1] / "benchmarks"
OUT = HERE / "cost_model_tpot.pdf"  # artifact: written next to this script
NODES = [("cost_model_results_pi-node-1_Pi5-8GB", "Pi 5 (8 GB)", False),
         ("cost_model_results_pi-node-2_4B-4g", "Pi 4B (4 GB)", True),
         ("cost_model_results_pi-node-3_4B-8g", "Pi 4B (8 GB)", True)]
SERIES = {1: ("#2a78d6", "o"), 2: ("#eb6834", "s"), 4: ("#1baf7a", "^")}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def cell_means(folder):
    cells = {}
    with open(BENCH / folder / "sweep_cost_model_summary.csv") as f:
        for r in csv.DictReader(f):
            if r["block"] != "backbone" or r["throttled"] not in ("0", ""):
                continue
            cells.setdefault((int(r["B"]), float(r["R_actual"])), []).append(float(r["tpot_mean"]))
    return {k: float(np.mean(v)) for k, v in cells.items()}


def main():
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), constrained_layout=True)
    for ax, (folder, title, fit) in zip(axes, NODES):
        means = cell_means(folder)
        if fit:
            X = np.array([[1, b, r] for (b, r) in means])
            y = np.array(list(means.values()))
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        for b, (color, marker) in SERIES.items():
            pts = sorted((r, t) for (bb, r), t in means.items() if bb == b)
            if not pts:
                continue
            rs, ts = zip(*pts)
            ax.plot(rs, ts, color=color, lw=2, marker=marker, ms=5,
                    markeredgecolor="white", markeredgewidth=1, label=f"B = {b}", zorder=3)
            if fit:
                grid = np.linspace(min(rs), max(rs), 20)
                ax.plot(grid, coef[0] + coef[1] * b + coef[2] * grid, color=color, lw=1,
                        ls="--", zorder=2)
            dy = -9 if (title, b) == ("Pi 4B (8 GB)", 2) else 0  # avoid the B=4 line
            ax.annotate(f"B={b}", (rs[-1], ts[-1]), xytext=(4, dy), textcoords="offset points",
                        va="center", fontsize=7, color=INK)
        ax.set_title(title, fontsize=8.5, color=INK)
        ax.set_xlabel("Total context R (tokens)")
        ax.set_xlim(0, 3900)
        ax.grid(True, color=GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("TPOT (ms)")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=MUTED, lw=1, ls="--"))
    labels.append("additive model (Pi 4B)")
    fig.legend(handles, labels, loc="outside upper center", ncol=4, frameon=False, fontsize=7.5)
    fig.savefig(OUT)
    fig.savefig(OUT.with_suffix(".png"), dpi=160)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
