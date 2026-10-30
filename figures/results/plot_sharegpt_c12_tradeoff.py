import csv
import os
from pathlib import Path
from statistics import median
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Variant of plot_sharegpt_c12_compact.py that shows both sides of the
# ShareGPT c12 trade-off (lower median, higher p95) instead of p95 and
# completion. Same data source and styling; the original is left untouched.
HERE = Path(__file__).resolve().parent
SOURCE = HERE / "sharegpt_c12_state_aware_vs_wlc_individual_runs.csv"
STEM = "sharegpt_c12_tradeoff"
# Grey baseline; state-aware in the green that marks resident context in Fig. 1b.
ARMS = (("weighted_least_connections", "WLC", "#7A7A7A"),
        ("state_aware", "State-aware", "#009E73"))
JITTER = (-0.2, 0.08, -0.06, 0.2, -0.13, 0.14)
INK, MUTED = "#222222", "#555555"


def load():
    with SOURCE.open(newline="") as stream:
        runs = list(csv.DictReader(stream))
    out = {}
    for arm, *_ in ARMS:
        rs = sorted((r for r in runs if r["arm"] == arm), key=lambda r: int(r["repetition"]))
        assert len(rs) == 6 and all(r["verdict"] == "clean" for r in rs)
        out[arm] = {"p50": [float(r["e2el_p50_s"]) for r in rs],
                    "p95": [float(r["e2el_p95_s"]) for r in rs]}
    return out


def panel(ax, data, key, title, note, ylabel):
    for x, (arm, _, color) in enumerate(ARMS):
        ys = data[arm][key]
        ax.scatter([x + j for j in JITTER], ys, s=13, color=color,
                   edgecolor="white", linewidth=0.5, zorder=5)
        m = median(ys)
        ax.hlines(m, x - 0.3, x + 0.3, color=INK, linewidth=1.3, zorder=4)
    ax.set_xticks(range(2), [name for _, name, _ in ARMS])
    ax.set_xlim(-0.55, 1.55)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK)
    ax.set_title(title, loc="left", fontsize=7.5, fontweight="bold", pad=13)
    ax.text(0, 1.02, note, transform=ax.transAxes, fontsize=6.3, color=MUTED,
            ha="left", va="bottom")
    ax.grid(axis="y", color="#e0e3e6", linewidth=0.45, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#aaaaaa")
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(length=2.5, color="#777777")
    ax.tick_params(axis="x", length=0)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7,
                         "axes.labelsize": 7, "xtick.labelsize": 7,
                         "ytick.labelsize": 6.8, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.unicode_minus": True})
    data = load()
    wlc, sa = data["weighted_least_connections"], data["state_aware"]
    p50_diff = median(sa["p50"]) - median(wlc["p50"])
    p95_diff = median(sa["p95"]) - median(wlc["p95"])
    # Must match the paper's reported effects (sq3b_bh_corrected.csv).
    assert abs(p50_diff - (-5.78)) < 0.01
    assert abs(p95_diff - 12.35) < 0.01

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.72))
    fig.subplots_adjust(left=0.135, right=0.985, bottom=0.14, top=0.78, wspace=0.45)
    panel(ax1, data, "p50", "(a) Median latency",
          f"{p50_diff:+.2f} s, adj. p = 0.035".replace("-", "−"), "Seconds")
    ax1.set_ylim(37, 54)
    panel(ax2, data, "p95", "(b) P95 latency",
          f"{p95_diff:+.2f} s, adj. p = 0.030", None)
    ax2.set_ylim(72, 102)
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"{STEM}.{ext}", dpi=300, facecolor="white")
        print(HERE / f"{STEM}.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()
