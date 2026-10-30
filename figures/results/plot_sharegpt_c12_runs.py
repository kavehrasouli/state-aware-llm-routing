import csv
import json
import math
import os
from pathlib import Path
from statistics import median
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "3e_results/analysis"
ARMS = ("weighted_least_connections", "state_aware")
COLORS = ("#697586", "#0072B2")
STEM = "sharegpt_c12_state_aware_vs_wlc_individual_runs"


def percentile(values, p):
    """Linear interpolation, matching analyze_3e.pct."""
    xs = sorted(values)
    rank = p / 100 * (len(xs) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (rank - lo)


def extract():
    rows = []
    with (DATA / "run_inclusion.csv").open(newline="") as stream:
        for entry in csv.DictReader(stream):
            if not (entry["included"].lower() == "true"
                    and entry["arm"] in ARMS
                    and entry["workload"] == "sharegpt"
                    and entry["concurrency"] == "12"):
                continue
            # All six runs per arm passed the existing validity gate here.
            assert entry["verdict"] == "clean"
            path = ROOT / "3e_results" / entry["metadata_path"]
            metadata = json.loads(path.read_text())
            bench = json.loads((path.parent / metadata["bench_result_file"]).read_text())
            raw = bench["raw"]
            assert len(raw) == bench["num_prompts"] == 100
            latency = [r["e2el_ms"] / 1000 for r in raw
                       if r.get("success")
                       and r.get("truncated") not in (1, True, "1", "true", "True")
                       and isinstance(r.get("e2el_ms"), (int, float))]
            completed = sum(bool(r.get("success")) for r in raw)
            assert completed == int(entry["completed"])
            rows.append(dict(arm=entry["arm"], repetition=int(entry["rep"]),
                             verdict=entry["verdict"], offered=100,
                             completed=completed, latency_requests=len(latency),
                             e2el_p50_s=percentile(latency, 50),
                             e2el_p95_s=percentile(latency, 95),
                             metadata_path=str(path.relative_to(ROOT))))
    rows.sort(key=lambda r: (ARMS.index(r["arm"]), r["repetition"]))
    for arm, total in zip(ARMS, (599, 592)):
        rs = [r for r in rows if r["arm"] == arm]
        assert len(rs) == 6 and sum(r["completed"] for r in rs) == total
    with (DATA / "sq3/b/sq3b_vs_wlc_state_aware_report_table.csv").open(newline="") as stream:
        comparisons = {r["metric"]: r for r in csv.DictReader(stream)
                       if r["cell"] == "c12_sgpt"}
    for metric in ("e2el_p50", "e2el_p95"):
        values = [median([r[metric + "_s"] for r in rows if r["arm"] == arm])
                  for arm in ARMS]
        reference = comparisons[metric]
        assert math.isclose(values[0], float(reference["base_median"]), abs_tol=1e-9)
        assert math.isclose(values[1], float(reference["cand_median"]), abs_tol=1e-9)
        assert math.isclose(values[1] - values[0], float(reference["difference"]), abs_tol=1e-9)
    return rows, comparisons


def main():
    rows, comparisons = extract()
    csv_path = HERE / f"{STEM}.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.unicode_minus": True})
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.2))
    fig.subplots_adjust(left=0.067, right=0.985, bottom=0.285, top=0.765, wspace=0.45)
    settings = [
        ("e2el_p50_s", "(a) Median end-to-end latency", "Seconds", (38, 54), [40, 44, 48, 52]),
        ("e2el_p95_s", "(b) P95 end-to-end latency", "Seconds", (73, 102), [75, 80, 85, 90, 95, 100]),
        ("completed", "(c) Completed requests", "Requests (out of 100)", (96.5, 100.6), [97, 98, 99, 100]),
    ]
    # Fixed horizontal displacement distinguishes overlapping observations.
    # It carries no numerical meaning, and no runs are connected across arms.
    offsets = (-0.15, -0.09, -0.03, 0.03, 0.09, 0.15)
    for ax, (metric, title, ylabel, limits, ticks) in zip(axes, settings):
        for x, (arm, color) in enumerate(zip(ARMS, COLORS)):
            values = [r[metric] for r in rows if r["arm"] == arm]
            ax.hlines(median(values), x - 0.26, x + 0.26,
                      color="#222222", linewidth=1.5, zorder=2)
            ax.scatter([x + d for d in offsets], values, s=20,
                       facecolor=color, edgecolor="white", linewidth=0.5,
                       zorder=3)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(*limits)
        ax.set_yticks(ticks)
        ax.set_xticks([0, 1], ["WLC\n(n = 6)", "State-aware\n(n = 6)"])
        ax.set_ylabel(ylabel, fontsize=7.5, labelpad=4)
        ax.set_title(title, fontsize=8, fontweight="bold", pad=28)
        ax.tick_params(axis="x", length=0, pad=5, labelsize=7.5)
        ax.tick_params(axis="y", length=2.5, labelsize=7, color="#777777")
        ax.grid(axis="y", linewidth=0.5, color="#e1e4e8")
        ax.set_axisbelow(True)
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color("#aaaaaa")
        ax.spines["left"].set_linewidth(0.6)
        if metric != "completed":
            row = comparisons[metric.removesuffix("_s")]
            effect = float(row["difference"])
            signed = f"{effect:+.2f}".replace("-", "−")
            text = f"Difference: {signed} s\nAdjusted p = {float(row['p_adj_B']):.3f}"
        else:
            text = "Totals: 599/600 vs 592/600\nDescriptive completion counts"
        ax.text(0.5, 1.045, text, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=6.8, linespacing=1.25, color="#444444")
    fig.text(0.5, 0.047, "ShareGPT c12  •  Each point: one run  •  Black bars: medians across runs",
             ha="center", fontsize=7.3, color="#444444")
    for ext in ("pdf", "png"):
        path = HERE / f"{STEM}.{ext}"
        fig.savefig(path, dpi=300, facecolor="white")
        print(path)
    plt.close(fig)
    print(csv_path)
    print("Verified 12 runs and both latency contrasts against the existing report table.")


if __name__ == "__main__":
    main()
