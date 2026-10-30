import argparse
import csv
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "3e_results" / "analysis").is_dir())
ANALYSIS = REPO / "3e_results" / "analysis"
for p in (ANALYSIS, ANALYSIS / "sq2"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import analyze_divergence as AD                    # noqa: E402
from analyze_sq2 import scan                        # noqa: E402
from analyze_sq1 import resolve_num_warmup          # noqa: E402

import matplotlib                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.lines import Line2D                 # noqa: E402

STALENESS_CSV = ANALYSIS / "sq2" / "sq2_staleness.csv"
SCOPE = "clean"

NODE_COLOUR = {0: "#2a78d6", 1: "#eb6834", 2: "#1baf7a"}
NODE_MARKER = {0: "o", 1: "s", 2: "^"}
NODE_LINESTYLE = {0: "-", 1: "--", 2: ":"}

TEXT_PRIMARY, TEXT_SECONDARY = "#0b0b0b", "#52514e"
GRID, AXIS, SURFACE = "#e6e5e1", "#bdbcb7", "#ffffff"
RULE_THRESHOLD, RULE_POLL = "#5c5b57", "#9a9994"

RC = {
    "font.size": 6.6, "axes.titlesize": 6.6, "axes.labelsize": 6.6,
    "xtick.labelsize": 5.9, "ytick.labelsize": 5.9, "legend.fontsize": 5.6,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0, "grid.linewidth": 0.4,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.edgecolor": AXIS,
    "text.color": TEXT_PRIMARY, "axes.labelcolor": TEXT_PRIMARY,
    "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "legend.frameon": False, "axes.grid": False,
}


def hardware_labels():
    """Parse 'cost_model_results_pi-node-1_Pi5-8GB' -> 'Pi 5, 8 GB'. Derived so
    a renamed node cannot silently keep a stale label; falls back to the bare
    node name if no directory matches."""
    out = {}
    for node in AD.NODES:
        hits = glob.glob(str(REPO / "benchmarks" / f"cost_model_results_{node}_*"))
        label = None
        if len(hits) == 1:
            suffix = os.path.basename(hits[0]).split("_")[-1]
            m = re.match(r"^(Pi ?5|4B)-(\d+)\s*(?:GB|G|g)$", suffix, re.IGNORECASE)
            if m:
                model = "Pi 5" if m.group(1).lower().replace(" ", "") == "pi5" else "Pi 4B"
                label = f"{model} {m.group(2)} GB"
        out[node] = f"{node} ({label})" if label else node
    return out


def conc_of(cell):
    return int(cell.split("_")[0].lstrip("c"))


def workload_of(cell):
    return cell.split("_", 1)[1]


def load_rates():
    with open(STALENESS_CSV) as f:
        rows = [r for r in csv.DictReader(f) if r["scope"] == SCOPE]
    if not rows:
        raise SystemExit(f"FATAL: no scope=={SCOPE} rows in {STALENESS_CSV}.")

    cells = sorted({r["cell"] for r in rows}, key=lambda c: (workload_of(c), conc_of(c)))
    rates, denom = {}, {}
    for cell in cells:
        sub = [r for r in rows if r["cell"] == cell]
        d = sum(int(r["decisions_complete_sv"]) for r in sub if r["node"] == AD.NODES[0])
        denom[cell] = d
        for node in AD.NODES:
            c = sum(int(r["stale_count"]) for r in sub if r["node"] == node)
            rates[(cell, node)] = (c / d) if d else None

    thresholds = set()
    for r in rows:
        for v in str(r.get("max_state_age_ms_observed", "")).split("|"):
            if v.strip():
                thresholds.add(float(v))
    return cells, rates, denom, sorted(thresholds)


def load_ages():
    """Raw age_ms per node over stale decisions, clean runs, warm-up excluded."""
    ages = {n: [] for n in AD.NODES}
    polls, n_runs, n_wu = set(), 0, {}
    for r in AD.load_all_runs():
        if not r.get("usable") or r.get("verdict") != "clean":
            continue
        meta = json.loads(open(os.path.join(r["dir"], "run_meta.json")).read())
        fc = meta.get("frozen_config") or {}
        if fc.get("poll_interval_ms") is not None:
            polls.add(float(fc["poll_interval_ms"]))
        nw, _ = resolve_num_warmup(meta, r["dir"])
        n_wu[nw] = n_wu.get(nw, 0) + 1
        s = scan(meta, os.path.join(r["dir"], meta["router_log_file"]), nw, "measured")
        if s is None:
            continue
        n_runs += 1
        for n in AD.NODES:
            ages[n].extend(s["node_age"][n])
    return {n: np.array(sorted(v)) for n, v in ages.items()}, sorted(polls), n_runs, n_wu


def style(ax):
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.4, linestyle="-")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def node_handles(hw):
    return [Line2D([], [], color=NODE_COLOUR[i], linestyle=NODE_LINESTYLE[i],
                   marker=NODE_MARKER[i], markersize=2.6, linewidth=1.0,
                   label=hw[n]) for i, n in enumerate(AD.NODES)]


def place_legend_below(fig, handles, axes_list, gap=0.05, ncol=1):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    lowest = 1.0
    for ax in axes_list:
        for art in [ax.xaxis.label] + list(ax.get_xticklabels()):
            bb = art.get_window_extent(renderer=r).transformed(fig.transFigure.inverted())
            if bb.height > 0:
                lowest = min(lowest, bb.y0)
    for t in fig.texts:
        bb = t.get_window_extent(renderer=r).transformed(fig.transFigure.inverted())
        if bb.height > 0:
            lowest = min(lowest, bb.y0)
    fig.legend(handles=handles, loc="upper left",
               bbox_to_anchor=(0.02, lowest - gap), bbox_transform=fig.transFigure,
               ncol=ncol, handlelength=1.8, handletextpad=0.4, borderpad=0.0,
               labelspacing=0.25, columnspacing=1.0)


def save(fig, outdir, stem, dpi):
    out = []
    for ext in ("pdf", "png"):
        p = Path(outdir) / f"{stem}.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=0.015)
        out.append(p)
    plt.close(fig)
    return out


# ───────────────────────────── figure 1 ───────────────────────────────────
def fig_rate(cells, rates, denom, hw, outdir, dpi, width):
    line_wls = [w for w in ("uni", "bim")
                if len([c for c in cells if workload_of(c) == w]) > 1]
    other = [c for c in cells if workload_of(c) not in line_wls]
    names = {"uni": "uniform", "bim": "bimodal", "sgpt": "ShareGPT"}

    ymax = max(v for v in rates.values() if v is not None) * 1.15
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, len(line_wls), figsize=(width, 1.95),
                                 sharey=True)
        axes = np.atleast_1d(axes)
        for ax, wl in zip(axes, line_wls):
            style(ax)
            cs = sorted([c for c in cells if workload_of(c) == wl], key=conc_of)
            xs = [conc_of(c) for c in cs]
            for i, node in enumerate(AD.NODES):
                ys = [rates[(c, node)] for c in cs]
                ax.plot(xs, ys, color=NODE_COLOUR[i], linestyle=NODE_LINESTYLE[i],
                        marker=NODE_MARKER[i], markersize=2.8, linewidth=1.0,
                        markeredgecolor=SURFACE, markeredgewidth=0.3, zorder=3)
            ax.set_xticks(xs)
            ax.set_xlabel("Concurrency", labelpad=1.5)
            ax.set_title(names.get(wl, wl), fontsize=6.5, pad=2.5)
            ax.set_ylim(0, ymax)
        axes[0].set_ylabel("Stale rate", labelpad=2)

        # ShareGPT: one concurrency level only, so points + a note, never a line.
        if other:
            # Kept on short lines so the note does not widen the figure.
            head = ("")
            body = ""
            fig.text(0.02, 0.055, head, fontsize=5.1, color=TEXT_SECONDARY,
                     ha="left", va="bottom")
            fig.text(0.02, 0.005, body, fontsize=5.1, color=TEXT_SECONDARY,
                     ha="left", va="bottom")
        fig.subplots_adjust(bottom=0.44, left=0.155, right=0.985, top=0.90,
                            wspace=0.12)
        place_legend_below(fig, node_handles(hw), list(axes), gap=0.055)
        return save(fig, outdir, "sq2_stale_rate_by_concurrency", dpi)


# ───────────────────────── figure 2a: CDF ─────────────────────────────────
def fig_cdf(ages, hw, threshold, poll, outdir, dpi, width):
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(width, 1.95))
        style(ax)
        lo = min(poll, threshold) * 0.35
        hi = max(a.max() for a in ages.values() if len(a)) * 1.6
        for i, node in enumerate(AD.NODES):
            a = ages[node]
            if not len(a):
                continue
            y = np.arange(1, len(a) + 1) / len(a)
            ax.step(np.concatenate([[lo], a]), np.concatenate([[0.0], y]),
                    where="post", color=NODE_COLOUR[i],
                    linestyle=NODE_LINESTYLE[i], linewidth=1.0, zorder=3)
        # Labels pushed to opposite sides of their own rules: centred on a log
        # axis the two run into each other, the rules being less than 3x apart.
        for x, c, lab, ha, mult in (
                (poll, RULE_POLL, f"poll {poll:.0f} ms", "right", 0.93),
                (threshold, RULE_THRESHOLD, f"threshold {threshold:.0f} ms",
                 "left", 1.07)):
            ax.axvline(x, color=c, linewidth=0.8, zorder=2)
            ax.text(x * mult, 1.02, lab, fontsize=5.2, color=c,
                    ha=ha, va="bottom")
        ax.set_xscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("State age at decision (ms, log scale)", labelpad=1.5)
        ax.set_ylabel("Cumulative share of\nstale decisions", labelpad=2)
        fig.subplots_adjust(bottom=0.36, left=0.20, right=0.985, top=0.86)
        place_legend_below(fig, node_handles(hw), [ax], gap=0.055)
        return save(fig, outdir, "sq2_age_cdf", dpi)


# ─────────────────────── figure 2b: histogram ─────────────────────────────
def fig_hist(ages, hw, threshold, poll, outdir, dpi, width):
    with plt.rc_context(RC):
        fig, axs = plt.subplots(len(AD.NODES), 1, figsize=(width, 2.5),
                                sharex=True)
        lo = min(poll, threshold) * 0.35
        hi = max(a.max() for a in ages.values() if len(a)) * 1.6
        bins = np.logspace(np.log10(lo), np.log10(hi), 34)
        for i, (ax, node) in enumerate(zip(axs, AD.NODES)):
            style(ax)
            a = ages[node]
            ax.hist(a, bins=bins, color=NODE_COLOUR[i], edgecolor=SURFACE,
                    linewidth=0.25, zorder=3)
            for x, c in ((poll, RULE_POLL), (threshold, RULE_THRESHOLD)):
                ax.axvline(x, color=c, linewidth=0.8, zorder=4)
            ax.set_xscale("log")
            ax.set_xlim(lo, hi)
            ax.set_ylabel(hw[node].split(" (")[0].replace("pi-node-", "node "),
                          fontsize=5.6, labelpad=2)
            ax.tick_params(labelsize=5.4)
            if i == 0:
                for x, c, lab, ha, mult in (
                        (poll, RULE_POLL, f"poll {poll:.0f} ms", "right", 0.93),
                        (threshold, RULE_THRESHOLD,
                         f"threshold {threshold:.0f} ms", "left", 1.07)):
                    ax.text(x * mult, ax.get_ylim()[1] * 1.03, lab, fontsize=5.2,
                            color=c, ha=ha, va="bottom")
        axs[-1].set_xlabel("State age at decision (ms, log scale)", labelpad=1.5)
        fig.subplots_adjust(bottom=0.30, left=0.20, right=0.985, top=0.90,
                            hspace=0.30)
        place_legend_below(fig, node_handles(hw), [axs[-1]], gap=0.05)
        return save(fig, outdir, "sq2_age_hist", dpi)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=str(HERE))
    ap.add_argument("--width", type=float, default=3.33)
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    hw = hardware_labels()
    cells, rates, denom, thresholds = load_rates()
    ages, polls, n_runs, n_wu = load_ages()

    if len(thresholds) != 1:
        raise SystemExit(f"FATAL: max_state_age_ms is not unique: {thresholds}")
    if len(polls) != 1:
        raise SystemExit(f"FATAL: poll_interval_ms is not unique: {polls}")
    threshold, poll = thresholds[0], polls[0]

    print("=" * 92)
    print(f"Figure 1 source: {STALENESS_CSV}  (scope={SCOPE}, warm-up already excluded)")
    print(f"Figure 2 source: router logs via analyze_sq2.scan(warmup_mode='measured'), "
          f"clean runs only ({n_runs} runs, num_warmup={n_wu}) -- the raw age_ms sample "
          f"is not in the CSV")
    print(f"Threshold {threshold:.0f} ms read from max_state_age_ms_observed; "
          f"poll interval {poll:.0f} ms read from run_meta.frozen_config")
    print("Node hardware parsed from benchmarks/cost_model_results_<node>_<hw>:")
    for n in AD.NODES:
        print(f"    {n}  ->  {hw[n]}")
    print("=" * 92)

    print("\nFIGURE 1 — stale rate per (cell, node), arms pooled")
    print(f"{'cell':<9}{'workload':<10}{'conc':>5}{'decisions':>11}"
          + "".join(f"{n:>14}" for n in AD.NODES))
    for c in cells:
        print(f"{c:<9}{workload_of(c):<10}{conc_of(c):>5}{denom[c]:>11}"
              + "".join(f"{rates[(c, n)]:>14.4f}" for n in AD.NODES))

    print("\nFIGURE 2 — age_ms over stale decisions (clean, warm-up excluded)")
    print(f"{'node':<12}{'n':>8}{'min':>11}{'p25':>11}{'median':>11}"
          f"{'p90':>11}{'max':>12}{'<=thresh':>10}")
    for n in AD.NODES:
        a = ages[n]
        print(f"{n:<12}{len(a):>8}{a.min():>11.1f}{np.percentile(a,25):>11.1f}"
              f"{np.median(a):>11.1f}{np.percentile(a,90):>11.1f}{a.max():>12.1f}"
              f"{int((a <= threshold).sum()):>10}")
    gmin = min(a.min() for a in ages.values() if len(a))
    print(f"  smallest stale age anywhere: {gmin:.1f} ms = {gmin/threshold:.2f}x the "
          f"{threshold:.0f} ms threshold, {gmin/poll:.2f}x the {poll:.0f} ms poll interval")

    written = []
    written += fig_rate(cells, rates, denom, hw, outdir, args.dpi, args.width)
    written += fig_cdf(ages, hw, threshold, poll, outdir, args.dpi, args.width)
    written += fig_hist(ages, hw, threshold, poll, outdir, args.dpi, args.width)
    print()
    for p in written:
        print(f"Wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
