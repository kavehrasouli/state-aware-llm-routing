import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "3e_results" / "analysis").is_dir())
ANALYSIS = REPO / "3e_results" / "analysis"
for p in (ANALYSIS, ANALYSIS / "sq2"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import analyze_divergence as AD                      # noqa: E402
from analyze_sq2 import scan                          # noqa: E402
from analyze_sq1 import resolve_num_warmup            # noqa: E402

import matplotlib                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.patches import Patch                  # noqa: E402

SUMMARY_CSV = ANALYSIS / "sq2" / "sq2_overhead_by_arm.csv"
SCOPE = "all"   # methodology Section 6: overhead over all runs

# Ordinal blue ramp; lightness carries the order, so it also works in greyscale.
GROUP_ORDER = ["stateless", "state-reading", "body-parsing"]
GROUP_COLOUR = {"stateless": "#5598e7", "state-reading": "#256abf",
                "body-parsing": "#0d366b"}

TEXT_PRIMARY, TEXT_SECONDARY = "#0b0b0b", "#52514e"
GRID, AXIS, SURFACE = "#e6e5e1", "#bdbcb7", "#ffffff"

RC = {
    "font.size": 6.6, "axes.titlesize": 6.6, "axes.labelsize": 6.6,
    "xtick.labelsize": 5.9, "ytick.labelsize": 5.9, "legend.fontsize": 5.7,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.0, "ytick.major.size": 0.0, "grid.linewidth": 0.4,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.edgecolor": AXIS,
    "text.color": TEXT_PRIMARY, "axes.labelcolor": TEXT_PRIMARY,
    "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "legend.frameon": False, "axes.grid": False,
}


def first_reads_state(log_path):
    """reads_state is a property of the policy, so one request record settles
    it for the whole run."""
    with open(log_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("type") == "request":
                return r.get("reads_state")
    return None


def load():
    with open(SUMMARY_CSV) as f:
        summary = {r["arm"]: r for r in csv.DictReader(f) if r["scope"] == SCOPE}
    if not summary:
        raise SystemExit(f"FATAL: no scope=={SCOPE} rows in {SUMMARY_CSV}.")

    per_request, reads_state = {}, {}
    n_wu = {}
    for r in AD.load_all_runs():
        if not r.get("usable"):
            continue
        meta = json.loads(open(os.path.join(r["dir"], "run_meta.json")).read())
        nw, _ = resolve_num_warmup(meta, r["dir"])
        n_wu[nw] = n_wu.get(nw, 0) + 1
        log = os.path.join(r["dir"], meta["router_log_file"])
        s = scan(meta, log, nw, "measured")
        if s is None:
            continue
        per_request.setdefault(r["arm"], []).extend(s["select_us"])
        if r["arm"] not in reads_state:
            reads_state[r["arm"]] = first_reads_state(log)

    arms = []
    for arm, row in summary.items():
        body = str(row.get("decomposition_available", "")).strip().lower() == "true"
        group = ("body-parsing" if body
                 else "stateless" if reads_state.get(arm) is False
                 else "state-reading")
        arms.append({
            "arm": arm, "group": group,
            "median": float(row["select_us_median_runmean"]),
            "median_sd": float(row["select_us_median_runsd"]),
            "p90": float(row["select_us_p90_runmean"]),
            "p99": float(row["select_us_p99_runmean"]),
            "p99_pooled": float(row["select_us_p99_pooled"]),
            "n_runs": int(row["n_runs"]), "n_requests": int(row["n_requests"]),
            "sample": np.array(per_request.get(arm, [])),
        })
    arms.sort(key=lambda a: a["median"])          # ascending, as specified
    return arms, n_wu


def style(ax, grid_axis="x"):
    ax.set_axisbelow(True)
    getattr(ax, f"{grid_axis}axis").grid(True, color=GRID, linewidth=0.4, linestyle="-")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def legend_handles(arms):
    present = [g for g in GROUP_ORDER if any(a["group"] == g for a in arms)]
    h = [Patch(facecolor=GROUP_COLOUR[g], edgecolor=SURFACE, linewidth=0.35,
               label=g) for g in present]
    h.append(Patch(facecolor="none", edgecolor="none",
                   label=""))
    return h


def annotate_n(ax, ys, values, labels, pad_frac=0.03):
    """n= to the RIGHT of each bar's error bar, with the x-limit extended to
    make room. Placing it under the bar (the obvious choice) collides with the
    next row's tick label at nine rows in a single column."""
    hi = max(values) if values else 1.0
    ax.set_xlim(ax.get_xlim()[0], hi * 1.26)
    xr = ax.get_xlim()
    for yi, v, lab in zip(ys, values, labels):
        ax.text(v + (xr[1] - xr[0]) * pad_frac, yi, lab, fontsize=4.9,
                color=TEXT_SECONDARY, va="center", ha="left")


def place_legend_below(fig, handles, anchor_axes, gap=0.045):
    """Draw, measure the lowest artist actually rendered, then hang the legend
    under it. Matplotlib's own outside-legend placement does not measure
    clip_on=False text or supxlabel, so padding alone moves the legend INTO
    those labels rather than away from them."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    lowest = 1.0
    for ax in anchor_axes:
        for art in [ax.xaxis.label] + list(ax.get_xticklabels()):
            bb = art.get_window_extent(renderer=r).transformed(
                fig.transFigure.inverted())
            if bb.height > 0:
                lowest = min(lowest, bb.y0)
    for t in fig.texts:
        bb = t.get_window_extent(renderer=r).transformed(fig.transFigure.inverted())
        if bb.height > 0:
            lowest = min(lowest, bb.y0)
    fig.legend(handles=handles, loc="upper left",
               bbox_to_anchor=(0.02, lowest - gap), bbox_transform=fig.transFigure,
               ncol=1, handlelength=1.3, handletextpad=0.4, borderpad=0.0,
               labelspacing=0.25)
    return lowest


def save(fig, outdir, stem, dpi):
    out = []
    for ext in ("pdf", "png"):
        p = Path(outdir) / f"{stem}.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=0.015)
        out.append(p)
    plt.close(fig)
    return out


# A: broken axis
def fig_a(arms, outdir, dpi, width):
    lo = [a for a in arms if a["group"] != "body-parsing"]
    hi = [a for a in arms if a["group"] == "body-parsing"]
    lo_hi = max(a["median"] + a["median_sd"] for a in lo) * 1.12
    hi_lo = min(a["median"] - a["median_sd"] for a in hi) * 0.90
    hi_hi = max(a["median"] + a["median_sd"] for a in hi) * 1.06
    print(f"  [A] left panel x = [0, {lo_hi:.2f}]   right panel x = "
          f"[{hi_lo:.2f}, {hi_hi:.2f}]  (computed from the data)")

    with plt.rc_context(RC):
        fig, (axl, axr) = plt.subplots(
            1, 2, figsize=(width, 2.55), sharey=True,
            gridspec_kw={"width_ratios": [3.1, 1.0], "wspace": 0.06})
        y = np.arange(len(arms))
        for ax in (axl, axr):
            style(ax)
            for a, yi in zip(arms, y):
                ax.barh(yi, a["median"], 0.66, xerr=a["median_sd"],
                        color=GROUP_COLOUR[a["group"]], edgecolor=SURFACE,
                        linewidth=0.35, zorder=3,
                        error_kw=dict(ecolor=TEXT_PRIMARY, elinewidth=0.6,
                                      capsize=1.3, capthick=0.6, zorder=4))
        axl.set_xlim(0, lo_hi)
        axr.set_xlim(hi_lo, hi_hi)
        axl.set_yticks(y)
        axl.set_yticklabels([a["arm"] for a in arms])
        axl.invert_yaxis()

        axl.spines["right"].set_visible(False)
        axr.spines["left"].set_visible(False)
        axr.tick_params(left=False)
        d = 0.9
        kw = dict(marker=[(-1, -d), (1, d)], markersize=4, linestyle="none",
                  color=AXIS, mec=AXIS, mew=0.7, clip_on=False)
        axl.plot([1, 1], [0, 1], transform=axl.transAxes, **kw)
        axr.plot([0, 0], [0, 1], transform=axr.transAxes, **kw)

        for ax, sel in ((axl, lambda a: a["group"] != "body-parsing"),
                        (axr, lambda a: a["group"] == "body-parsing")):
            idx = [i for i, a in enumerate(arms) if sel(a)]
            annotate_n(ax, [y[i] for i in idx],
                       [arms[i]["median"] + arms[i]["median_sd"] for i in idx],
                       [f"n={arms[i]['n_runs']}" for i in idx])

        fig.supxlabel("Selection time (µs), mean across runs of each run's median",
                      fontsize=RC["axes.labelsize"], y=0.10)
        fig.subplots_adjust(bottom=0.34, left=0.34, right=0.99, top=0.985)
        place_legend_below(fig, legend_handles(arms), (axl, axr))
        return save(fig, outdir, "sq2_overhead_A_broken_axis", dpi)


# B: grouped by timer coverage
def fig_b(arms, outdir, dpi, width):
    lo = [a for a in arms if a["group"] != "body-parsing"]
    hi = [a for a in arms if a["group"] == "body-parsing"]
    print(f"  [B] top panel: {len(lo)} select-only arms; "
          f"bottom panel: {len(hi)} body-parsing arm(s)")

    with plt.rc_context(RC):
        fig, (axt, axb) = plt.subplots(
            2, 1, figsize=(width, 2.9),
            gridspec_kw={"height_ratios": [len(lo), max(len(hi), 1) + 0.35],
                         "hspace": 0.42})
        for ax, group, title in (
                (axt, lo, "timed region = selection only"),
                (axb, hi, "timed region = selection + body read + token estimate")):
            style(ax)
            y = np.arange(len(group))
            for a, yi in zip(group, y):
                ax.barh(yi, a["median"], 0.62, xerr=a["median_sd"],
                        color=GROUP_COLOUR[a["group"]], edgecolor=SURFACE,
                        linewidth=0.35, zorder=3,
                        error_kw=dict(ecolor=TEXT_PRIMARY, elinewidth=0.6,
                                      capsize=1.3, capthick=0.6, zorder=4))
            ax.set_yticks(y)
            ax.set_yticklabels([a["arm"] for a in group])
            ax.invert_yaxis()
            ax.set_title(title, fontsize=6.1, pad=2.5, loc="left")
            annotate_n(ax, y, [a["median"] + a["median_sd"] for a in group],
                       [f"n={a['n_runs']}" for a in group])
        axb.set_xlabel("Selection time (µs), mean across runs of each run's median",
                       labelpad=2)
        fig.subplots_adjust(bottom=0.30, left=0.34, right=0.99, top=0.955)
        place_legend_below(fig, legend_handles(arms), (axb,))
        return save(fig, outdir, "sq2_overhead_B_grouped", dpi)


# C: box, log axis
def fig_c(arms, outdir, dpi, width):
    print("  [C] box: median line, box = IQR, whiskers = 1st/99th percentile, "
          "log x-axis")
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(width, 2.55))
        style(ax)
        data = [a["sample"] for a in arms]
        bp = ax.boxplot(data, vert=False, whis=(1, 99), showfliers=False,
                        widths=0.62, patch_artist=True,
                        medianprops=dict(color=SURFACE, linewidth=0.9),
                        boxprops=dict(linewidth=0.4, edgecolor=SURFACE),
                        whiskerprops=dict(linewidth=0.6, color=TEXT_PRIMARY),
                        capprops=dict(linewidth=0.6, color=TEXT_PRIMARY))
        for patch, a in zip(bp["boxes"], arms):
            patch.set_facecolor(GROUP_COLOUR[a["group"]])
        ax.set_xscale("log")
        ax.set_yticks(range(1, len(arms) + 1))
        ax.set_yticklabels([a["arm"] for a in arms])
        ax.invert_yaxis()
        ax.set_xlabel("Per-request selection time (µs, log scale)", labelpad=2)
        # Log axis: right-align n= at the axes edge rather than beside a whisker,
        # where a multiplicative offset would land differently on every row.
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo, hi * 2.2)
        for i, a in enumerate(arms, start=1):
            ax.text(ax.get_xlim()[1], i, f"n={a['n_requests']}", fontsize=4.9,
                    color=TEXT_SECONDARY, va="center", ha="right")
        fig.subplots_adjust(bottom=0.30, left=0.34, right=0.99, top=0.985)
        place_legend_below(fig, legend_handles(arms), (ax,))
        return save(fig, outdir, "sq2_overhead_C_box_log", dpi)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=str(HERE))
    ap.add_argument("--width", type=float, default=3.33)
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    arms, n_wu = load()

    print("=" * 96)
    print(f"Summary source: {SUMMARY_CSV}  (scope={SCOPE})")
    print(f"Per-request sample: router logs, warm-up EXCLUDED via "
          f"analyze_sq2.scan(warmup_mode='measured'); num_warmup per run = {n_wu}")
    print("=" * 96)
    print("Plotted statistic for A and B: mean across runs of each run's own median;")
    print("error bar = one SAMPLE standard deviation (ddof=1) across runs.")
    print("C plots the raw per-request sample. Arms ordered by median ascending.\n")
    print(f"{'arm':<28}{'group':<15}{'n_runs':>7}{'n_req':>7}"
          f"{'median':>9}{'±sd':>7}{'p90':>8}{'p99':>9}{'p99 pool':>10}"
          f"{'sample med':>11}{'sample p99':>11}")
    for a in arms:
        s = a["sample"]
        print(f"{a['arm']:<28}{a['group']:<15}{a['n_runs']:>7}{a['n_requests']:>7}"
              f"{a['median']:>9.2f}{a['median_sd']:>7.2f}{a['p90']:>8.2f}"
              f"{a['p99']:>9.2f}{a['p99_pooled']:>10.2f}"
              f"{np.median(s):>11.2f}{np.percentile(s, 99):>11.2f}")
    print("\n  'sample med' / 'sample p99' are over the raw per-request sample used by C;")
    print("  they are pooled over requests and so differ from the run-level columns.")
    print()

    written = []
    written += fig_a(arms, outdir, args.dpi, args.width)
    written += fig_b(arms, outdir, args.dpi, args.width)
    written += fig_c(arms, outdir, args.dpi, args.width)
    print()
    for p in written:
        print(f"Wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
