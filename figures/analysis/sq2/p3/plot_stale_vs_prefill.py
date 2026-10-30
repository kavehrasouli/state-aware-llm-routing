import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "3e_results" / "analysis").is_dir())
ANALYSIS = REPO / "3e_results" / "analysis"
for p in (ANALYSIS, ANALYSIS / "sq2"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import analyze_divergence as AD                     # noqa: E402

import matplotlib                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.lines import Line2D                  # noqa: E402

STALENESS_CSV = ANALYSIS / "sq2" / "sq2_staleness.csv"
PROFILE_CSV = ANALYSIS / "sq2" / "cell_load_profile.csv"
SCOPE = "clean"

WL = {"uniform": "uni", "bimodal": "bim", "sharegpt": "sgpt"}
SOURCE_ORDER = [AD.col_label(w,c) for w,c in AD.CELL_COLS]
GRID_ORDER = SOURCE_ORDER
WL_NAME = {"uni": "uniform", "bim": "bimodal", "sgpt": "ShareGPT"}


OFF_GRID_CELL = None

TEXT_PRIMARY, TEXT_SECONDARY = "#0b0b0b", "#52514e"
GRID, AXIS, SURFACE = "#e6e5e1", "#bdbcb7", "#ffffff"

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

# Workloads differ by marker shape and fill, so the figure also works in greyscale.
WL_STYLE = {
    "uni":  dict(marker="o", facecolor="#ffffff", edgecolor="#0b0b0b"),
    "bim":  dict(marker="s", facecolor="#0b0b0b", edgecolor="#0b0b0b"),
    "sgpt": dict(marker="^", facecolor="#9a9994", edgecolor="#0b0b0b"),
}


def conc_of(cell):
    return int(cell.split("_")[0].lstrip("c"))


def workload_of(cell):
    return cell.split("_", 1)[1]


def rankdata(vals):
    """Average ranks for ties, so that in-flight medians (which do tie) are
    handled the way scipy.stats.spearmanr would handle them."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / (sxx * syy) ** 0.5


def spearman(x, y):
    if len(x) < 3:
        return None
    return pearson(rankdata(x), rankdata(y))


# ── stale rate, from sq2_staleness.csv ────────────────────────────────────
def load_stale():
    """Run-weighted mean over arms of the per-arm run-mean stale rate, per node;
    then the mean over the three nodes. Also returns the pooled rate as a
    cross-check only -- it is not what gets correlated or plotted."""
    with open(STALENESS_CSV) as f:
        rows = [r for r in csv.DictReader(f) if r["scope"] == SCOPE]
    if not rows:
        raise SystemExit(f"FATAL: no scope=={SCOPE} rows in {STALENESS_CSV}")

    per_node, pooled, runs = {}, {}, {}
    for cell in sorted({r["cell"] for r in rows}):
        sub = [r for r in rows if r["cell"] == cell]
        runs[cell] = sum(int(r["stale_rate_n_runs"]) for r in sub
                         if r["node"] == AD.NODES[0])
        for node in AD.NODES:
            num = den = 0.0
            for r in (x for x in sub if x["node"] == node):
                k = int(r["stale_rate_n_runs"])
                v = r["stale_rate_runmean"]
                if k and v not in ("", None):
                    num += float(v) * k
                    den += k
            per_node[(cell, node)] = (num / den) if den else None
        d = sum(int(r["decisions_complete_sv"]) for r in sub
                if r["node"] == AD.NODES[0])
        c = sum(int(r["stale_count"]) for r in sub)
        pooled[cell] = (c / (d * len(AD.NODES))) if d else None
    return per_node, pooled, runs


# ── published per-cell values, from cell_load_profile.csv ─────────────────
def load_profile():
    """The profile CSV is sectioned and headerless per section; column meaning
    comes from cell_load_profile.py's writer, mirrored here."""
    sec = defaultdict(dict)
    with open(PROFILE_CSV) as f:
        rd = csv.reader(f)
        next(rd)
        for row in rd:
            sec[row[0]][row[1]] = row[2:]

    out = {}
    for cell in sec["throughput"]:
        thr = sec["throughput"][cell]        # runs, decisions, dur/run, dec/s, dec/run
        siz = sec["sizes"][cell]             # n, prompt p50, prompt p90, out p50, out p90
        lat = sec["latency"][cell]           # e2el p50/p90, ttft p50/p90, decode p50/p90
        eng = sec["engine_load"][cell]       # resident x3, in-flight x3
        idl = sec["idleness"][cell]          # decisions, any idle, share
        res = [float(v) for v in eng[0:3]]
        inf = [float(v) for v in eng[3:6]]
        out[cell] = {
            "runs_profile": int(thr[0]),
            "duration_med_s": float(thr[2]),
            "prompt_p50": float(siz[1]),
            "prompt_p90": float(siz[2]),
            "ttft_p50_ms": float(lat[2]),
            "e2el_p50_ms": float(lat[0]),
            "resident_per_node": res,
            "resident_mean": sum(res) / len(res),
            "inflight_per_node": inf,
            "inflight_mean": sum(inf) / len(inf),
            "idle_share": float(idl[2]),
        }
    return out


# ── per-run token rates, recomputed over the same clean-run set ───────────
def load_token_rates():
    """Total prompt (and output) tokens over a run's successful requests divided
    by that run's duration, then averaged over runs.

    Duration is the bench measurement window (bench JSON duration_s), not
    run_meta start_ts..end_ts. The token totals come from the measured requests
    only -- warm-up is already excluded client-side -- so the matching
    denominator is the window those requests were issued in. run_meta wall clock
    additionally covers router start-up and the warm-up requests, which would
    deflate every rate by a few per cent; it is reported alongside so the choice
    is visible rather than assumed."""
    acc = defaultdict(lambda: {"prompt_s": [], "output_s": [], "prompt_s_wc": [],
                               "prompt_tot": [], "output_tot": [], "dur": [],
                               "prefill_duty": [], "runs": 0, "slots": set()})
    for r in AD.load_all_runs():
        if not r.get("usable") or r.get("verdict") != "clean":
            continue
        d = r["dir"]
        meta = json.loads(open(os.path.join(d, "run_meta.json")).read())
        cell = "c%s_%s" % (meta.get("concurrency"), WL[meta.get("workload")])
        bf = meta.get("bench_result_file")
        if not bf or not os.path.exists(os.path.join(d, bf)):
            continue
        b = json.loads(open(os.path.join(d, bf)).read())
        dur = b.get("duration_s")
        if not (isinstance(dur, (int, float)) and dur > 0):
            continue

        ptok = otok = ttft_s = 0.0
        for rec in b.get("raw", []):
            if not rec.get("success"):
                continue
            v = rec.get("actual_prompt_len")
            if isinstance(v, (int, float)) and v >= 0:
                ptok += v
            v = rec.get("generated_output_tokens")
            if isinstance(v, (int, float)):
                otok += v
            v = rec.get("ttft_ms")
            if isinstance(v, (int, float)):
                ttft_s += v / 1000.0

        a = acc[cell]
        a["runs"] += 1
        a["prompt_tot"].append(ptok)
        a["output_tot"].append(otok)
        a["dur"].append(dur)
        a["prompt_s"].append(ptok / dur)
        a["output_s"].append(otok / dur)
        # Prefill duty cycle: share of slot-time between dispatch and first token. TTFT includes queueing,
        # so this is an upper bound on prefill occupancy.
        slots = len(AD.NODES) * 4
        a["slots"].add(slots)
        a["prefill_duty"].append(ttft_s / (dur * slots))

        st, en = meta.get("start_ts"), meta.get("end_ts")
        try:
            wc = (datetime.fromisoformat(str(en).replace("Z", "+00:00"))
                  - datetime.fromisoformat(str(st).replace("Z", "+00:00"))).total_seconds()
        except Exception:
            wc = None
        if wc and wc > 0:
            a["prompt_s_wc"].append(ptok / wc)
    return acc


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def f(v, nd=3):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "n/a"


# ── separation diagnostic ─────────────────────────────────────────────────
def separation(xs):
    """How well an x variable spreads the selected cells.

    min_gap    smallest distance between adjacent cells once x is scaled to
               [0, 1]. Small means two cells land on top of each other and the
               figure cannot show them as distinct conditions.
    within_wl  mean over workloads of that workload's x span, same scale. Near
               zero means the variable is a workload label in disguise: the
               concurrency sweep collapses and each workload becomes a single
               vertical stack of points."""
    lo, hi = min(xs.values()), max(xs.values())
    if hi <= lo:
        return None, None
    norm = {c: (v - lo) / (hi - lo) for c, v in xs.items()}
    vals = sorted(norm.values())
    min_gap = min(b - a for a, b in zip(vals, vals[1:]))
    spans = []
    for wl in ("uni", "bim", "sgpt"):
        sub = [norm[c] for c in norm if workload_of(c) == wl]
        if len(sub) > 1:
            spans.append(max(sub) - min(sub))
    return min_gap, (mean(spans) if spans else None)


# ── figure ────────────────────────────────────────────────────────────────
def style(ax):
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.4, linestyle="-")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


LABEL_FS = 5.1
# Candidate label placements, in points from the marker: (dx, dy, ha).
LABEL_SLOTS = [(3.6, 2.4, "left"), (3.6, -7.2, "left"),
               (-3.6, 2.4, "right"), (-3.6, -7.2, "right")]


def _rect(fx, fy, dx, dy, ha, wl, hl, sx, sy):
    """Label rectangle in axes fraction, given the marker at (fx, fy), an offset
    in points and the label's size in points. sx, sy convert points to fraction."""
    x = fx + dx * sx
    if ha == "right":
        x -= wl * sx
    return (x, fy + dy * sy, x + wl * sx, fy + dy * sy + hl * sy)


def place_labels(ax, cells, xs, ys, texts):
    """Pick, per point, the label corner that neither leaves the axes nor lands
    on another marker or an already-placed label.

    Done by scoring four candidate corners rather than by a hand-tuned offset
    table: with several cells the collisions move whenever a value moves, and a
    table would quietly go wrong the next time the grid is re-run."""
    ax.figure.canvas.draw()
    bb = ax.get_window_extent()
    w_pt = bb.width * 72.0 / ax.figure.dpi
    h_pt = bb.height * 72.0 / ax.figure.dpi
    sx, sy = 1.0 / w_pt, 1.0 / h_pt          # points -> axes fraction

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    pts = [c for c in cells if xs.get(c) is not None and ys.get(c) is not None]
    fx = {c: (xs[c] - x0) / (x1 - x0) for c in pts}
    fy = {c: (ys[c] - y0) / (y1 - y0) for c in pts}

    m = 3.4 * sx, 3.4 * sy                    # marker half-extent, points
    markers = [(fx[c] - m[0], fy[c] - m[1], fx[c] + m[0], fy[c] + m[1])
               for c in pts]

    def overlap(a, b):
        return (max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
                * max(0.0, min(a[3], b[3]) - max(a[1], b[1])))

    placed, out = [], {}
    for c in sorted(pts, key=lambda k: -fy[k]):
        wl = LABEL_FS * 0.60 * len(texts[c])   # ~0.6 em per glyph at this size
        hl = LABEL_FS * 0.95
        best, best_cost = LABEL_SLOTS[0], None
        for dx, dy, ha in LABEL_SLOTS:
            r = _rect(fx[c], fy[c], dx, dy, ha, wl, hl, sx, sy)
            clip = (max(0.0, -r[0]) + max(0.0, r[2] - 1.0)
                    + max(0.0, -r[1]) + max(0.0, r[3] - 1.0))
            cost = 40.0 * clip
            cost += 12.0 * sum(overlap(r, mk) for mk in markers)
            cost += 12.0 * sum(overlap(r, pr) for pr in placed)
            if dx < 0:                        # mild preference for right-hand
                cost += 0.004                 # labels, all else being equal
            if dy < 0:
                cost += 0.002
            if best_cost is None or cost < best_cost:
                best, best_cost = (dx, dy, ha), cost
        out[c] = best
        placed.append(_rect(fx[c], fy[c], best[0], best[1], best[2], wl, hl, sx, sy))
    return out


def scatter_panel(ax, cells, xs, ys, xlabel, single):
    """Markers only. Labels are added later, once the axis limits are final --
    placement is computed in axes fraction, so it is wrong if the limits move
    afterwards."""
    for cell in cells:
        if xs.get(cell) is None or ys.get(cell) is None:
            continue
        st = WL_STYLE[workload_of(cell)]
        ax.scatter([xs[cell]], [ys[cell]], s=17,
                   marker=st["marker"], facecolor=st["facecolor"],
                   edgecolor=st["edgecolor"], linewidth=0.7,
                   alpha=0.55 if cell == single else 1.0, zorder=3)
    style(ax)
    ax.set_xlabel(xlabel, labelpad=1.5)


def annotate_panel(ax, cells, xs, ys, single):
    texts = {c: f"c{conc_of(c)}" + ("*" if c == single else "") for c in cells
             if xs.get(c) is not None and ys.get(c) is not None}
    for cell, (dx, dy, ha) in place_labels(ax, cells, xs, ys, texts).items():
        ax.annotate(texts[cell], (xs[cell], ys[cell]),
                    textcoords="offset points", xytext=(dx, dy),
                    fontsize=LABEL_FS, color=TEXT_SECONDARY, ha=ha, zorder=4)


def make_figure(cells, stale, xa, xb, labels, stem, outdir, width, dpi, rhos):
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(width, 1.95), sharey=True)
        scatter_panel(axes[0], cells, xa, stale, labels[0], None)
        scatter_panel(axes[1], cells, xb, stale, labels[1], None)
        axes[0].set_ylabel("Stale rate", labelpad=2)

        ymax = max(v for v in stale.values() if v is not None)
        for ax in axes:
            ax.set_ylim(-0.045, ymax * 1.16)
        for ax, rho in zip(axes, rhos):
            ax.set_title(r"$\rho_s$ = " + f"{rho:.2f}" if rho is not None
                         else "", fontsize=6.2, pad=2.5)

        # Limits are final only now, and label placement is computed against
        # them, so the annotations have to come after every set_ylim/set_xlim.
        annotate_panel(axes[0], cells, xa, stale, None)
        annotate_panel(axes[1], cells, xb, stale, None)

        handles = [Line2D([], [], linestyle="none", marker=WL_STYLE[w]["marker"],
                          markerfacecolor=WL_STYLE[w]["facecolor"],
                          markeredgecolor=WL_STYLE[w]["edgecolor"],
                          markeredgewidth=0.7, markersize=3.4, label=WL_NAME[w])
                   for w in ("uni", "bim", "sgpt")]
        fig.subplots_adjust(bottom=0.33, left=0.135, right=0.985, top=0.885,
                            wspace=0.10)
        fig.legend(handles=handles, loc="upper left",
                   bbox_to_anchor=(0.02, 0.175), bbox_transform=fig.transFigure,
                   ncol=3, handlelength=1.2, handletextpad=0.4, borderpad=0.0,
                   labelspacing=0.25, columnspacing=1.1)

        written = []
        for ext in ("pdf", "png"):
            p = Path(outdir) / f"{stem}.{ext}"
            fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=0.015)
            written.append(p)
        plt.close(fig)
        return written


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=str(HERE))
    ap.add_argument("--width", type=float, default=3.33)   # ACM single column
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    per_node, pooled, stale_runs = load_stale()
    prof = load_profile()
    tok = load_token_rates()

    source_cells = [c for c in SOURCE_ORDER if c in prof and c in tok]
    grid_cells = [c for c in GRID_ORDER if c in prof and c in tok]
    missing_grid = [c for c in GRID_ORDER if c not in grid_cells]
    if missing_grid:
        raise SystemExit(f"FATAL: missing formal grid cells: {missing_grid}")

    # ---- consistency gate: the recomputed run count must match the profile --
    print("=" * 100)
    print("Stale rate vs prefill work vs occupancy — per-cell table, clean runs")
    print("=" * 100)
    print("\n0. SOURCE CROSS-CHECK (recomputed run counts against the two CSVs)")
    print(f"{'cell':<9}{'profile n':>11}{'staleness n':>13}{'recomputed n':>14}"
          f"{'dur med s (profile)':>21}{'dur med s (bench)':>19}")
    bad = []
    for c in source_cells:
        rec = tok[c]["runs"]
        dur_bench = sorted(tok[c]["dur"])[len(tok[c]["dur"]) // 2]
        print(f"{c:<9}{prof[c]['runs_profile']:>11}{stale_runs.get(c, 0):>13}"
              f"{rec:>14}{prof[c]['duration_med_s']:>21.1f}{dur_bench:>19.1f}")
        if not (prof[c]["runs_profile"] == stale_runs.get(c, 0) == rec):
            bad.append(c)
    if bad:
        raise SystemExit(
            f"FATAL: run counts disagree between sources for {bad}. The table "
            "would mix different run sets; refusing to plot.")
    print("  all three sources agree on the clean-run set for every cell.")
    print("  duration differs by construction: profile uses run_meta wall clock "
          "(includes\n  router start-up and warm-up), the token rates use the "
          "bench measurement window.")

    # ---- build the per-cell table ----
    rows = {}
    for c in source_cells:
        p = prof[c]
        node_rates = [per_node[(c, n)] for n in AD.NODES]
        rows[c] = {
            "cell": c,
            "workload": WL_NAME[workload_of(c)],
            "conc": conc_of(c),
            "runs": p["runs_profile"],
            "single_run": c == OFF_GRID_CELL,
            "stale_mean": mean([v for v in node_rates if v is not None]),
            "stale_n1": node_rates[0],
            "stale_n2": node_rates[1],
            "stale_n3": node_rates[2],
            "stale_pooled_xcheck": pooled.get(c),
            "inflight_mean": p["inflight_mean"],
            "idle_share": p["idle_share"],
            "prompt_p50": p["prompt_p50"],
            "prompt_tok_s": mean(tok[c]["prompt_s"]),
            "output_tok_s": mean(tok[c]["output_s"]),
            "resident_mean": p["resident_mean"],
            # extra prefill candidates
            "prompt_tok_per_req": mean(tok[c]["prompt_tot"]) / 100.0,
            "prompt_over_output": (mean(tok[c]["prompt_s"]) / mean(tok[c]["output_s"])
                                   if mean(tok[c]["output_s"]) else None),
            "prefill_duty": mean(tok[c]["prefill_duty"]),
            "prompt_tok_s_wallclock": mean(tok[c]["prompt_s_wc"]),
        }

    print("\n1. PER-CELL DIAGNOSTIC TABLE (stale rate is the run-level mean, "
          "not pooled)")
    hdr = (f"{'cell':<9}{'runs':>5}{'stale':>8}{'n1':>8}{'n2':>8}{'n3':>8}"
           f"{'inflight':>10}{'idle sh':>9}{'prompt p50':>12}{'ptok/s':>9}"
           f"{'otok/s':>9}{'resident':>10}")
    print(hdr)
    for c in source_cells:
        r = rows[c]
        print(f"{c + ('*' if r['single_run'] else ''):<9}{r['runs']:>5}"
              f"{f(r['stale_mean']):>8}{f(r['stale_n1']):>8}{f(r['stale_n2']):>8}"
              f"{f(r['stale_n3']):>8}{f(r['inflight_mean'], 2):>10}"
              f"{f(r['idle_share'], 3):>9}{f(r['prompt_p50'], 1):>12}"
              f"{f(r['prompt_tok_s'], 2):>9}{f(r['output_tok_s'], 2):>9}"
              f"{f(r['resident_mean'], 1):>10}")
    print("All nine cells use the common inclusion ledger; the pilot is excluded.")

    print("\n1b. CROSS-CHECK: pooled stale rate (decision-weighted) beside the "
          "run-level mean")
    print(f"{'cell':<9}{'run-level':>11}{'pooled':>9}{'diff':>9}")
    for c in source_cells:
        r = rows[c]
        pv = r["stale_pooled_xcheck"]
        dv = (r["stale_mean"] - pv) if (pv is not None) else None
        print(f"{c:<9}{f(r['stale_mean']):>11}{f(pv):>9}{f(dv):>9}")
        rows[c].pop("stale_pooled_xcheck", None)
        rows[c]["stale_pooled_xcheck"] = pv

    # ---- correlations ----
    cand = [
        ("prompt tokens/s (prefill)", "prompt_tok_s"),
        ("median in-flight count", "inflight_mean"),
        ("idle-replica share", "idle_share"),
        ("median prompt length", "prompt_p50"),
        ("output tokens/s (contrast)", "output_tok_s"),
        ("median resident tokens", "resident_mean"),
    ]
    extra = [
        ("mean prompt tokens/request", "prompt_tok_per_req"),
        ("prompt:output token ratio", "prompt_over_output"),
        ("prefill duty cycle (TTFT-based)", "prefill_duty"),
        ("prompt tokens/s, wall clock", "prompt_tok_s_wallclock"),
    ]

    def rho_for(key, use_cells):
        xs = [rows[c][key] for c in use_cells]
        ys = [rows[c]["stale_mean"] for c in use_cells]
        if any(v is None for v in xs + ys):
            return None
        return spearman(xs, ys)

    with open(os.path.join(args.outdir, "staleness_load_associations.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(next(iter(rows.values())).keys()))
        writer.writeheader(); writer.writerows(rows.values())
    with open(os.path.join(args.outdir, "staleness_spearman.csv"), "w", newline="") as fh:
        writer = csv.writer(fh); writer.writerow(["predictor", "n_cells", "spearman_rho"])
        for label,key in cand + extra:
            writer.writerow([key, len(grid_cells), rho_for(key, grid_cells)])

    all_cells = source_cells

    print("\n2. SPEARMAN RANK CORRELATION WITH THE MEAN STALE RATE")
    print("   DESCRIPTIVE ONLY. The headline uses the nine combined grid cells. "
          "No p-values or significance claims.\n   The all-source column is a "
          "cross-check using the same inclusion ledger.")
    print(f"\n{'predictor':<34}{'grid rho (n=9)':>16}{'all-source (n=9)':>17}"
          f"{'min gap':>10}{'within-wl':>11}")
    for label, key in cand + extra:
        r7, r8 = rho_for(key, grid_cells), rho_for(key, all_cells)
        xs = {c: rows[c][key] for c in grid_cells if rows[c][key] is not None}
        gap, wl = separation(xs) if len(xs) == len(grid_cells) else (None, None)
        mark = "   <- headline" if key in ("prompt_tok_s", "inflight_mean") else ""
        print(f"{label:<34}{f(r7, 3):>16}{f(r8, 3):>17}{f(gap, 3):>10}"
              f"{f(wl, 3):>11}{mark}")
    print("\n  min gap    smallest spacing between adjacent cells with x scaled "
          "to [0,1];\n             a small value means two cells overlap in the "
          "figure.")
    print("  within-wl  mean x span within a workload on the same scale; near "
          "zero means the\n             variable is a workload label and the "
          "concurrency sweep collapses.")

    # ---- pick the prefill x axis for the alternative figure ----
    prefill_keys = [("prompt_tok_s", "Prompt tokens / s"),
                    ("prompt_tok_per_req", "Mean prompt tokens / request"),
                    ("prompt_over_output", "Prompt : output token ratio"),
                    ("prefill_duty", "Prefill duty cycle (TTFT-based)")]
    print("\n3. PREFILL-MEASURE COMPARISON (which one separates the cells)")
    print(f"{'measure':<34}{'grid rho (n=9)':>16}{'all-source (n=9)':>17}{'min gap':>10}"
          f"{'within-wl':>11}")
    scored = []
    for key, label in prefill_keys:
        xs = {c: rows[c][key] for c in grid_cells if rows[c][key] is not None}
        if len(xs) != len(grid_cells):
            continue
        gap, wl = separation(xs)
        r7, r8 = rho_for(key, grid_cells), rho_for(key, all_cells)
        scored.append((key, label, r7, r8, gap, wl))
        print(f"{label:<34}{f(r7, 3):>16}{f(r8, 3):>17}{f(gap, 3):>10}{f(wl, 3):>11}")

    base = next(s for s in scored if s[0] == "prompt_tok_s")
    others = [s for s in scored if s[0] != "prompt_tok_s"]
    best_rho = max(scored, key=lambda s: abs(s[2] or 0))
    # Pick the measure that best separates concurrency levels within each workload, with a minimum gap.
    sep_ok = [s for s in others if (s[4] or 0) >= 0.02]
    best_sep = max(sep_ok, key=lambda s: (s[5] or 0)) if sep_ok else None
    print(f"\n  highest |rho|:              {best_rho[1]} ({f(best_rho[2], 3)})")
    if best_sep is not None:
        print(f"  best within-workload spread: {best_sep[1]} "
              f"(within-wl {f(best_sep[5], 3)} vs {f(base[5], 3)} for "
              f"prompt tokens/s)")

    alt = None
    if best_sep is not None and (best_sep[5] or 0) > (base[5] or 0):
        alt = best_sep
        print(f"  -> building the alternative figure with {alt[1]} on the left "
              "panel, since it\n     spreads the cells within a workload better "
              "than prompt tokens/s does.")
    else:
        print("  -> no alternative prefill measure separates the cells better; "
              "only one figure built.")

    # ---- figures ----
    stale = {c: rows[c]["stale_mean"] for c in grid_cells}
    xa = {c: rows[c]["prompt_tok_s"] for c in grid_cells}
    xb = {c: rows[c]["inflight_mean"] for c in grid_cells}
    written = make_figure(
        grid_cells, stale, xa, xb,
        ("Prompt tokens / s", "Median in-flight count"),
        "sq2_stale_vs_prefill", outdir, args.width, args.dpi,
        (rho_for("prompt_tok_s", grid_cells),
         rho_for("inflight_mean", grid_cells)))

    if alt is not None:
        xalt = {c: rows[c][alt[0]] for c in grid_cells}
        written += make_figure(
            grid_cells, stale, xalt, xb, (alt[1], "Median in-flight count"),
            "sq2_stale_vs_prefill_alt", outdir, args.width, args.dpi,
            (alt[2], rho_for("inflight_mean", grid_cells)))

    # ---- table out ----
    fields = ["cell", "workload", "conc", "runs", "single_run", "stale_mean",
              "stale_n1", "stale_n2", "stale_n3", "stale_pooled_xcheck",
              "inflight_mean", "idle_share", "prompt_p50", "prompt_tok_s",
              "output_tok_s", "resident_mean", "prompt_tok_per_req",
              "prompt_over_output", "prefill_duty", "prompt_tok_s_wallclock"]
    csv_path = Path(outdir) / "stale_vs_prefill_cells.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for c in source_cells:
            w.writerow({k: rows[c].get(k) for k in fields})

    print("\n4. WRITTEN")
    for p in [csv_path] + written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
