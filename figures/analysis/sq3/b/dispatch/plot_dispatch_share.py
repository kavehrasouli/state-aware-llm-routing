"""Figure of the Pi 5 dispatch share per policy and cell, in two layouts.

Source: sq3/b/sq3b_dispatch_share.csv; the capacity-proportional reference line is computed.
"""
import argparse
import csv
import json
import glob
import math
import sys
import tomllib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.lines import Line2D                   # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "3e_results" / "analysis").is_dir())
ANALYSIS = REPO / "3e_results" / "analysis"
CSV_PATH = ANALYSIS / "sq3" / "b" / "sq3b_dispatch_share.csv"
ROUTER_CFG = REPO / "routers" / "router" / "router_3nodes.toml"
ROUTER_CFG_CONC = REPO / "routers" / "router" / "router_3nodes_conc.toml"
RUN_METAS = REPO / "3e_results" / "*" / "*" / "*" / "run_meta.json"

NODE = "pi-node-1"

# ── declared presentation order (not data) ────────────────────────────────
WORKLOAD_ORDER = [("uniform", "uniform"), ("bimodal", "bimodal"),
                  ("sharegpt", "ShareGPT")]
LADDER_ORDER = ["random", "round_robin", "weighted_round_robin",
                "least_connections", "weighted_least_connections",
                "state_aware"]
EXTENSION_ORDER = ["wlc_conc", "state_aware_v2", "cost_predict_v3"]
STATELESS = {"random", "round_robin", "weighted_round_robin"}
CONC_ARM = "wlc_conc"          # the arm the secondary reference line belongs to

ARM_LABEL = {
    "random": "random", "round_robin": "round robin",
    "weighted_round_robin": "weighted round robin",
    "least_connections": "least connections",
    "weighted_least_connections": "weighted least connections",
    "state_aware": "state aware",
    "wlc_conc": "wlc_conc (re-weighted control)",
    "state_aware_v2": "state aware v2",
    "cost_predict_v3": "cost predict v3",
}

# Two palettes: "current" (one blue ramp, as in figures/sq3/a) and "grouped"
# (grey = stateless, blue = state-reading, amber = extensions; also distinct in greyscale).
LADDER_MARKERS = ["o", "s", "^", "D", "v", "P"]
EXTENSION_MARKERS = ["o", "s", "^"]
EXTENSION_DASHES = [(3.2, 1.3), (1.6, 1.1), (4.5, 1.2, 1.0, 1.2)]

CURRENT_LADDER_RAMP = ["#88b4e6", "#5f95d6", "#3d74bd",
                       "#245a9c", "#14407a", "#0a2854"]
STATELESS_RAMP = ["#b5b5b0", "#949490", "#6e6e69"]
STATE_READING_RAMP = ["#2a66b0", "#14407a", "#08234d"]
EXTENSION_RAMP = ["#dba33f", "#a86a10", "#63400a"]


def build_styles(palette):
    """arm -> (colour, marker, dash). Markers and line styles are identical in
    both palettes: they are already carrying identity and are not the thing
    under test."""
    styles = {}
    for i, arm in enumerate(EXTENSION_ORDER):
        styles[arm] = (EXTENSION_RAMP[i], EXTENSION_MARKERS[i],
                       EXTENSION_DASHES[i])
    if palette == "current":
        for i, arm in enumerate(LADDER_ORDER):
            styles[arm] = (CURRENT_LADDER_RAMP[i], LADDER_MARKERS[i], None)
        return styles
    stateless = [a for a in LADDER_ORDER if a in STATELESS]
    reading = [a for a in LADDER_ORDER if a not in STATELESS]
    if len(stateless) != len(STATELESS_RAMP) or len(reading) != len(STATE_READING_RAMP):
        raise SystemExit(
            f"FATAL: the grouped palette has {len(STATELESS_RAMP)} stateless and "
            f"{len(STATE_READING_RAMP)} state-reading steps, but the ladder holds "
            f"{len(stateless)} and {len(reading)}.")
    for group, ramp in ((stateless, STATELESS_RAMP), (reading, STATE_READING_RAMP)):
        for arm, colour in zip(group, ramp):
            styles[arm] = (colour, LADDER_MARKERS[LADDER_ORDER.index(arm)], None)
    return styles

TEXT_PRIMARY, TEXT_SECONDARY = "#0b0b0b", "#52514e"
GRID, AXIS, SURFACE, BAND = "#e6e5e1", "#bdbcb7", "#ffffff", "#f2f1ee"
REF_RULE, REF_RULE_2 = "#0b0b0b", "#7a7a76"

RC = {
    "font.size": 6.6, "axes.titlesize": 6.6, "axes.labelsize": 6.6,
    "xtick.labelsize": 5.9, "ytick.labelsize": 5.9, "legend.fontsize": 5.5,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 0.0, "ytick.major.size": 2.0, "grid.linewidth": 0.4,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.edgecolor": AXIS,
    "text.color": TEXT_PRIMARY, "axes.labelcolor": TEXT_PRIMARY,
    "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "legend.frameon": False, "axes.grid": False,
}

MS_MIN, MS_MAX = 2.3, 4.9
CELL_W, GROUP_GAP = 1.0, 0.62
DODGE_SPAN_A, DODGE_SPAN_B = 0.46, 0.38
SD_TOL = 1e-12          # below this the split is treated as deterministic


# ───────────────────────── reference line ─────────────────────────────────
def capacity_share(cfg_path, node):
    """weight(node) / sum(weights), read out of the router config.

    tomllib gives the replica table in file order, so the node's POSITION in the
    weight vector is read rather than assumed -- which is the part that would
    silently produce a wrong line if the config were ever reordered.
    """
    if not cfg_path.exists():
        raise SystemExit(f"FATAL: router config not found at {cfg_path}; "
                         f"refusing to assume a capacity-proportional share.")
    cfg = tomllib.loads(cfg_path.read_text())
    replicas = cfg.get("replicas") or []
    names = [r.get("name") for r in replicas]
    weights = [r.get("weight") for r in replicas]
    if node not in names:
        raise SystemExit(f"FATAL: {node} is not a replica in {cfg_path.name}: "
                         f"{names}")
    if any(not isinstance(w, (int, float)) for w in weights):
        raise SystemExit(f"FATAL: non-numeric weight in {cfg_path.name}: "
                         f"{weights}")
    idx = names.index(node)
    total = sum(weights)
    return {"path": cfg_path, "names": names, "weights": weights,
            "index": idx, "weight": weights[idx], "total": total,
            "share": weights[idx] / total}


def weights_in_runs():
    """The weight vector each arm actually ran, from run_meta.frozen_config.

    The config file on disk is what the reference line is computed from; this
    confirms the runs loaded the same numbers rather than a since-edited file.
    """
    by_arm = {}
    for mp in glob.glob(str(RUN_METAS)):
        try:
            m = json.loads(Path(mp).read_text())
        except OSError:
            continue
        arm = m.get("arm") or m.get("policy")
        fc = m.get("frozen_config") or {}
        w = tuple(fc.get("weights") or ())
        by_arm.setdefault(arm, {}).setdefault(w, 0)
        by_arm[arm][w] += 1
    return by_arm


# ───────────────────────── data ───────────────────────────────────────────
def load():
    if not CSV_PATH.exists():
        raise SystemExit(f"FATAL: {CSV_PATH} not found. Run analyze_sq3b.py first.")
    rows = [r for r in csv.DictReader(open(CSV_PATH)) if r["node"] == NODE]
    if not rows:
        raise SystemExit(f"FATAL: no {NODE} rows in {CSV_PATH}.")

    seen = {}
    for r in rows:
        seen.setdefault(r["workload"], set()).add(int(r["concurrency"]))
    unknown = set(seen) - {w for w, _ in WORKLOAD_ORDER}
    if unknown:
        raise SystemExit(f"FATAL: workloads with no declared display order: "
                         f"{sorted(unknown)}")
    suffix = {"uniform": "uni", "bimodal": "bim", "sharegpt": "sgpt"}
    cells, xpos, groups = [], [], []
    cursor = 0.0
    for wl, label in WORKLOAD_ORDER:
        if wl not in seen:
            continue
        start_i, start_x = len(cells), cursor
        for conc in sorted(seen[wl]):
            cells.append(f"c{conc}_{suffix[wl]}")
            xpos.append(cursor + CELL_W / 2)
            cursor += CELL_W
        groups.append((label, start_i, len(cells), start_x, cursor))
        cursor += GROUP_GAP

    present = {r["arm"] for r in rows}
    declared = set(LADDER_ORDER) | set(EXTENSION_ORDER)
    if present != declared:
        raise SystemExit(
            "FATAL: the declared ladder/extension split does not match the data.\n"
            f"  in CSV but not declared: {sorted(present - declared)}\n"
            f"  declared but not in CSV: {sorted(declared - present)}")

    data, omitted = {}, []
    for r in rows:
        n = int(r["n_runs"])
        key = (r["arm"], r["cell"])
        if n == 0:
            # No run passed the gate: the share is UNMEASURED here, which is a
            # different statement from a share of zero, so the point is omitted
            # rather than plotted at 0.
            omitted.append({"arm": r["arm"], "cell": r["cell"],
                            "n_all": int(r["n_all"]),
                            "n_excluded": int(r["n_excluded"])})
            continue
        sd = r["share_runsd"]
        data[key] = {
            "share": float(r["share_runmean"]),
            "sd": (float(sd) if sd not in ("", None) else None),
            "sd_defined": r["share_runsd_is_defined"] == "True",
            "n_runs": n, "n_all": int(r["n_all"]),
            "n_excluded": int(r["n_excluded"]),
            "share_min": float(r["share_min"]),
            "share_max": float(r["share_max"]),
            "share_pooled": float(r["share_pooled"]),
        }
    return data, cells, xpos, groups, omitted


def marker_size(n, nmin, nmax):
    if nmax == nmin:
        return (MS_MIN + MS_MAX) / 2
    t = (math.sqrt(n) - math.sqrt(nmin)) / (math.sqrt(nmax) - math.sqrt(nmin))
    return MS_MIN + t * (MS_MAX - MS_MIN)


def has_whisker(d):
    """A whisker is drawn only where the SD is meaningfully above zero. Three
    policies split deterministically, so their SD is zero (two of them exactly,
    one to within floating-point residue) and a zero-length whisker would be
    ink asserting a spread that does not exist."""
    return d["sd"] is not None and d["sd"] > SD_TOL


# ───────────────────────── drawing ────────────────────────────────────────
def style_axes(ax, cells, xpos, groups, ylo, yhi, show_ylabel):
    for _label, _s, _e, bx0, bx1 in groups:
        ax.axvspan(bx0, bx1, color=BAND, lw=0, zorder=0)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.4, linestyle="-", zorder=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_ylim(ylo, yhi)
    ax.set_xlim(groups[0][3] - 0.10, groups[-1][4] + 0.10)
    ax.set_xticks(xpos)
    ax.set_xticklabels([c.split("_")[0] for c in cells])
    ax.tick_params(axis="x", pad=1.0)
    if show_ylabel:
        ax.set_ylabel(f"Dispatch share to {NODE}", labelpad=2)


def ref_lines(ax, ref, ref2):
    """Both rules are named in the legend, with their weight vector and value,
    so no inline text is drawn: at this width an in-panel label lands on the
    data it is meant to explain."""
    ax.axhline(ref["share"], color=REF_RULE, linewidth=0.7,
               linestyle=(0, (4.0, 1.6)), zorder=2)
    if ref2 is not None:
        ax.axhline(ref2["share"], color=REF_RULE_2, linewidth=0.6,
                   linestyle=(0, (1.1, 1.3)), zorder=2)


def group_labels(ax, groups, y=-0.135, size=5.9):
    trans = ax.get_xaxis_transform()
    return [ax.text((bx0 + bx1) / 2, y, label, transform=trans, ha="center",
                    va="top", fontsize=size, color=TEXT_PRIMARY, clip_on=False)
            for label, _s, _e, bx0, bx1 in groups]


def draw_series(ax, arm, spec, data, cells, xpos, groups, nmin, nmax,
                subordinate, dx):
    colour, marker, dash = spec
    lw = 0.75 if subordinate else 1.2
    elw = 0.38 if subordinate else 0.5
    face = "none" if subordinate else colour
    # Lines are broken at every workload boundary: within a group the x axis is
    # an ordered concurrency sweep, across groups it would join two different
    # experiments. A group holding one cell gets markers and no line.
    for _label, s, e, _bx0, _bx1 in groups:
        seg = [(xpos[i] + dx, data[(arm, cells[i])]["share"])
               for i in range(s, e) if (arm, cells[i]) in data]
        if len(seg) < 2:
            continue
        ax.plot([p[0] for p in seg], [p[1] for p in seg], color=colour,
                linewidth=lw, zorder=3,
                linestyle=("-" if dash is None else (0, dash)),
                solid_capstyle="round")
    for xi, cell in zip(xpos, cells):
        d = data.get((arm, cell))
        if d is None:
            continue
        x = xi + dx
        ms = marker_size(d["n_runs"], nmin, nmax)
        if has_whisker(d):
            ax.errorbar(x, d["share"], yerr=d["sd"], ecolor=colour,
                        elinewidth=elw, capsize=0.9, capthick=elw,
                        fmt="none", zorder=3)
        ax.plot([x], [d["share"]], marker=marker, markersize=ms,
                markerfacecolor=face, markeredgecolor=colour,
                markeredgewidth=0.65, linestyle="none", zorder=5)


def dodges(arms, span):
    if len(arms) == 1:
        return {arms[0]: 0.0}
    step = span / (len(arms) - 1)
    return {a: -span / 2 + i * step for i, a in enumerate(arms)}


def find_dip(data, cells, ref, arm="state_aware"):
    """The arm's largest DOWNWARD departure from the reference line.

    Found from the data, never named in code, so the highlight cannot drift
    away from what the numbers say.
    """
    cand = [c for c in cells if (arm, c) in data]
    if not cand:
        return None
    cell = min(cand, key=lambda c: data[(arm, c)]["share"] - ref["share"])
    d = data[(arm, cell)]
    dev = d["share"] - ref["share"]
    if dev >= 0:
        return None
    return {"arm": arm, "cell": cell, "dev": dev, "share": d["share"]}


def mark_dip(ax, dip, data, cells, xpos, dx, nmin, nmax):
    """A ring around the marker rather than a leader and a text block.

    At single-column width a leader long enough to reach clear space crosses the
    dense 0.55-0.65 band, and the text lands on another series. The ring carries
    the location; the legend carries the number.
    """
    if dip is None or (dip["arm"], dip["cell"]) not in data:
        return
    d = data[(dip["arm"], dip["cell"])]
    x = xpos[cells.index(dip["cell"])] + dx.get(dip["arm"], 0.0)
    ms = marker_size(d["n_runs"], nmin, nmax)
    ax.plot([x], [d["share"]], marker="o", markersize=ms + 3.0,
            markerfacecolor="none", markeredgecolor=TEXT_PRIMARY,
            markeredgewidth=0.55, linestyle="none", zorder=6)


def arm_handle(arm, spec, subordinate, nmin, nmax, nref):
    colour, marker, dash = spec
    return Line2D([], [], color=colour,
                  linewidth=(0.75 if subordinate else 1.2),
                  linestyle=("-" if dash is None else (0, dash)),
                  marker=marker, markersize=marker_size(nref, nmin, nmax),
                  markerfacecolor=("none" if subordinate else colour),
                  markeredgecolor=colour, markeredgewidth=0.65,
                  label=ARM_LABEL[arm])


def size_key(nmin, nmax, nmid):
    return [Line2D([], [], marker="o", linestyle="none",
                   markersize=marker_size(n, nmin, nmax),
                   markerfacecolor=TEXT_SECONDARY,
                   markeredgecolor=TEXT_SECONDARY, markeredgewidth=0.0,
                   label=f"n={n}")
            for n in (nmin, nmid, nmax)]


def place_legend(fig, ax, group_texts, handles, ncol, extra):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    lowest = min(t.get_window_extent(renderer=r)
                 .transformed(fig.transFigure.inverted()).y0
                 for t in group_texts)
    x0 = ax.get_position().x0
    leg = fig.legend(handles=handles, loc="upper left",
                     bbox_to_anchor=(x0, lowest - 0.045),
                     bbox_transform=fig.transFigure, ncol=ncol,
                     handlelength=2.0, handletextpad=0.4, borderpad=0.0,
                     labelspacing=0.26, columnspacing=0.9)
    fig.canvas.draw()
    lb = leg.get_window_extent(renderer=r).transformed(fig.transFigure.inverted())
    fig.legend(handles=extra, loc="upper left",
               bbox_to_anchor=(x0, lb.y0 - 0.012),
               bbox_transform=fig.transFigure, ncol=3,
               handlelength=1.0, handletextpad=0.3, borderpad=0.0,
               columnspacing=0.7,
               title="marker area $\\propto$ runs", title_fontsize=5.5)


def save(fig, args, stem):
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in ("pdf", "png"):
        p = outdir / f"{stem}.{ext}"
        fig.savefig(p, dpi=args.dpi, bbox_inches="tight", pad_inches=0.015)
        written.append(p)
    plt.close(fig)
    return written


def legend_handles(styles, nmin, nmax, nmid, ref, ref2, dip):
    handles = ([arm_handle(a, styles[a], False, nmin, nmax, nmid)
                for a in LADDER_ORDER]
               + [arm_handle(a, styles[a], True, nmin, nmax, nmid)
                  for a in EXTENSION_ORDER])
    handles.append(Line2D([], [], color=REF_RULE, linewidth=0.7,
                          linestyle=(0, (4.0, 1.6)),
                          label=f"capacity-proportional, w={fmt_w(ref)} "
                                f"({ref['share']:.3f})"))
    handles.append(Line2D([], [], color=REF_RULE_2, linewidth=0.6,
                          linestyle=(0, (1.1, 1.3)),
                          label=f"same for wlc_conc, w={fmt_w(ref2)} "
                                f"({ref2['share']:.3f})"))
    if dip is not None:
        handles.append(Line2D(
            [], [], marker="o", linestyle="none", markersize=4.6,
            markerfacecolor="none", markeredgecolor=TEXT_PRIMARY,
            markeredgewidth=0.55,
            label=f"{ARM_LABEL[dip['arm']]}'s largest dip: "
                  f"{dip['cell']}, {dip['dev']:+.3f}"))
    return handles


def fmt_w(ref):
    return "/".join(f"{w:g}" for w in ref["weights"])


# ───────────────────────── figures ────────────────────────────────────────
def figure_A(data, cells, xpos, groups, styles, ylo, yhi, nmin, nmax, nmid,
             ref, ref2, args, suffix=""):
    dx = dodges(LADDER_ORDER + EXTENSION_ORDER, DODGE_SPAN_A)
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(args.width, args.height_a))
        fig.subplots_adjust(left=0.155, right=0.99, top=0.985, bottom=0.42)
        style_axes(ax, cells, xpos, groups, ylo, yhi, True)
        ref_lines(ax, ref, ref2)
        for arm in EXTENSION_ORDER:
            draw_series(ax, arm, styles[arm], data, cells, xpos, groups,
                        nmin, nmax, True, dx[arm])
        for arm in LADDER_ORDER:
            draw_series(ax, arm, styles[arm], data, cells, xpos, groups,
                        nmin, nmax, False, dx[arm])
        dip = find_dip(data, cells, ref)
        mark_dip(ax, dip, data, cells, xpos, dx, nmin, nmax)
        gl = group_labels(ax, groups)
        place_legend(fig, ax, gl,
                     legend_handles(styles, nmin, nmax, nmid, ref, ref2, dip),
                     ncol=2, extra=size_key(nmin, nmax, nmid))
        return save(fig, args, "sq3b_dispatch_A_single_panel" + suffix), dip


def figure_B(data, cells, xpos, groups, styles, ylo, yhi, nmin, nmax, nmid,
             ref, ref2, args, suffix=""):
    dxl = dodges(LADDER_ORDER, DODGE_SPAN_B)
    dxe = dodges(EXTENSION_ORDER, DODGE_SPAN_B)
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(args.width, args.height_b),
                                 sharey=True)
        fig.subplots_adjust(left=0.155, right=0.99, top=0.93, bottom=0.42,
                            wspace=0.07)
        for ax, arms, sub, dx, title in (
                (axes[0], LADDER_ORDER, False, dxl, "baseline ladder"),
                (axes[1], EXTENSION_ORDER, True, dxe, "extensions")):
            style_axes(ax, cells, xpos, groups, ylo, yhi, ax is axes[0])
            # The wlc_conc line belongs on the panel wlc_conc is drawn on.
            ref_lines(ax, ref, ref2 if ax is axes[1] else None)
            for arm in arms:
                draw_series(ax, arm, styles[arm], data, cells, xpos, groups,
                            nmin, nmax, sub, dx[arm])
            ax.set_title(title, pad=2.0, color=TEXT_PRIMARY)
        dip = find_dip(data, cells, ref)
        mark_dip(axes[0], dip, data, cells, xpos, dxl, nmin, nmax)
        axes[1].tick_params(axis="y", length=0)
        gl = group_labels(axes[0], groups) + group_labels(axes[1], groups)
        place_legend(fig, axes[0], gl,
                     legend_handles(styles, nmin, nmax, nmid, ref, ref2, dip),
                     ncol=2, extra=size_key(nmin, nmax, nmid))
        return save(fig, args, "sq3b_dispatch_B_two_panel" + suffix), dip


# ───────────────────────── main ───────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=str(HERE))
    ap.add_argument("--width", type=float, default=3.33,
                    help="figure width in inches (default 3.33 = ACM single column)")
    ap.add_argument("--height-a", type=float, default=3.15)
    ap.add_argument("--height-b", type=float, default=3.15)
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--palette", choices=("current", "grouped", "both"),
                    default="both")
    args = ap.parse_args()

    data, cells, xpos, groups, omitted = load()
    arms = LADDER_ORDER + EXTENSION_ORDER

    print("=" * 100)
    print(f"Source: {CSV_PATH}")
    print(f"Filter: node == {NODE}")
    print("=" * 100)

    # ---- reference line ----
    ref = capacity_share(ROUTER_CFG, NODE)
    ref2 = capacity_share(ROUTER_CFG_CONC, NODE)
    print("\nREFERENCE LINE — computed, not typed")
    for tag, rr in (("primary", ref), ("wlc_conc", ref2)):
        print(f"  {tag:<9} {rr['path'].name}")
        print(f"    replicas in file order: "
              + ", ".join(f"{n}={w:g}" for n, w in zip(rr["names"], rr["weights"])))
        print(f"    {NODE} is replica #{rr['index'] + 1}; share = "
              f"{rr['weight']:g} / {rr['total']:g} = {rr['share']:.6f}")
    by_arm = weights_in_runs()
    print("\n  cross-check against frozen_config.weights in every run_meta:")
    mismatch = []
    for arm in arms:
        seen = by_arm.get(arm, {})
        want = tuple(ref2["weights"]) if arm == CONC_ARM else tuple(ref["weights"])
        ok = set(seen) == {want}
        if not ok:
            mismatch.append((arm, seen, want))
        print(f"    {arm:<28} "
              + "; ".join(f"{list(w)} x{n}" for w, n in seen.items())
              + ("  OK" if ok else f"  *** expected {list(want)}"))
    if mismatch:
        print("  *** the runs did not all load the weights the config on disk "
              "states.")
    print(f"\n  The {CONC_ARM} arm runs a DIFFERENT weight vector, so the primary")
    print(f"  line is not its capacity-proportional share. A second, subordinate")
    print(f"  line at {ref2['share']:.3f} is drawn for it; without that, "
          f"{CONC_ARM} would")
    print(f"  read as over-concentrating when it is doing what it was configured "
          f"to do.")

    print(f"\nCells plotted, in display order: {cells}")
    for label, s, e, bx0, bx1 in groups:
        print(f"  {label:<10} {cells[s:e]} at x="
              f"{[round(v, 2) for v in xpos[s:e]]}, band [{bx0:.2f}, {bx1:.2f}]")

    # ---- every plotted value ----
    ns = [d["n_runs"] for d in data.values()]
    nmin, nmax = min(ns), max(ns)
    nmid = sorted(ns)[len(ns) // 2]
    print("\n" + "-" * 100)
    print("EVERY PLOTTED VALUE  (share_runmean +- share_runsd; sample SD, ddof=1)")
    print("-" * 100)
    print(f"{'arm':<28}{'cell':<9}{'n':>4}{'excl':>6}{'share':>9}{'sd':>10}"
          f"{'lo':>9}{'hi':>9}{'dev ref':>10}{'whisker':>9}")
    for arm in arms:
        for cell in cells:
            d = data.get((arm, cell))
            if d is None:
                print(f"{arm:<28}{cell:<9}{'-':>4}{'-':>6}{'-':>9}{'-':>10}"
                      f"{'-':>9}{'-':>9}{'-':>10}{'-':>9}  OMITTED (no run "
                      f"passed the gate)")
                continue
            sd = d["sd"] if d["sd"] is not None else 0.0
            w = has_whisker(d)
            print(f"{arm:<28}{cell:<9}{d['n_runs']:>4}{d['n_excluded']:>6}"
                  f"{d['share']:>9.4f}{sd:>10.2e}"
                  f"{d['share'] - sd:>9.4f}{d['share'] + sd:>9.4f}"
                  f"{d['share'] - ref['share']:>+10.4f}"
                  f"{('yes' if w else 'no'):>9}")

    # ---- verification grid, for diffing against the LaTeX table ----
    print("\n" + "-" * 100)
    print("VERIFICATION GRID — exactly what is plotted, to 3 dp")
    print("-" * 100)
    print("Every number below is share_runmean / share_runsd read straight from "
          "the CSV and")
    print("passed to the plotting call unchanged: no rounding, clipping, "
          "rescaling, smoothing,")
    print("re-aggregation or unit change is applied anywhere between the file "
          "and the marker.")
    print("The 3 dp here is display only; the plotted value is the full float.")
    print(f"\nSHARE (share_runmean)")
    print(f"  {'policy':<28}" + "".join(f"{c:>10}" for c in cells))
    for arm in arms:
        row = ""
        for cell in cells:
            d = data.get((arm, cell))
            row += f"{'---' if d is None else f'{d["share"]:.3f}':>10}"
        print(f"  {arm:<28}{row}")
    print(f"\nSD (share_runsd, sample SD, ddof=1; '.' = whisker suppressed)")
    print(f"  {'policy':<28}" + "".join(f"{c:>10}" for c in cells))
    for arm in arms:
        row = ""
        for cell in cells:
            d = data.get((arm, cell))
            if d is None:
                row += f"{'---':>10}"
            elif not has_whisker(d):
                row += f"{f'{d["sd"]:.3f}.':>10}"
            else:
                row += f"{d["sd"]:>10.3f}"
        print(f"  {arm:<28}{row}")
    print("  A trailing '.' marks a suppressed whisker. Those SDs are 0.000 at "
          "3 dp;")
    print("  see the determinism section below for which are exactly zero.")

    # ---- omitted cells ----
    print(f"\nCells OMITTED because no run passed the validity gate "
          f"({len(omitted)}):")
    for o in omitted:
        print(f"  {o['arm']:<28}{o['cell']:<9} {o['n_all']} run(s) attempted, "
              f"all {o['n_excluded']} excluded")
    print("  These are omitted, not plotted at zero: the share is UNMEASURED "
          "there,")
    print("  which is a different statement from a share of nothing.")

    # ---- deterministic splits ----
    print("\n" + "-" * 100)
    print("DETERMINISTIC SPLITS — whiskers suppressed (for the caption)")
    print("-" * 100)
    zero_rows = [(a, c, data[(a, c)]) for a in arms for c in cells
                 if (a, c) in data and not has_whisker(data[(a, c)])]
    exact = [(a, c, d) for a, c, d in zero_rows if d["sd"] == 0.0]
    residue = [(a, c, d) for a, c, d in zero_rows if d["sd"] != 0.0]
    drawn = [data[(a, c)]["sd"] for a in arms for c in cells
             if (a, c) in data and has_whisker(data[(a, c)])]
    print(f"Tolerance: a whisker is drawn only where share_runsd > {SD_TOL:g}.")
    print(f"  rows with no whisker: {len(zero_rows)}   "
          f"of which SD is EXACTLY 0.0: {len(exact)}, "
          f"floating-point residue: {len(residue)}")
    if residue:
        print(f"  residue values: "
              f"{sorted({f'{d['sd']:.3e}' for _a, _c, d in residue})}")
    if drawn:
        print(f"  largest suppressed SD {max(d['sd'] for _a, _c, d in zero_rows):.3e}"
              f" vs smallest drawn SD {min(drawn):.3e} — the threshold sits in a "
              f"gap of ~{math.log10(min(drawn) / max(1e-300, max(d['sd'] for _a, _c, d in zero_rows))):.0f} "
              f"orders of magnitude.")
    per_arm = {}
    for a in arms:
        rows = [(a, c) for c in cells if (a, c) in data]
        if rows and all(not has_whisker(data[k]) for k in rows):
            per_arm[a] = len(rows)
    print(f"\nPolicies with no whisker in ANY plotted cell ({len(per_arm)}):")
    for a, n in per_arm.items():
        sds = {data[(a, c)]["sd"] for c in cells if (a, c) in data}
        kind = ("exactly 0.0" if sds == {0.0}
                else f"{max(sds):.3e} (floating-point residue, not exactly 0)")
        print(f"  {a:<28} {n} cell(s), SD {kind}")
    other = [(a, c) for a, c, d in zero_rows if a not in per_arm]
    if other:
        print(f"\nIndividual cells with a zero SD in a policy that is NOT "
              f"deterministic overall ({len(other)}):")
        for a, c in other:
            d = data[(a, c)]
            print(f"  {a:<28}{c:<9} share={d['share']:.4f} over {d['n_runs']} "
                  f"runs — every run landed on the same split in this cell")
    print("\nCAPTION FACT: three policies split deterministically and carry no "
          "whisker in")
    print("any cell. Only two of them record an SD of EXACTLY zero; the third "
          "records")
    print("floating-point residue near 1e-17, which is why the suppression rule "
          "is a")
    print("tolerance rather than an equality test.")

    # ---- the two things the prose argues ----
    print("\n" + "-" * 100)
    print("WHAT THE PROSE ARGUES — checked against the data")
    print("-" * 100)
    pair = ("weighted_least_connections", "state_aware")
    print(f"1. {pair[0]} vs {pair[1]}, per cell:")
    print(f"  {'cell':<9}{'wlc':>9}{'state_aware':>13}{'gap':>9}")
    gaps = []
    for cell in cells:
        a, b = data.get((pair[0], cell)), data.get((pair[1], cell))
        if not a or not b:
            continue
        g = abs(a["share"] - b["share"])
        gaps.append((cell, g))
        print(f"  {cell:<9}{a['share']:>9.4f}{b['share']:>13.4f}{g:>9.4f}")
    if gaps:
        wc, wg = max(gaps, key=lambda t: t[1])
        print(f"  max gap {wg:.4f} at {wc}; mean gap "
              f"{sum(g for _c, g in gaps) / len(gaps):.4f} over {len(gaps)} cells")

    state_reading = [a for a in arms if a not in STATELESS]
    devs = sorted(((a, c, data[(a, c)]["share"] - ref["share"])
                   for a in state_reading for c in cells if (a, c) in data),
                  key=lambda t: -abs(t[2]))
    print(f"\n2. Deviation from the reference line, state-reading policies "
          f"({len(state_reading)} arms), ranked by |deviation|:")
    print(f"  {'rank':<6}{'arm':<28}{'cell':<9}{'share':>9}{'deviation':>11}")
    for i, (a, c, dv) in enumerate(devs[:8], start=1):
        print(f"  {i:<6}{a:<28}{c:<9}{data[(a, c)]['share']:>9.4f}{dv:>+11.4f}")
    sa = [(a, c, dv) for a, c, dv in devs if a == "state_aware"]
    sa_worst = max(sa, key=lambda t: abs(t[2])) if sa else None
    down = sorted((t for t in devs if t[2] < 0), key=lambda t: t[2])
    print(f"\n  state_aware's largest |deviation|: "
          f"{sa_worst[1]} {sa_worst[2]:+.4f} (rank "
          f"{[i for i, t in enumerate(devs, 1) if t[:2] == sa_worst[:2]][0]} of "
          f"{len(devs)})")
    print(f"  largest |deviation| overall:        {devs[0][0]} {devs[0][1]} "
          f"{devs[0][2]:+.4f}")
    print(f"  largest DOWNWARD deviation:         {down[0][0]} {down[0][1]} "
          f"{down[0][2]:+.4f}")
    if devs[0][0] != "state_aware" or down[0][0] != "state_aware":
        print("\n  *** NOTE: state_aware's c4_uni dip is NOT the largest "
              "deviation from the")
        print("  reference among the state-reading policies, on either reading. "
              "The figure")
        print("  annotates state_aware's own largest downward departure and "
              "reports this")
        print("  ranking rather than asserting the stronger claim.")

    # ---- y window ----
    lo = min(d["share"] - (d["sd"] if has_whisker(d) else 0.0)
             for d in data.values())
    hi = max(d["share"] + (d["sd"] if has_whisker(d) else 0.0)
             for d in data.values())
    lo, hi = min(lo, ref["share"], ref2["share"]), max(hi, ref["share"],
                                                       ref2["share"])
    pad = (hi - lo) * 0.09
    ylo, yhi = lo - pad, hi + pad
    print(f"\nLowest plotted extent {lo:.4f}, highest {hi:.4f} (both reference "
          f"lines included in the window), pad {pad:.4f}")
    print(f"y limits: [{ylo:.4f}, {yhi:.4f}] — not anchored at zero; the data "
          f"spans {lo:.2f}-{hi:.2f}")
    print(f"and the equal-split value 1/3 = {1/3:.4f} sits just below the "
          f"lowest point.")

    print(f"\nRun counts: min {nmin}, max {nmax}, median {nmid}; marker AREA is "
          f"proportional to n_runs.")

    palettes = (("current", ""), ("grouped", "_grouped"))
    if args.palette != "both":
        palettes = [(p, sfx) for p, sfx in palettes if p == args.palette]
    written, dip_a = [], None
    for name, sfx in palettes:
        styles = build_styles(name)
        print(f"\nPalette '{name}':")
        print(f"  {'arm':<28}{'family':<16}{'colour':<10}{'marker':<8}line")
        for arm in arms:
            c, m, dsh = styles[arm]
            fam = ("extension" if arm in EXTENSION_ORDER
                   else ("stateless" if arm in STATELESS else "state-reading"))
            print(f"  {arm:<28}{fam:<16}{c:<10}{m:<8}"
                  f"{'dashed' if dsh else 'solid'}")
        wa, da = figure_A(data, cells, xpos, groups, styles, ylo, yhi, nmin,
                          nmax, nmid, ref, ref2, args, sfx)
        wb, _db = figure_B(data, cells, xpos, groups, styles, ylo, yhi, nmin,
                           nmax, nmid, ref, ref2, args, sfx)
        written += wa + wb
        dip_a = dip_a or da
    if dip_a:
        print(f"\nRinged dip (found from the data, not named in code): "
              f"{dip_a['arm']} at {dip_a['cell']}, share {dip_a['share']:.4f}, "
              f"{dip_a['dev']:+.4f} from the reference line. The ring marks the "
              f"location; the legend carries the number.")
    print()
    for p in written:
        print(f"Wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
