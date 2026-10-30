import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.lines import Line2D                   # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "3e_results" / "analysis").is_dir())
ANALYSIS = REPO / "3e_results" / "analysis"
CSV_PATH = ANALYSIS / "sq3" / "a" / "sq3a_ladder_completion.csv"

# ── declared presentation order (not data) ────────────────────────────────
PANELS = [
    ("A", "random", "round_robin"),
    ("B", "round_robin", "weighted_round_robin"),
    ("C", "round_robin", "least_connections"),
    ("D", "least_connections", "weighted_least_connections"),
    ("E", "weighted_least_connections", "state_aware"),
]
WORKLOAD_ORDER = [("uniform", "uniform"), ("bimodal", "bimodal"),
                  ("sharegpt", "ShareGPT")]
ARM_LABEL = {
    "random": "random", "round_robin": "round robin",
    "weighted_round_robin": "weighted round robin",
    "least_connections": "least connections",
    "weighted_least_connections": "weighted least connections",
    "state_aware": "state aware",
}

ALPHA = 0.05          # only for LABELLING; the CSV carries the tested values

SIG, NULL = "#14407a", "#9b9b98"      # dark/filled vs light/open: greyscale-safe
TEXT_PRIMARY, TEXT_SECONDARY = "#0b0b0b", "#52514e"
GRID, AXIS, SURFACE, BAND = "#e6e5e1", "#bdbcb7", "#ffffff", "#f4f3f0"
ZERO_RULE = "#5c5b57"

RC = {
    "font.size": 6.6, "axes.titlesize": 6.8, "axes.labelsize": 6.8,
    "xtick.labelsize": 6.0, "ytick.labelsize": 6.0, "legend.fontsize": 5.9,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.0, "ytick.major.size": 0.0, "grid.linewidth": 0.4,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.edgecolor": AXIS,
    "text.color": TEXT_PRIMARY, "axes.labelcolor": TEXT_PRIMARY,
    "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "legend.frameon": False, "axes.grid": False,
}

ROW_H = 0.132         # inches per forest row; figure height follows from it
PANEL_GAP = 0.30
TOP_PAD, BOTTOM_PAD = 0.34, 0.86
COL_N, COL_P = 1.035, 1.135      # text columns, in axes-width fractions


def num(v):
    return None if v in ("", None) else float(v)


def load():
    if not CSV_PATH.exists():
        raise SystemExit(f"FATAL: {CSV_PATH} not found. Run analyze_sq3a.py first.")
    with open(CSV_PATH) as f:
        allrows = list(csv.DictReader(f))
    rows = [r for r in allrows if r["in_grid"] == "True"]
    offgrid = [r for r in allrows if r["in_grid"] != "True"]
    if not rows:
        raise SystemExit(f"FATAL: no in_grid rows in {CSV_PATH}.")

    steps_in_file = list(dict.fromkeys((r["base_arm"], r["cand_arm"]) for r in rows))
    declared = [(b, c) for _l, b, c in PANELS]
    if steps_in_file != declared:
        raise SystemExit(
            "FATAL: the declared panel list does not match the steps in the CSV.\n"
            f"  in CSV:   {steps_in_file}\n"
            f"  declared: {declared}")

    # Cell display order, derived from the file.
    seen = {}
    for r in rows:
        seen.setdefault(r["workload"], set()).add(int(r["concurrency"]))
    unknown = set(seen) - {w for w, _ in WORKLOAD_ORDER}
    if unknown:
        raise SystemExit(f"FATAL: workloads with no declared display order: "
                         f"{sorted(unknown)}")
    suffix = {"uniform": "uni", "bimodal": "bim", "sharegpt": "sgpt"}
    cells, groups = [], []
    for wl, label in WORKLOAD_ORDER:
        if wl not in seen:
            continue
        start = len(cells)
        for conc in sorted(seen[wl]):
            cells.append(f"c{conc}_{suffix[wl]}")
        groups.append((label, start, len(cells)))

    data = {}
    for r in rows:
        lo, hi, p = num(r["ci_low_pp"]), num(r["ci_high_pp"]), num(r["p_permutation"])
        tested = r["tested"] == "True"
        ci_excludes = (tested and lo is not None
                       and not (lo <= 0.0 <= hi))
        p_sig = tested and p is not None and p <= ALPHA
        data[(r["base_arm"], r["cand_arm"], r["cell"])] = {
            "tested": tested,
            "diff": num(r["diff_pp"]), "lo": lo, "hi": hi, "p": p,
            "n_base": int(r["n_base"]), "n_cand": int(r["n_cand"]),
            "enum": r["enumeration"],
            "amp": num(r["attainable_min_p"]),
            "verdict": r["verdict"],
            "why": r["untestable_reason"],
            "ci_excludes_zero": ci_excludes,
            "p_significant": p_sig,
            "disagree": tested and (ci_excludes != p_sig),
            "degenerate_ci": (tested and lo is not None and hi is not None
                              and hi - lo == 0.0),
        }
    return data, cells, groups, offgrid


def fmt_p(p):
    """p as the prose quotes it. '<0.001' rather than '0.000', because a printed
    zero would claim more than an estimated p-value can support."""
    if p is None:
        return "--"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ───────────────────────── drawing ────────────────────────────────────────
def draw_panel(ax, letter, base, cand, data, cells, groups, xlo, xhi,
               is_bottom, is_top):
    ys = list(range(len(cells)))[::-1]     # first cell at the top
    for _label, s, e in groups:
        if s % 2 == 0:                     # alternate bands, by workload group
            ax.axhspan(ys[e - 1] - 0.5, ys[s] + 0.5, color=BAND, lw=0, zorder=0)
    ax.set_axisbelow(True)
    ax.axvline(0.0, color=ZERO_RULE, linewidth=0.7, zorder=2)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(-0.7, len(cells) - 0.3)
    ax.set_yticks(ys)
    ax.set_yticklabels(cells)
    ax.tick_params(axis="y", length=0, pad=1.5)
    if not is_bottom:
        ax.tick_params(axis="x", labelbottom=False, length=0)
        ax.spines["bottom"].set_visible(False)

    ax.set_title(f"{letter}   {ARM_LABEL[base]} $\\rightarrow$ {ARM_LABEL[cand]}",
                 loc="left", pad=2.5, color=TEXT_PRIMARY)

    if is_top:
        for xf, head in ((COL_N, "n"), (COL_P, "p")):
            ax.text(xf, 1.0, head, transform=ax.transAxes, ha="center",
                    va="bottom", fontsize=6.0, color=TEXT_SECONDARY,
                    clip_on=False)

    for y, cell in zip(ys, cells):
        d = data[(base, cand, cell)]
        # Comparisons with fewer than two runs keep their row, so all panels have the same rows.
        if not d["tested"]:
            ax.text(0.5, y, "untestable", transform=ax.get_yaxis_transform(),
                    ha="center", va="center", fontsize=5.6, style="italic",
                    color=TEXT_SECONDARY, clip_on=False)
        else:
            # Shading follows the permutation test at alpha; the bootstrap interval is drawn but does not set it.
            solid = d["p_significant"]
            colour = SIG if solid else NULL
            ax.plot([d["lo"], d["hi"]], [y, y], color=colour,
                    linewidth=(1.15 if solid else 0.85),
                    solid_capstyle="butt", zorder=3)
            for xend in (d["lo"], d["hi"]):
                ax.plot([xend, xend], [y - 0.17, y + 0.17], color=colour,
                        linewidth=(1.15 if solid else 0.85), zorder=3)
            ax.plot([d["diff"]], [y], marker="o", markersize=3.4,
                    markerfacecolor=(colour if solid else SURFACE),
                    markeredgecolor=colour, markeredgewidth=0.85,
                    linestyle="none", zorder=4)

        ax.text(COL_N, y, f"{d['n_base']}v{d['n_cand']}",
                transform=ax.get_yaxis_transform(), ha="center", va="center",
                fontsize=5.8, color=TEXT_SECONDARY, clip_on=False)
        # Flags, appended to the p column so they sit beside the number they
        # qualify: a dagger where the two significance criteria disagree, a
        # double dagger where the run counts cannot reach alpha at all.
        flags = ("†" if d["disagree"] else "") \
            + ("‡" if (d["amp"] is not None and d["amp"] > ALPHA) else "")
        ax.text(COL_P, y, fmt_p(d["p"]) + flags,
                transform=ax.get_yaxis_transform(), ha="center", va="center",
                fontsize=5.8,
                color=(TEXT_PRIMARY if d["p_significant"] else TEXT_SECONDARY),
                clip_on=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=str(HERE))
    ap.add_argument("--width", type=float, default=7.0,
                    help="figure width in inches (default 7.0 = ACM full text width)")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--stem", default="sq3a_ladder_forest")
    args = ap.parse_args()

    data, cells, groups, offgrid = load()

    print("=" * 104)
    print(f"Source: {CSV_PATH}")
    print("Quantity drawn: the DIFFERENCE between two arms (candidate minus "
          "baseline, percentage")
    print("points) with analyze_3e.bootstrap_ci's percentile interval on that "
          "difference. Per-arm")
    print("SD/SE is deliberately NOT drawn: it does not determine whether the "
          "difference is")
    print("significant, and the interval on the difference does.")
    print("=" * 104)
    if offgrid:
        seen_cells = sorted({r["cell"] for r in offgrid})
        print(f"Off-grid rows excluded by their own in_grid flag: {len(offgrid)} "
              f"({', '.join(seen_cells)})")
    print(f"Rows per panel: {len(cells)} -> {cells}")
    print("Workload groups: " + ", ".join(f"{lab} [{s}:{e}]"
                                          for lab, s, e in groups))

    # ---- every plotted value ----
    print("\n" + "-" * 104)
    print("EVERY PLOTTED VALUE")
    print("-" * 104)
    print(f"{'panel':<6}{'step':<52}{'cell':<9}{'n':>7}{'diff':>7}"
          f"{'CI low':>8}{'CI high':>9}{'p':>9}{'enum':>11}{'min p':>9}")
    for letter, base, cand in PANELS:
        step = f"{base} -> {cand}"
        for cell in cells:
            d = data[(base, cand, cell)]
            n = f"{d['n_base']}v{d['n_cand']}"
            amp = "--" if d["amp"] is None else f"{d['amp']:.4f}"
            if not d["tested"]:
                print(f"{letter:<6}{step:<52}{cell:<9}{n:>7}"
                      f"{'--':>7}{'--':>8}{'--':>9}{'--':>9}{'--':>11}{amp:>9}"
                      f"   UNTESTABLE: {d['why']}")
                continue
            print(f"{letter:<6}{step:<52}{cell:<9}{n:>7}"
                  f"{d['diff']:>7.1f}{d['lo']:>8.1f}{d['hi']:>9.1f}"
                  f"{fmt_p(d['p']):>9}{d['enum']:>11}{amp:>9}"
                  + ("  CI excludes 0" if d["ci_excludes_zero"] else "")
                  + ("  p<=alpha" if d["p_significant"] else "")
                  + ("  ** DISAGREE **" if d["disagree"] else ""))

    flat = [data[(b, c, cell)] for _l, b, c in PANELS for cell in cells]
    tested = [d for d in flat if d["tested"]]
    untestable = [d for d in flat if not d["tested"]]

    # ---- resolution limit ----
    print("\n" + "-" * 104)
    print("RESOLUTION LIMIT (attainable_min_p > alpha)")
    print("-" * 104)
    limited = [(l, b, c, cell) for l, b, c in PANELS for cell in cells
               if (data[(b, c, cell)]["amp"] is not None
                   and data[(b, c, cell)]["amp"] > ALPHA)]
    if limited:
        print(f"{len(limited)} row(s) whose run counts CANNOT produce a "
              f"significant result at alpha={ALPHA},")
        print("however large the effect. A non-significant outcome there is a "
              "resolution limit,")
        print("not evidence of no difference. Marked on the figure with a "
              "double dagger:")
        for l, b, c, cell in limited:
            d = data[(b, c, cell)]
            print(f"  panel {l}  {cell:<9} {d['n_base']}v{d['n_cand']}  "
                  f"attainable_min_p={d['amp']:.4f} > {ALPHA}")
    else:
        amps = [d["amp"] for d in tested if d["amp"] is not None]
        print(f"NONE. Every plotted comparison could in principle have reached "
              f"alpha={ALPHA}:")
        print(f"  attainable_min_p across the {len(amps)} tested rows ranges "
              f"{min(amps):.5f} to {max(amps):.4f}, all <= {ALPHA}.")
        print("  So no non-significant row here can be dismissed as a "
              "resolution limit -- each one")
        print("  had the power to resolve a large enough effect and did not "
              "find one. The double-")
        print("  dagger flag is implemented and simply never fires; it is "
              "omitted from the legend.")

    # ---- untestable ----
    print("\n" + "-" * 104)
    print("UNTESTABLE ROWS (fewer than two runs in an arm)")
    print("-" * 104)
    if untestable:
        print(f"{len(untestable)} row(s), kept as labelled gaps so every panel "
              f"has the same rows.")
    else:
        print("NONE among the grid cells. Every untestable comparison in the "
              "CSV was in c4_sgpt,")
        print("which is off-grid and excluded above, so all "
              f"{len(PANELS)} x {len(cells)} = {len(flat)} plotted rows carry a "
              f"real test.")
        print("The labelled-gap path is implemented and simply never fires.")

    # ---- the two criteria ----
    print("\n" + "-" * 104)
    print("THE TWO SIGNIFICANCE CRITERIA DISAGREE ON SOME ROWS")
    print("-" * 104)
    dis = [(l, cell, data[(b, c, cell)]) for l, b, c in PANELS for cell in cells
           if data[(b, c, cell)]["disagree"]]
    n_ci = sum(1 for d in tested if d["ci_excludes_zero"])
    n_p = sum(1 for d in tested if d["p_significant"])
    print(f"Of {len(tested)} tested rows: {n_p} have p <= {ALPHA}, "
          f"{n_ci} have a CI excluding zero.")
    print(f"SHADING FOLLOWS p, the declared decision rule. The methodology "
          f"declares the two-sided")
    print("permutation test at alpha as the test and reports the bootstrap CI "
          "alongside each")
    print("effect rather than as the test, so the figure shades by p and keeps "
          "the interval")
    print(f"drawn on every row. The two criteria disagree on {len(dis)} row(s); "
          f"each carries a")
    print("dagger, meaning the bootstrap interval would have given a different "
          "verdict:")
    for l, cell, d in dis:
        which = ("CI excludes 0 but p > alpha -> now drawn OPEN"
                 if d["ci_excludes_zero"]
                 else "p <= alpha but CI contains 0 -> now drawn SOLID")
        print(f"  panel {l}  {cell:<9} {d['n_base']}v{d['n_cand']:<3} "
              f"diff={d['diff']:+.1f}  CI=[{d['lo']:+.1f}, {d['hi']:+.1f}]  "
              f"p={fmt_p(d['p'])}  -- {which}")
    print("\nWhy they can differ: the permutation test is a DISCRETE exact test "
          "on the median")
    print("statistic, so at small run counts its p-value can only take a few "
          "values; the")
    print("bootstrap resamples runs, and when an arm's runs are nearly "
          "identical every resample")
    print("is nearly identical too, so the interval collapses and excludes zero "
          "while the")
    print("permutation test still cannot reject.")

    # ---- degenerate intervals: the caption needs this ----
    print("\n" + "-" * 104)
    print("ZERO-WIDTH INTERVALS (for the caption)")
    print("-" * 104)
    degen = [(l, cell, data[(b, c, cell)]) for l, b, c in PANELS
             for cell in cells if data[(b, c, cell)]["degenerate_ci"]]
    print(f"{len(degen)} of {len(tested)} tested rows have a CI of width exactly "
          f"zero. On the figure each")
    print("is a lone marker sitting on the zero rule with no visible bar: that "
          "is a DEGENERATE")
    print("INTERVAL, not a missing value. It arises where every run in both arms "
          "completed")
    print("identically, so every bootstrap resample reproduces the same "
          "difference.")
    if degen:
        print(f"\n{'panel':<7}{'cell':<9}{'n':>7}{'diff':>7}"
              f"{'CI':>14}{'p':>9}")
        for l, cell, d in degen:
            n = f"{d['n_base']}v{d['n_cand']}"
            ci = f"[{d['lo']:+.1f}, {d['hi']:+.1f}]"
            print(f"{l:<7}{cell:<9}{n:>7}{d['diff']:>+7.1f}{ci:>14}"
                  f"{fmt_p(d['p']):>9}")
        by_cell = {}
        for _l, cell, _d in degen:
            by_cell[cell] = by_cell.get(cell, 0) + 1
        print(f"\nBy cell: "
              + ", ".join(f"{c} ({n})" for c, n in by_cell.items()))
        nz = [d for _l, _c, d in degen if d["diff"] != 0.0]
        print(f"All {len(degen)} sit at diff = 0.0"
              if not nz else
              f"{len(nz)} of them sit at a non-zero difference: "
              + ", ".join(f"{d['diff']:+.1f}" for d in nz))

    # ---- x window ----
    lo = min(d["lo"] for d in tested)
    hi = max(d["hi"] for d in tested)
    pad = (hi - lo) * 0.06
    xlo, xhi = lo - pad, hi + pad
    print("\n" + "-" * 104)
    print("SHARED X AXIS")
    print("-" * 104)
    print(f"Global interval extent [{lo:+.1f}, {hi:+.1f}] pp; drawn window "
          f"[{xlo:+.1f}, {xhi:+.1f}].")
    print(f"{'panel':<6}{'step':<52}{'panel extent':>18}{'share of window':>18}")
    worst = None
    for letter, base, cand in PANELS:
        ds = [data[(base, cand, cell)] for cell in cells
              if data[(base, cand, cell)]["tested"]]
        plo, phi = min(d["lo"] for d in ds), max(d["hi"] for d in ds)
        share = (phi - plo) / (xhi - xlo)
        print(f"{letter:<6}{base + ' -> ' + cand:<52}"
              f"{f'[{plo:+.1f}, {phi:+.1f}]':>18}{share:>17.1%}")
        if worst is None or share < worst[1]:
            worst = (letter, share)
    print(f"\nSmallest panel occupies {worst[1]:.0%} of the shared window "
          f"(panel {worst[0]}).")
    print("KEPT SHARED. Panel E's intervals sit near zero on the same scale as "
          "panels B and C,")
    print("which is the comparison the figure exists to make: a per-panel x "
          "axis would rescale")
    print("E's near-null result to look as wide as B's 20-point effect. The "
          "narrowest panel")
    print("still spans a third of the window, so nothing is compressed to "
          "invisibility.")

    # ---- geometry ----
    height = (TOP_PAD + BOTTOM_PAD + len(PANELS) * len(cells) * ROW_H
              + (len(PANELS) - 1) * PANEL_GAP)
    print("\n" + "-" * 104)
    print("GEOMETRY")
    print("-" * 104)
    print(f"Width {args.width}in (ACM full text width). Height {height:.2f}in, "
          f"computed from the data:")
    print(f"  {len(PANELS)} panels x {len(cells)} rows x {ROW_H}in per row "
          f"+ {len(PANELS) - 1} gaps x {PANEL_GAP}in + {TOP_PAD}/"
          f"{BOTTOM_PAD}in padding.")
    print("Five panels side by side would leave ~1.2in of plot area each, too "
          "narrow for the n")
    print("and p columns, so the panels are stacked and the columns sit in the "
          "right margin.")

    # ---- draw ----
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(args.width, height))
        gs = fig.add_gridspec(len(PANELS), 1,
                              left=0.088, right=0.795,
                              top=1 - TOP_PAD / height,
                              bottom=BOTTOM_PAD / height,
                              hspace=PANEL_GAP / (len(cells) * ROW_H))
        axes = [fig.add_subplot(gs[i, 0]) for i in range(len(PANELS))]
        for i, ((letter, base, cand), ax) in enumerate(zip(PANELS, axes)):
            draw_panel(ax, letter, base, cand, data, cells, groups, xlo, xhi,
                       is_bottom=(i == len(PANELS) - 1), is_top=(i == 0))
        axes[-1].set_xlabel(
            "Difference in completion rate (percentage points), "
            "candidate $-$ baseline", labelpad=3)

        handles = [
            Line2D([], [], color=SIG, linewidth=1.15, marker="o",
                   markersize=3.4, markerfacecolor=SIG, markeredgecolor=SIG,
                   label=f"permutation $p \\leq$ {ALPHA}"),
            Line2D([], [], color=NULL, linewidth=0.85, marker="o",
                   markersize=3.4, markerfacecolor=SURFACE,
                   markeredgecolor=NULL, label=f"permutation $p >$ {ALPHA}"),
        ]
        if dis:
            handles.append(Line2D(
                [], [], linestyle="none", marker="$\\dagger$", markersize=4.4,
                markeredgecolor=TEXT_PRIMARY, markerfacecolor=TEXT_PRIMARY,
                label=f""
                      f"({len(dis)} rows)"))
        if limited:
            handles.append(Line2D(
                [], [], linestyle="none", marker="$\\ddagger$", markersize=4.4,
                markeredgecolor=TEXT_PRIMARY, markerfacecolor=TEXT_PRIMARY,
                label=f"run counts cannot reach $\\alpha$={ALPHA} "
                      f"({len(limited)} rows)"))
        if untestable:
            handles.append(Line2D(
                [], [], linestyle="none", marker=None,
                label=f"'untestable' = fewer than two runs in an arm "
                      f"({len(untestable)} rows)"))
        fig.legend(handles=handles, loc="lower left",
                   bbox_to_anchor=(0.088, 0.012), bbox_transform=fig.transFigure,
                   ncol=2, handlelength=2.2, handletextpad=0.5, borderpad=0.0,
                   labelspacing=0.32, columnspacing=1.4)

        outdir = Path(args.outdir).expanduser().resolve()
        outdir.mkdir(parents=True, exist_ok=True)
        written = []
        for ext in ("pdf", "png"):
            p = outdir / f"{args.stem}.{ext}"
            fig.savefig(p, dpi=args.dpi, bbox_inches="tight", pad_inches=0.02)
            written.append(p)
        plt.close(fig)

    print()
    for p in written:
        print(f"Wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
