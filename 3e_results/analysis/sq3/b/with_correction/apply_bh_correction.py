"""Benjamini-Hochberg correction of the latency comparisons (reads p-values, recomputes no test).

Definition B (one family per comparison group and metric) is the one reported.
"""
import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)                     # .../3e_results/sq3/b

VS_WLC = os.path.join(SRC, "sq3b_vs_wlc.csv")
VS_WRR = os.path.join(SRC, "sq3b_vs_wrr.csv")
TTFT = os.path.join(SRC, "sq3b_ttft_decode.csv")

# Which SQ3b block each source is, for the family keys.
BLOCK_OF = {VS_WLC: 2, VS_WRR: 3, TTFT: 4}
BLOCK_LABEL = {2: "block 2 (vs weighted_least_connections)",
               3: "block 3 (vs weighted_round_robin)",
               4: "block 4 (ttft/decode contrasts)"}

# analyze_sq3b's block-4 contrast column prefixes and the baseline each means.
TAG_BASE = {"vs_wrr": "weighted_round_robin",
            "vs_wlc": "weighted_least_connections"}

OUT_ALL = "sq3b_bh_corrected.csv"
OUT_FAM = "sq3b_bh_families.csv"
OUT_LOST = "sq3b_bh_lost_significance.csv"

# The comparisons the chapter currently relies on, declared here so each can be
# flagged individually. (base_arm, arm, cell, metric).
QUOTED = [
    ("weighted_round_robin", "state_aware", "c4_uni", "e2el_p50"),
    ("weighted_round_robin", "state_aware", "c12_uni", "e2el_p50"),
    ("weighted_round_robin", "state_aware", "c4_uni", "e2el_p95"),
    ("weighted_round_robin", "state_aware", "c8_uni", "e2el_p95"),
    ("weighted_round_robin", "state_aware", "c8_bim", "e2el_p95"),
    ("weighted_round_robin", "state_aware", "c8_uni", "tpot_p50"),
    ("weighted_round_robin", "state_aware", "c4_bim", "tpot_p50"),
    ("weighted_round_robin", "state_aware", "c12_uni", "tpot_p95"),
    ("weighted_least_connections", "state_aware", "c8_sgpt", "e2el_p50"),
    ("weighted_least_connections", "state_aware", "c8_sgpt", "tpot_p50"),
    ("weighted_least_connections", "state_aware", "c8_sgpt", "tpot_p95"),
]


# ───────────────────────── Benjamini-Hochberg ─────────────────────────────
def bh_adjust(ps):
    """BH adjusted p-values, as R's p.adjust(method="BH") defines them.

    Walk the p-values in DESCENDING order, keeping a running minimum of
    m * p / rank where rank is the ascending rank. The running minimum is what
    enforces monotonicity, and it is why tied raw p-values come out with
    identical adjusted values.
    """
    m = len(ps)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: -ps[i])
    adj = [None] * m
    running = 1.0
    for k, idx in enumerate(order):
        rank = m - k                       # ascending rank of this element
        running = min(running, ps[idx] * m / rank)
        adj[idx] = min(running, 1.0)
    return adj


def bh_reject_set(ps, alpha):
    """The BH rejection set, coded independently of bh_adjust: find the largest
    k with p_(k) <= k*alpha/m and reject the k smallest. Used only to verify
    that {adjusted <= alpha} is the same set."""
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    kmax = 0
    for k, idx in enumerate(order, start=1):
        if ps[idx] <= k * alpha / m:
            kmax = k
    return set(order[:kmax])


def verify_bh(alpha):
    """Three checks, run before any output. Any failure aborts."""
    problems = []
    # 1. Hand-computed vectors.
    cases = [
        ([0.01, 0.02, 0.03, 0.04, 0.05], [0.05] * 5),
        ([0.001, 0.5], [0.002, 0.5]),
        ([0.04, 0.01], [0.04, 0.02]),
    ]
    for ps, want in cases:
        got = bh_adjust(ps)
        if any(abs(a - b) > 1e-12 for a, b in zip(got, want)):
            problems.append(f"hand case {ps}: got {got}, want {want}")
    # 2. Ties get identical adjusted values.
    tied = bh_adjust([0.02, 0.02, 0.9, 0.001])
    if abs(tied[0] - tied[1]) > 1e-15:
        problems.append(f"tied raw p produced different adjusted p: {tied}")
    return problems


def check_properties(ps, adj, alpha, label, problems):
    """adjusted >= raw; adjusted monotone in raw; {adj <= alpha} == step-up
    rejection set."""
    for p, a in zip(ps, adj):
        if a < p - 1e-12:
            problems.append(f"{label}: adjusted {a} < raw {p}")
            break
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    for x, y in zip(order, order[1:]):
        if adj[y] < adj[x] - 1e-12:
            problems.append(f"{label}: adjusted not monotone in raw")
            break
    by_adj = {i for i, a in enumerate(adj) if a <= alpha}
    by_stepup = bh_reject_set(ps, alpha)
    if by_adj != by_stepup:
        problems.append(f"{label}: {{adj<=alpha}} != step-up rejection set "
                        f"({len(by_adj)} vs {len(by_stepup)})")


# ───────────────────────── reading the SQ3b output ────────────────────────
def num(v):
    return None if v in ("", None) else float(v)


def load_records():
    """Every tested comparison in blocks 2, 3 and 4, with its raw p.

    Nothing is recomputed: the p column is taken as analyze_sq3b.py wrote it.
    """
    recs = []
    for path in (VS_WLC, VS_WRR):
        if not os.path.exists(path):
            raise SystemExit(f"FATAL: {path} not found. Run analyze_sq3b.py first.")
        for r in csv.DictReader(open(path)):
            if r["tested"] != "True":
                continue
            recs.append({
                "block": BLOCK_OF[path],
                "source": os.path.basename(path),
                "base_arm": r["base_arm"], "arm": r["arm"], "cell": r["cell"],
                "metric": r["metric"],
                "n": f"{r['n_base']}v{r['n_cand']}",
                "diff": num(r["diff"]),
                "p_raw": float(r["p_permutation"]),
                "enumeration": r["enumeration"],
                "attainable_min_p": num(r["attainable_min_p"]),
                "verdict_uncorrected": r["verdict"],
            })
    if not os.path.exists(TTFT):
        raise SystemExit(f"FATAL: {TTFT} not found. Run analyze_sq3b.py first.")
    ttft_rows = list(csv.DictReader(open(TTFT)))
    # The contrast columns exist only on the state_aware rows; discover the
    # metric names from the header rather than restating them.
    header = ttft_rows[0].keys() if ttft_rows else []
    metrics = []
    for col in header:
        if col.endswith("_vs_wrr_p"):
            metrics.append(col[:-len("_vs_wrr_p")])
    for r in ttft_rows:
        for metric in metrics:
            for tag, base in TAG_BASE.items():
                p = num(r.get(f"{metric}_{tag}_p"))
                if p is None:
                    continue
                recs.append({
                    "block": 4,
                    "source": os.path.basename(TTFT),
                    "base_arm": base, "arm": r["arm"], "cell": r["cell"],
                    "metric": metric,
                    "n": "",
                    "diff": num(r.get(f"{metric}_{tag}_diff")),
                    "p_raw": p,
                    "enumeration": r.get(f"{metric}_{tag}_enumeration", ""),
                    "attainable_min_p":
                        num(r.get(f"{metric}_{tag}_attainable_min_p")),
                    "verdict_uncorrected": r.get(f"{metric}_{tag}_verdict", ""),
                })
    return recs


# ───────────────────────── families ───────────────────────────────────────
def family_key(rec, definition):
    if definition == "A":
        return f"block{rec['block']}"
    if definition == "B":
        return f"block{rec['block']}|{rec['metric']}"
    return "all"


def apply_definition(recs, definition, alpha, problems):
    groups = {}
    for i, rec in enumerate(recs):
        groups.setdefault(family_key(rec, definition), []).append(i)
    sizes = {}
    for key, idxs in groups.items():
        ps = [recs[i]["p_raw"] for i in idxs]
        adj = bh_adjust(ps)
        check_properties(ps, adj, alpha, f"def {definition} family {key}",
                         problems)
        for i, a in zip(idxs, adj):
            recs[i][f"p_adj_{definition}"] = a
            recs[i][f"clears_{definition}"] = (a <= alpha)
            recs[i][f"family_{definition}"] = key
            recs[i][f"family_{definition}_size"] = len(idxs)
        sizes[key] = len(idxs)
    return sizes


# ───────────────────────── output ─────────────────────────────────────────
def write_csv(path, rows, fields=None):
    if not rows:
        return 0, 0
    if fields is None:
        fields = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    return len(rows), len(fields)


def label_of(rec):
    return (f"{rec['arm']} vs {rec['base_arm']} | {rec['cell']} | "
            f"{rec['metric']}")


def main():
    ap = argparse.ArgumentParser(
        description="BH FDR correction of the SQ3b comparisons (read-only).")
    ap.add_argument("--outdir", default=HERE)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    alpha = args.alpha

    print("=" * 108)
    print("SQ3b — Benjamini-Hochberg FDR correction, as a separate sensitivity "
          "analysis.")
    print("No test is recomputed; the p-values written by analyze_sq3b.py are "
          "read and corrected.")
    print("Inputs are opened read-only; analyze_sq3b.py and its CSVs are not "
          "modified.")
    print("=" * 108)

    problems = verify_bh(alpha)
    recs = load_records()
    print(f"\nSources read:")
    for path in (VS_WLC, VS_WRR, TTFT):
        n = sum(1 for r in recs if r["source"] == os.path.basename(path))
        print(f"  {os.path.relpath(path, SRC):<26} block "
              f"{BLOCK_OF[path]}   {n} tested comparisons")
    print(f"  TOTAL {len(recs)} tests carried into the correction "
          f"(alpha = {alpha})")

    sizes = {d: apply_definition(recs, d, alpha, problems)
             for d in ("A", "B", "C")}

    if problems:
        print("\n*** BH IMPLEMENTATION OR PROPERTY CHECK FAILED. Aborting.")
        for p in problems:
            print(f"    {p}")
        return 2
    print("\nBH implementation verified: 3 hand-computed vectors, tie handling,")
    print("adjusted >= raw, adjusted monotone in raw, and {adjusted <= alpha}")
    print("identical to an independently coded step-up rejection set, in every "
          "family.")

    # ---------------- family sizes ----------------
    print("\n" + "#" * 108)
    print("# FAMILY DEFINITIONS AND SIZES")
    print("#" * 108)
    fam_rows = []
    for d, title in (("A", "one family per block"),
                     ("B", "one family per (block, metric)"),
                     ("C", "one family across everything")):
        print(f"\nDefinition {d}: {title} — {len(sizes[d])} famil"
              f"{'y' if len(sizes[d]) == 1 else 'ies'}, "
              f"{sum(sizes[d].values())} tests")
        print(f"  {'family':<34}{'tests':>7}{'cleared raw':>13}"
              f"{'cleared BH':>12}{'lost':>7}")
        for key in sorted(sizes[d]):
            grp = [r for r in recs if r[f"family_{d}"] == key]
            raw = sum(1 for r in grp if r["p_raw"] <= alpha)
            adjn = sum(1 for r in grp if r[f"clears_{d}"])
            print(f"  {key:<34}{len(grp):>7}{raw:>13}{adjn:>12}{raw - adjn:>7}")
            fam_rows.append({"definition": d, "definition_label": title,
                             "family": key, "n_tests": len(grp),
                             "n_cleared_raw": raw, "n_cleared_bh": adjn,
                             "n_lost": raw - adjn, "alpha": alpha})

    # ---------------- every comparison ----------------
    print("\n" + "#" * 108)
    print("# EVERY COMPARISON: raw p and adjusted p under each definition")
    print("#" * 108)
    print(f"{'blk':<4}{'arm':<17}{'base':<28}{'cell':<9}{'metric':<17}"
          f"{'raw p':>9}{'adj A':>9}{'adj B':>9}{'adj C':>9}"
          f"{'raw':>5}{'A':>3}{'B':>3}{'C':>3}")
    y = lambda b: ("Y" if b else ".")
    for r in sorted(recs, key=lambda x: (x["block"], x["arm"], x["metric"],
                                         x["cell"])):
        print(f"{r['block']:<4}{r['arm']:<17}{r['base_arm']:<28}{r['cell']:<9}"
              f"{r['metric']:<17}"
              f"{r['p_raw']:>9.4f}{r['p_adj_A']:>9.4f}{r['p_adj_B']:>9.4f}"
              f"{r['p_adj_C']:>9.4f}"
              f"{y(r['p_raw'] <= alpha):>5}{y(r['clears_A']):>3}"
              f"{y(r['clears_B']):>3}{y(r['clears_C']):>3}")

    # ---------------- what changed ----------------
    print("\n" + "#" * 108)
    print("# WHAT CHANGED, PER FAMILY DEFINITION")
    print("#" * 108)
    n_raw = sum(1 for r in recs if r["p_raw"] <= alpha)
    lost_rows = []
    for d in ("A", "B", "C"):
        lost = [r for r in recs if r["p_raw"] <= alpha and not r[f"clears_{d}"]]
        gained = [r for r in recs
                  if r["p_raw"] > alpha and r[f"clears_{d}"]]
        print(f"\nDefinition {d}: cleared raw {n_raw}, cleared after BH "
              f"{n_raw - len(lost)}, lost {len(lost)}")
        if gained:
            print(f"  *** {len(gained)} comparison(s) GAINED significance, "
                  f"which BH cannot do — investigate.")
        if not lost:
            print("  Nothing lost significance under this definition.")
        else:
            print(f"  {'blk':<4}{'arm':<17}{'base':<28}{'cell':<9}"
                  f"{'metric':<17}{'raw p':>9}{'adj p':>9}")
            for r in sorted(lost, key=lambda x: (x["block"], x["arm"],
                                                 x["metric"], x["cell"])):
                print(f"  {r['block']:<4}{r['arm']:<17}{r['base_arm']:<28}"
                      f"{r['cell']:<9}{r['metric']:<17}"
                      f"{r['p_raw']:>9.4f}{r[f'p_adj_{d}']:>9.4f}")
                lost_rows.append({
                    "definition": d, "block": r["block"], "arm": r["arm"],
                    "base_arm": r["base_arm"], "cell": r["cell"],
                    "metric": r["metric"], "n": r["n"],
                    "p_raw": r["p_raw"], "p_adj": r[f"p_adj_{d}"],
                    "family": r[f"family_{d}"],
                    "family_size": r[f"family_{d}_size"],
                    "quoted_in_prose": (r["base_arm"], r["arm"], r["cell"],
                                        r["metric"]) in set(QUOTED),
                })

    # ---------------- the quoted comparisons ----------------
    print("\n" + "#" * 108)
    print("# THE COMPARISONS THE PROSE RELIES ON")
    print("#" * 108)
    index = {(r["base_arm"], r["arm"], r["cell"], r["metric"]): r for r in recs}
    missing = [q for q in QUOTED if q not in index]
    if missing:
        print("*** These declared comparisons are not in the SQ3b output at all:")
        for q in missing:
            print(f"    {q}")
    print(f"{'arm':<13}{'base':<28}{'cell':<9}{'metric':<11}"
          f"{'raw p':>9}{'adj A':>9}{'adj B':>9}{'adj C':>9}"
          f"{'raw':>6}{'A':>7}{'B':>7}{'C':>7}")
    quoted_rows = []
    for base, arm, cell, metric in QUOTED:
        r = index.get((base, arm, cell, metric))
        if r is None:
            continue
        s = lambda b: ("keep" if b else "LOST")
        print(f"{arm:<13}{base:<28}{cell:<9}{metric:<11}"
              f"{r['p_raw']:>9.4f}{r['p_adj_A']:>9.4f}{r['p_adj_B']:>9.4f}"
              f"{r['p_adj_C']:>9.4f}"
              f"{y(r['p_raw'] <= alpha):>6}{s(r['clears_A']):>7}"
              f"{s(r['clears_B']):>7}{s(r['clears_C']):>7}")
        quoted_rows.append(r)
    for d in ("A", "B", "C"):
        lost_q = [r for r in quoted_rows if not r[f"clears_{d}"]]
        print(f"\nDefinition {d}: {len(quoted_rows) - len(lost_q)} of "
              f"{len(quoted_rows)} quoted comparisons survive, {len(lost_q)} "
              f"lost.")
        for r in lost_q:
            print(f"    LOST  {label_of(r)}  raw p={r['p_raw']:.4f} -> "
                  f"adj={r[f'p_adj_{d}']:.4f} "
                  f"(family {r[f'family_{d}']}, {r[f'family_{d}_size']} tests)")

    # ---------------- files ----------------
    out_rows = []
    for r in sorted(recs, key=lambda x: (x["block"], x["arm"], x["metric"],
                                         x["cell"])):
        row = {
            "block": r["block"], "source": r["source"], "arm": r["arm"],
            "base_arm": r["base_arm"], "cell": r["cell"],
            "metric": r["metric"], "n": r["n"], "diff": r["diff"],
            "enumeration": r["enumeration"],
            "attainable_min_p": r["attainable_min_p"],
            "verdict_uncorrected": r["verdict_uncorrected"],
            "p_raw": r["p_raw"],
            "clears_alpha_raw": r["p_raw"] <= alpha,
            "quoted_in_prose": (r["base_arm"], r["arm"], r["cell"],
                                r["metric"]) in set(QUOTED),
        }
        for d in ("A", "B", "C"):
            row[f"family_{d}"] = r[f"family_{d}"]
            row[f"family_{d}_size"] = r[f"family_{d}_size"]
            row[f"p_adj_{d}"] = r[f"p_adj_{d}"]
            row[f"clears_alpha_{d}"] = r[f"clears_{d}"]
            row[f"lost_significance_{d}"] = (r["p_raw"] <= alpha
                                             and not r[f"clears_{d}"])
        out_rows.append(row)

    written = []
    for name, rows in ((OUT_ALL, out_rows), (OUT_FAM, fam_rows),
                       (OUT_LOST, lost_rows)):
        nr, nc = write_csv(os.path.join(outdir, name), rows)
        written.append((name, nr, nc))

    write_readme(outdir, written, recs, quoted_rows, sizes, alpha, n_raw)
    print(f"\nWritten to {outdir}:")
    for name, nr, nc in written:
        print(f"  {name}  ({nr} rows, {nc} cols)")
    print("  README.md")
    return 0


def write_readme(outdir, written, recs, quoted_rows, sizes, alpha, n_raw):
    L = [
        "# SQ3b with a Benjamini-Hochberg FDR correction",
        "",
        "Generated by `apply_bh_correction.py`. **Definition B is the reported "
        "correction; A and C are sensitivity alternatives.** `analyze_sq3b.py` and its CSVs are inputs, opened "
        "read-only, and are not modified. The uncorrected p-values remain the "
        "ones that script reports.",
        "",
        "**No test is recomputed.** The p-values already written into blocks 2, "
        "3 and 4 are read and adjusted. This script does not import "
        "`analyze_3e`, does not load a run folder, and runs no permutation "
        "test.",
        "",
        "## Files",
        "",
        "| file | rows | cols | contents |",
        "|---|---:|---:|---|",
        f"| `{written[0][0]}` | {written[0][1]} | {written[0][2]} | every "
        "comparison: raw p, adjusted p under all three family definitions, and "
        "whether it clears alpha before and after each |",
        f"| `{written[1][0]}` | {written[1][1]} | {written[1][2]} | one row per "
        "family: its size, how many cleared alpha raw, how many after BH |",
        f"| `{written[2][0]}` | {written[2][1]} | {written[2][2]} | one row per "
        "comparison that lost significance, per definition, flagged if it is "
        "quoted in the prose |",
        "",
        "## Sources",
        "",
        "| block | file | tested comparisons |",
        "|---|---|---:|",
    ]
    for b in (2, 3, 4):
        n = sum(1 for r in recs if r["block"] == b)
        src = next(r["source"] for r in recs if r["block"] == b)
        L.append(f"| {b} | `../{src}` | {n} |")
    L += [
        f"| **total** | | **{len(recs)}** |",
        "",
        "Block 2 and block 3 contribute the rows with `tested == True`; the "
        "untestable rows carry no p-value and enter no family. Block 4 "
        "contributes the `state_aware` contrast columns "
        "`<metric>_vs_wrr_p` and `<metric>_vs_wlc_p`, whose metric names are "
        "read from the CSV header rather than restated here.",
        "",
        "## The three family definitions",
        "",
        "| definition | family key | families | tests |",
        "|---|---|---:|---:|",
    ]
    for d, title in (("A", "one family per block"),
                     ("B", "one family per (block, metric)"),
                     ("C", "one family across everything")):
        L.append(f"| {d} | {title} | {len(sizes[d])} | "
                 f"{sum(sizes[d].values())} |")
    L += [
        "",
        "Definition B keys on `(block, metric)` **only**, as specified. A "
        "block-2 metric family therefore spans all four candidate arms, and a "
        "block-4 metric family spans both baselines. Family sizes are in "
        f"`{written[1][0]}` so this is visible rather than assumed.",
        "",
        "## Method",
        "",
        "BH adjusted p-values follow the step-up definition used by R's "
        "`p.adjust(method = \"BH\")`: sort descending, take the running minimum "
        "of `m * p / rank`, cap at 1. Tied raw p-values receive identical "
        "adjusted values, which matters here because the permutation test is "
        "discrete and produces many ties.",
        "",
        "The implementation is verified before any output is produced, and a "
        "failure aborts the run:",
        "",
        "- three hand-computed vectors reproduce exactly;",
        "- tied raw p-values come out identical;",
        "- adjusted >= raw, and adjusted is monotone in raw, in every family;",
        "- `{adjusted <= alpha}` equals an independently coded step-up "
        "rejection set (largest k with p_(k) <= k*alpha/m), in every family.",
        "",
        f"alpha = {alpha}.",
        "",
        "## Headline counts",
        "",
        "| definition | cleared raw | cleared after BH | lost |",
        "|---|---:|---:|---:|",
    ]
    for d in ("A", "B", "C"):
        lost = sum(1 for r in recs if r["p_raw"] <= alpha and not r[f"clears_{d}"])
        L.append(f"| {d} | {n_raw} | {n_raw - lost} | {lost} |")
    L += [
        "",
        "## The comparisons the prose relies on",
        "",
        "| arm | baseline | cell | metric | raw p | adj A | adj B | adj C | A | B | C |",
        "|---|---|---|---|---:|---:|---:|---:|---|---|---|",
    ]
    for r in quoted_rows:
        s = lambda b: ("keep" if b else "**LOST**")
        L.append(f"| {r['arm']} | {r['base_arm']} | {r['cell']} | "
                 f"{r['metric']} | {r['p_raw']:.4f} | {r['p_adj_A']:.4f} | "
                 f"{r['p_adj_B']:.4f} | {r['p_adj_C']:.4f} | "
                 f"{s(r['clears_A'])} | {s(r['clears_B'])} | "
                 f"{s(r['clears_C'])} |")
    L.append("")
    for d in ("A", "B", "C"):
        lost_q = [r for r in quoted_rows if not r[f"clears_{d}"]]
        L.append(f"- Definition {d}: {len(quoted_rows) - len(lost_q)} of "
                 f"{len(quoted_rows)} survive"
                 + ("." if not lost_q else
                    "; lost: " + ", ".join(
                        f"`{r['arm']} vs {r['base_arm']} {r['cell']} "
                        f"{r['metric']}`" for r in lost_q) + "."))
    L.append("")
    open(os.path.join(outdir, "README.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
