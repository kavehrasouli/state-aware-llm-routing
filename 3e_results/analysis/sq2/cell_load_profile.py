import argparse
import csv
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)  # combined analysis helpers
for p in (HERE, PARENT):
    if p not in sys.path:
        sys.path.insert(0, p)

import analyze_divergence as AD                  # noqa: E402
from analyze_sq2 import scan                      # noqa: E402
from analyze_sq1 import resolve_num_warmup        # noqa: E402

WL = {"uniform": "uni", "bimodal": "bim", "sharegpt": "sgpt"}
ORDER = [AD.col_label(w, c) for w, c in AD.CELL_COLS]


def context_budgets(config_names):
    """u is resident_tokens / context_budget (poller.rs), and context_budget is
    NOT carried in the logged state_vector, so resident tokens are recoverable
    only from the router config. Read ONLY the file(s) frozen_config actually
    names -- the repo holds other configs (e.g. router_diagnose_pi2.toml at
    2048) that these runs never used, and scanning them all would make the
    budget look non-unique when it is not."""
    repo = os.path.dirname(os.path.dirname(PARENT))
    vals = set()
    for root, _dirs, files in os.walk(os.path.join(repo, "routers")):
        for fn in files:
            if fn in config_names:
                for line in open(os.path.join(root, fn)):
                    line = line.strip()
                    if line.startswith("context_budget"):
                        try:
                            vals.add(float(line.split("=", 1)[1].strip()))
                        except ValueError:
                            pass
    return vals


def q(vals, p):
    """Percentile via analyze_divergence.pct so it matches every other artefact."""
    return AD.pct(sorted(vals), p) if vals else None


def med(vals):
    return AD.median(vals) if vals else None


def f(v, nd=1):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "n/a"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=HERE)
    ap.add_argument("--stem", default="cell_load_profile")
    args = ap.parse_args()
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    print("=" * 104)
    print("Per-cell load profile — clean runs, warm-up excluded")
    print("=" * 104)

    acc = defaultdict(lambda: {
        "runs": 0, "decisions": 0, "duration_s": [], "dec_per_s": [],
        "prompt": [], "output": [], "e2el": [], "ttft": [], "decode": [],
        "resident": {n: [] for n in AD.NODES},
        "inflight": {n: [] for n in AD.NODES},
        "any_idle": 0, "complete_sv": 0,
        "conc": set(), "rate": set(), "nprompts": set(), "seed": set(),
        "ctx": set(), "dataset": set(), "cfg": set(),
    })

    for r in AD.load_all_runs():
        if not r.get("usable") or r.get("verdict") != "clean":
            continue
        d = r["dir"]
        meta = json.loads(open(os.path.join(d, "run_meta.json")).read())
        fc = meta.get("frozen_config") or {}
        cell = "c%s_%s" % (meta.get("concurrency"), WL[meta.get("workload")])
        a = acc[cell]
        a["runs"] += 1
        a["conc"].add(meta.get("concurrency"))
        a["rate"].add(fc.get("request_rate"))
        a["nprompts"].add(fc.get("num_prompts"))
        a["seed"].add(fc.get("seed"))
        if fc.get("router_config_file"):
            a["cfg"].add(fc["router_config_file"])

        # ---- wall clock, from run_meta ----
        st, en = meta.get("start_ts"), meta.get("end_ts")
        dur = None
        if isinstance(st, (int, float)) and isinstance(en, (int, float)):
            dur = en - st
        elif st and en:
            try:
                from datetime import datetime
                dur = (datetime.fromisoformat(str(en))
                       - datetime.fromisoformat(str(st))).total_seconds()
            except Exception:
                dur = None

        # ---- router side: decisions and engine state ----
        nw, _ = resolve_num_warmup(meta, d)
        s = scan(meta, os.path.join(d, meta["router_log_file"]), nw, "measured")
        if s is None:
            continue
        n_dec = s["n_records"]
        a["decisions"] += n_dec
        if dur and dur > 0:
            a["duration_s"].append(dur)
            a["dec_per_s"].append(n_dec / dur)

        # resident tokens / in-flight per replica, re-read because scan keeps
        # only b_metrics distributions, not u or per-decision idleness
        with open(os.path.join(d, meta["router_log_file"])) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("type") != "request":
                    continue
                rid = rec.get("request_id")
                if not (isinstance(rid, int) and rid >= nw):
                    continue
                sv = rec.get("state_vector")
                if not sv or not isinstance(sv, list):
                    continue
                by = {e.get("replica"): e for e in sv}
                if not all(n in by for n in AD.NODES):
                    continue
                a["complete_sv"] += 1
                idle = False
                for n in AD.NODES:
                    e = by[n]
                    u, cb = e.get("u"), e.get("context_budget")
                    if isinstance(u, (int, float)):
                        # context_budget is not in the state_vector; resident
                        # tokens are recovered from u only if it is.
                        if isinstance(cb, (int, float)) and cb:
                            a["resident"][n].append(u * cb)
                            a["ctx"].add(cb)
                        else:
                            a["resident"][n].append(u)   # ratio, flagged below
                    b = e.get("b_metrics")
                    if isinstance(b, (int, float)):
                        a["inflight"][n].append(b)
                        if b == 0:
                            idle = True
                if idle:
                    a["any_idle"] += 1

        # ---- client side: lengths and latencies ----
        bf = meta.get("bench_result_file")
        if bf and os.path.exists(os.path.join(d, bf)):
            b = json.loads(open(os.path.join(d, bf)).read())
            a["dataset"].add((b.get("dataset_name") or b.get("dataset")
                              or meta.get("workload")))
            for rec in b.get("raw", []):
                if not rec.get("success"):
                    continue
                for key, bucket in (("actual_prompt_len", "prompt"),
                                    ("generated_output_tokens", "output"),
                                    ("e2el_ms", "e2el"), ("ttft_ms", "ttft")):
                    v = rec.get(key)
                    if isinstance(v, (int, float)) and not (key == "actual_prompt_len"
                                                            and v < 0):
                        a[bucket].append(v)
                e, t = rec.get("e2el_ms"), rec.get("ttft_ms")
                if isinstance(e, (int, float)) and isinstance(t, (int, float)):
                    a["decode"].append(e - t)

    cells = [c for c in ORDER if c in acc]
    cfg_names = {c for x in cells for c in acc[x]["cfg"]}
    budgets = context_budgets(cfg_names)
    if len(budgets) == 1:
        cb = budgets.pop()
        for c in cells:
            for n in AD.NODES:
                acc[c]["resident"][n] = [v * cb for v in acc[c]["resident"][n]]
        resident_is_ratio = False
        print(f"\ncontext_budget = {cb:.0f}, unique across {sorted(cfg_names)} "
              f"(the config frozen_config names); resident tokens = u x context_budget")
    else:
        resident_is_ratio = True
        print(f"\ncontext_budget not unique ({sorted(budgets)}); resident "
              f"reported as the raw u ratio")

    # ---------------- 1. throughput ----------------
    print("\n1. WALL CLOCK AND DECISION RATE")
    print(f"{'cell':<9}{'runs':>6}{'decisions':>11}{'dur/run s':>12}"
          f"{'dec/s':>9}{'dec/run':>9}")
    rows1 = []
    for c in cells:
        a = acc[c]
        dur = med(a["duration_s"])
        dps = med(a["dec_per_s"])
        rows1.append([c, a["runs"], a["decisions"], f(dur), f(dps, 3),
                      f(a["decisions"] / a["runs"] if a["runs"] else None)])
        print(f"{c:<9}{a['runs']:>6}{a['decisions']:>11}{f(dur):>12}"
              f"{f(dps,3):>9}{f(a['decisions']/a['runs'] if a['runs'] else None):>9}")

    # ---------------- 2. lengths ----------------
    print("\n2. REQUEST SIZE (tokens, pooled over requests)")
    print(f"{'cell':<9}{'n req':>8}{'prompt p50':>12}{'prompt p90':>12}"
          f"{'output p50':>12}{'output p90':>12}")
    rows2 = []
    for c in cells:
        a = acc[c]
        row = [c, len(a["prompt"]), f(med(a["prompt"])), f(q(a["prompt"], 90)),
               f(med(a["output"])), f(q(a["output"], 90))]
        rows2.append(row)
        print(f"{row[0]:<9}{row[1]:>8}{row[2]:>12}{row[3]:>12}{row[4]:>12}{row[5]:>12}")

    # ---------------- 3. latency ----------------
    print("\n3. LATENCY (ms, pooled over requests — descriptive, not the run-level "
          "unit analyze_3e tests on)")
    print(f"{'cell':<9}{'e2el p50':>11}{'e2el p90':>11}{'ttft p50':>11}"
          f"{'ttft p90':>11}{'decode p50':>12}{'decode p90':>12}")
    rows3 = []
    for c in cells:
        a = acc[c]
        row = [c, f(med(a["e2el"]), 0), f(q(a["e2el"], 90), 0),
               f(med(a["ttft"]), 0), f(q(a["ttft"], 90), 0),
               f(med(a["decode"]), 0), f(q(a["decode"], 90), 0)]
        rows3.append(row)
        print(f"{row[0]:<9}{row[1]:>11}{row[2]:>11}{row[3]:>11}{row[4]:>11}"
              f"{row[5]:>12}{row[6]:>12}")

    # ---------------- 4. engine-side load ----------------
    unit = "u ratio" if resident_is_ratio else "tokens"
    print(f"\n4. ENGINE-SIDE LOAD ACROSS DECISIONS (median; resident in {unit})")
    print(f"{'cell':<9}" + "".join(f"{n.replace('pi-node-','res n'):>10}"
                                   for n in AD.NODES)
          + "".join(f"{n.replace('pi-node-','B n'):>8}" for n in AD.NODES))
    rows4 = []
    for c in cells:
        a = acc[c]
        res = [med(a["resident"][n]) for n in AD.NODES]
        inf = [med(a["inflight"][n]) for n in AD.NODES]
        rows4.append([c] + [f(v, 1) for v in res] + [f(v, 2) for v in inf])
        print(f"{c:<9}" + "".join(f"{f(v,1):>10}" for v in res)
              + "".join(f"{f(v,2):>8}" for v in inf))

    # ---------------- 5. idleness ----------------
    print("\n5. SHARE OF DECISIONS WITH AT LEAST ONE REPLICA IDLE (b_metrics == 0)")
    print(f"{'cell':<9}{'decisions':>11}{'any idle':>10}{'share':>9}")
    rows5 = []
    for c in cells:
        a = acc[c]
        n = a["complete_sv"]
        share = (a["any_idle"] / n) if n else None
        rows5.append([c, n, a["any_idle"], f(share, 3)])
        print(f"{c:<9}{n:>11}{a['any_idle']:>10}{f(share,3):>9}")

    # ---------------- config ----------------
    print("\n6. RUN CONFIGURATION AS RECORDED (not assumed)")
    print(f"{'cell':<9}{'conc':>6}{'rate/s':>8}{'prompts':>9}{'seed':>7}  dataset")
    rows6 = []
    for c in cells:
        a = acc[c]
        row = [c, "|".join(str(x) for x in sorted(a["conc"])),
               "|".join(str(x) for x in sorted(a["rate"])),
               "|".join(str(x) for x in sorted(a["nprompts"])),
               "|".join(str(x) for x in sorted(a["seed"])),
               "|".join(str(x) for x in sorted(a["dataset"]))]
        rows6.append(row)
        print(f"{row[0]:<9}{row[1]:>6}{row[2]:>8}{row[3]:>9}{row[4]:>7}  {row[5]}")

    md = ["# Per-cell load profile (clean runs, warm-up excluded)", "",
          "Generated by `3e_results/sq2/cell_load_profile.py`. Descriptive only; "
          "nothing here is tested. Latency and length figures are pooled over "
          "requests, not the run-level unit `analyze_3e.py` uses for its "
          "comparisons.", "",
          "## 1. Wall clock and decision rate", "",
          _t(["cell", "runs", "decisions", "duration/run s", "decisions/s",
              "decisions/run"], rows1),
          "", "## 2. Request size (tokens)", "",
          _t(["cell", "n requests", "prompt p50", "prompt p90", "output p50",
              "output p90"], rows2),
          "", "## 3. Latency (ms, pooled over requests)", "",
          _t(["cell", "e2el p50", "e2el p90", "ttft p50", "ttft p90",
              "decode p50", "decode p90"], rows3),
          "", f"## 4. Engine-side load across decisions (median, resident in {unit})", "",
          _t(["cell"] + [f"resident {n}" for n in AD.NODES]
             + [f"in-flight {n}" for n in AD.NODES], rows4),
          "", "## 5. Decisions with at least one idle replica", "",
          _t(["cell", "decisions", "any idle", "share"], rows5),
          "", "## 6. Run configuration as recorded", "",
          _t(["cell", "concurrency", "rate/s", "num_prompts", "seed", "dataset"],
             rows6), ""]
    md_path = os.path.join(outdir, args.stem + ".md")
    open(md_path, "w").write("\n".join(md) + "\n")
    csv_path = os.path.join(outdir, args.stem + ".csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["section", "cell", "values"])
        for name, rows in (("throughput", rows1), ("sizes", rows2),
                           ("latency", rows3), ("engine_load", rows4),
                           ("idleness", rows5), ("config", rows6)):
            for row in rows:
                w.writerow([name, row[0]] + list(row[1:]))
    print(f"\nWrote {md_path}")
    print(f"Wrote {csv_path}")
    return 0


def _t(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    sys.exit(main())
