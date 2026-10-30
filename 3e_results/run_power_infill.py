"""Adds runs to cells with too few repetitions for the permutation test to reach p < 0.05.

Without flags it only prints the plan; --run executes it on the cluster.
"""

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_engine():
    """Import run_3e.py as a library (it is __main__-guarded, so this is safe)."""
    spec = importlib.util.spec_from_file_location("run_3e", HERE / "run_3e.py")
    mod = importlib.util.module_from_spec(spec)
    saved = sys.argv
    sys.argv = ["run_3e"]           # the driver parses argv only inside main()
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = saved
    return mod


eng = load_engine()

# ───────────────────────── What to fill ─────────────────────────
GROUP = "power_infill_v1"
GROUP_DIR = HERE / GROUP
MANIFEST = GROUP_DIR / "manifest.json"
GROUP_README = GROUP_DIR / "README.md"

ALL_CELLS = [("uniform", 4), ("uniform", 8), ("uniform", 12),
             ("bimodal", 4), ("bimodal", 8), ("bimodal", 12), ("sharegpt", 8)]

# Arm groups, by what they support in the write-up.
HEADLINE = ["weighted_least_connections",   # baseline every comparison is against
            "state_aware",                  # headline treatment
            "state_aware_v2", "cost_predict_v3"]
LADDER = ["round_robin", "weighted_round_robin", "least_connections"]
CONC_ARM = ["wlc_conc"]                     # the re-weighted WLC control
RANDOM = ["random"]

SCOPES = {
    "A": (HEADLINE, "headline only: WLC vs state_aware / v2 / v3"),
    "B": (HEADLINE + LADDER, "A + ladder baselines (rr->wrr, lc->wlc)"),
    "C": (HEADLINE + LADDER + CONC_ARM, "B + wlc_conc in every cell"),
    "D": (HEADLINE + LADDER + CONC_ARM + RANDOM, "C + random (everything achievable)"),
}
DEFAULT_SCOPE = "C"
DEFAULT_TARGET_CLEAN = 6

# Cells with no clean run after three or more attempts: the policy fails the validity filter there,
# so more runs would not help. Reported as a result and skipped.
STRUCTURALLY_FAILING = {
    ("random", "uniform", 12), ("random", "bimodal", 4),
    ("random", "bimodal", 8), ("random", "bimodal", 12),
    ("round_robin", "bimodal", 8), ("round_robin", "bimodal", 12),
    ("least_connections", "bimodal", 12),
    ("state_aware_v2", "bimodal", 12),
}

# Retry limits: cells in STRUCTURALLY_FAILING are never queued, and each cell gets a bounded
# number of new attempts per session (then reported as suspect-limited).
ATTEMPT_CAP_PER_CELL = 6            # historical suspects tolerated before a 0-clean cell is dead


def attempt_budget(need):
    """New attempts allowed this session for a cell needing `need` clean runs.
    need*2+2 tolerates roughly a 60% suspect rate before giving up."""
    return need * 2 + 2


# (arm, workload, conc) -> attempts made in THIS session. Populated by do_run;
# empty in plan mode, so planning is unaffected.
SESSION_ATTEMPTS = {}
# Cells abandoned this session because their attempt budget ran out.
SUSPECT_LIMITED = {}


# ───────────────────────── Deficit accounting ─────────────────────────
def scan_cell(arm, conc, workload):
    """Return (clean, total, max_rep) for one arm x cell from run_meta.json."""
    d = eng.OUT_ROOT / arm / eng.block_name(conc, workload)
    clean = total = 0
    max_rep = 0
    if not d.is_dir():
        return 0, 0, 0
    for rep_dir in d.iterdir():
        mp = rep_dir / "run_meta.json"
        if not mp.is_file():
            continue
        try:
            meta = json.loads(mp.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        total += 1
        if meta.get("verdict") == "clean":
            clean += 1
        try:
            max_rep = max(max_rep, int(str(rep_dir.name).replace("rep", "")))
        except ValueError:
            pass
    return clean, total, max_rep


def attainable_min_p(n_a, n_b):
    """Smallest p the median-statistic permutation test can produce at these
    rep counts (perfect separation). Mirrors analyze_3e.perm_test exactly."""
    import itertools

    def med(xs):
        xs = sorted(xs)
        n = len(xs)
        return float(xs[n // 2]) if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2.0

    a = [float(i) for i in range(n_a)]
    b = [1e6 + i for i in range(n_b)]
    pool, na, n = a + b, len(a), len(a) + len(b)
    obs = med(b) - med(a)
    total = math.comb(n, na)
    if total > 200_000:                      # matches PERM_EXACT_LIMIT
        return 1.0 / 20_001
    hits = 0
    for idx in itertools.combinations(range(n), na):
        sel = set(idx)
        x = [pool[i] for i in idx]
        y = [pool[i] for i in range(n) if i not in sel]
        if abs(med(y) - med(x)) >= abs(obs) - 1e-12:
            hits += 1
    return hits / total


def build_plan(target_clean, scope=DEFAULT_SCOPE):
    """Per arm x cell: what exists, what is still needed, and why. Returns
    (rows, deficit_total). Structurally-failing combinations are skipped."""
    arms, _ = SCOPES[scope]
    rows = []
    deficit = 0
    for workload, conc in ALL_CELLS:
        for arm in arms:
            if (arm, workload, conc) in STRUCTURALLY_FAILING:
                rows.append({"arm": arm, "conc": conc, "workload": workload,
                             "clean": 0, "total": 0, "max_rep": 0,
                             "need": 0, "capped": False, "dead": True})
                continue
            clean, total, max_rep = scan_cell(arm, conc, workload)
            need = max(0, target_clean - clean)
            historical_suspects = max(0, total - clean)
            key = (arm, workload, conc)

            # Brake 1: never queue a cell that has produced nothing clean.
            capped = historical_suspects >= ATTEMPT_CAP_PER_CELL and clean == 0
            # Brake 2: within this session, stop once the attempt budget for the
            # cell is spent, even if it still has clean runs outstanding.
            spent = SESSION_ATTEMPTS.get(key, 0)
            budget = SESSION_ATTEMPTS.get(("budget",) + key)
            if budget is None and need > 0:
                budget = attempt_budget(need)          # fixed on first sight
                SESSION_ATTEMPTS[("budget",) + key] = budget
            exhausted = need > 0 and budget is not None and spent >= budget
            if exhausted:
                SUSPECT_LIMITED[key] = {"clean": clean, "need": need,
                                        "attempts_this_session": spent, "budget": budget}

            blocked = capped or exhausted
            rows.append({
                "arm": arm, "conc": conc, "workload": workload,
                "clean": clean, "total": total, "max_rep": max_rep,
                "need": 0 if blocked else need, "capped": capped,
                "exhausted": exhausted, "dead": False,
            })
            if not blocked:
                deficit += need
    return rows, deficit


def make_cell(arm, conc, workload, rep):
    """Build the cell dict run_3e.run_cell expects. An 'arm' such as wlc_conc is
    a (policy, config) variant, so policy and arm differ; for a plain policy they
    are the same and no arm key is set (matching the original sweep's records)."""
    spec = eng.arm_spec(arm)
    cell = {"policy": spec["policy"], "conc": conc, "workload": workload,
            "rep": rep, "group": GROUP}
    if spec["policy"] != arm:
        cell["arm"] = arm
    return cell


def pending_cells(target_clean, wave_idx=0, scope=DEFAULT_SCOPE):
    """One run per still-deficient arm x cell (a 'wave'). Waves repeat until the
    target is met, so each arm in a cell gains a rep before any arm gains its
    second: the baseline and the treatments it is compared against are collected
    close together in time, which matters more than absolute drift because the
    comparison is between them. Arm order is rotated by both cell and wave so no
    arm sits systematically early or late in the session."""
    rows, _ = build_plan(target_clean, scope)
    by_cell = {}
    for r in rows:
        by_cell.setdefault((r["workload"], r["conc"]), []).append(r)

    wave = []
    for ci, (workload, conc) in enumerate(ALL_CELLS):
        arms_here = [r for r in by_cell.get((workload, conc), []) if r["need"] > 0]
        if not arms_here:
            continue
        k = (ci + wave_idx) % max(1, len(arms_here))   # counterbalance cell x wave
        arms_here = arms_here[k:] + arms_here[:k]
        for r in arms_here:
            cell = make_cell(r["arm"], conc, workload, r["max_rep"] + 1)
            cell["_need"], cell["_clean"], cell["_arm"] = r["need"], r["clean"], r["arm"]
            wave.append(cell)
    return wave


# ───────────────────────── Reporting ─────────────────────────
def print_plan(target_clean, scope=DEFAULT_SCOPE):
    rows, deficit = build_plan(target_clean, scope)
    arms, desc = SCOPES[scope]
    print("=" * 96)
    print(f"§3e POWER INFILL ({GROUP}) — scope {scope}: {desc}")
    print(f"target {target_clean} clean reps per arm, across all {len(ALL_CELLS)} cells")
    print("=" * 96)

    print("\nResolution of the median-statistic permutation test:")
    for pair in [(3, 3), (3, 5), (5, 5), (6, 5), (6, 6)]:
        p = attainable_min_p(*pair)
        print(f"    {pair[0]} vs {pair[1]:<2d} -> smallest attainable p = {p:.3f}   "
              f"{'OK' if p <= 0.05 else 'CANNOT REACH 0.05'}")

    print("\nAll scopes:")
    for key in ("A", "B", "C", "D"):
        _, d = build_plan(target_clean, key)
        star = " <-- selected" if key == scope else ""
        print(f"    {key}: {SCOPES[key][1]:52s} {d:4d} runs (~{d*8/60:.1f} h){star}")

    print(f"\n{'cell':12s} {'arm':28s} {'clean':>5s} {'need':>5s}  note")
    print("-" * 96)
    for r in rows:
        if r.get("dead"):
            note = "SKIPPED: 0 clean in >=3 attempts — policy fails here (a result, not a gap)"
            print(f"{eng.block_name(r['conc'], r['workload']):12s} {r['arm']:28s} "
                  f"{'-':>5s} {'-':>5s}  {note}")
            continue
        if r["need"] == 0 and not r["capped"]:
            continue
        note = "AT CAP — investigate" if r["capped"] else \
               f"-> reps {r['max_rep']+1}..{r['max_rep']+r['need']}"
        print(f"{eng.block_name(r['conc'], r['workload']):12s} {r['arm']:28s} "
              f"{r['clean']:5d} {r['need']:5d}  {note}")

    print(f"\nClean runs still needed (scope {scope}): {deficit}")
    print(f"Each run = 3 warm-up + 100 measured requests at rate 2, plus health/thermal "
          f"gate and {eng.COOLDOWN_S}s cooldown (~6-14 min observed).")
    print(f"Estimated wall time: {deficit*6//60}-{deficit*14//60} h if every run comes back "
          f"clean; suspect runs do not count toward the target, so budget headroom.")
    print(f"\nAfter completion: {target_clean} vs {target_clean} -> "
          f"p_min = {attainable_min_p(target_clean, target_clean):.3f} in every filled cell.")
    print(f"\nProvenance: extension_group=\"{GROUP}\", manifest at "
          f"{MANIFEST.relative_to(HERE.parent)}, rationale at "
          f"{GROUP_README.relative_to(HERE.parent)}.")
    print(f"\nPLAN ONLY — nothing was run. Execute with:  "
          f"python3 run_power_infill.py --run --scope {scope}")
    return deficit


def write_group_docs(target_clean, rows, scope=DEFAULT_SCOPE):
    GROUP_DIR.mkdir(parents=True, exist_ok=True)
    pairs = [(3, 3), (3, 5), (5, 5), (target_clean, 5), (target_clean, target_clean)]
    res = "\n".join(
        f"| {a} vs {b} | {attainable_min_p(a, b):.3f} | "
        f"{'yes' if attainable_min_p(a, b) <= 0.05 else 'NO'} |" for a, b in pairs)
    tbl = "\n".join(
        f"| {eng.block_name(r['conc'], r['workload'])} | {r['arm']} | {r['clean']} | {r['need']} |"
        for r in rows if r["need"] > 0)
    GROUP_README.write_text(f"""# Power infill (`{GROUP}`, scope {scope}: {SCOPES[scope][1]})

These runs exist for ONE reason: to raise four under-repeated cells to
**{target_clean} clean repetitions per arm**, so that comparisons which were
previously impossible to resolve become genuinely testable.

## The problem being fixed

`analyze_3e.py` tests policies with a two-sided permutation test on **per-run
medians**. Its resolution is capped by the repetition count, regardless of how
large the true effect is — and because the median ties across many relabellings,
the attainable minimum is worse than the combinatorial bound:

| reps per arm | smallest attainable p | can reach alpha=0.05? |
| --- | --- | --- |
{res}

Four cells — uniform c4, uniform c8, uniform c12, bimodal c4 — held comparisons
at 3 vs 3 and 3 vs 5. Those **cannot** produce a significant result under any
circumstances, so a non-significant outcome there says nothing about whether the
policies differ. This infill removes that limitation rather than reporting those
cells as underpowered.

Cells NOT touched: bimodal c8 (11v5), bimodal c12 (8v5), sharegpt c8 (9v3) —
already testable.

## What was collected

| cell | arm | clean reps before | runs added |
| --- | --- | --- | --- |
{tbl}

## Why these runs are comparable to the originals

They are produced by `run_power_infill.py`, which imports and calls
`run_3e.run_cell()` **unmodified**. Same frozen `router_3nodes.toml`, same
`llama-server` build, same bench invocation (100 measured requests + 3 warm-up,
rate 2, seed 42), same health/thermal gate, same cooldown, same clean/suspect
verdict rule. Only the choice of which cells to run differs. They append to the
existing `<arm>/<cell>/rep<N>/` tree with rep numbers continuing after the
highest existing rep, so `analyze_3e.py` consumes them with no changes.

## Provenance

- `run_meta.json` of every run here carries `"extension_group": "{GROUP}"`.
- `manifest.json` in this folder lists each run, the deficit that justified it,
  and the verdict it returned.
- Runs are physically stored in the normal results tree (not in this folder);
  this folder holds only the rationale and the manifest.

## Honest caveat

These runs were collected **after** the original sweep and after the analysis had
been seen. They increase resolution; they do not re-randomise anything. The cells
and arms were chosen by a power criterion that is independent of the direction of
any observed effect (rep count, not p-value or effect sign), which is what keeps
this from being outcome-driven data collection. State that in the thesis.
""")


def append_manifest(entry):
    GROUP_DIR.mkdir(parents=True, exist_ok=True)
    data = []
    if MANIFEST.is_file():
        try:
            data = json.loads(MANIFEST.read_text())
        except (OSError, json.JSONDecodeError):
            data = []
    data.append(entry)
    MANIFEST.write_text(json.dumps(data, indent=2) + "\n")


# ───────────────────────── Execution ─────────────────────────
def do_run(target_clean, force, scope=DEFAULT_SCOPE):
    # Check every router config this scope will launch, including the wlc_conc config.
    arms, _ = SCOPES[scope]
    configs = {}
    for arm in arms:
        spec = eng.arm_spec(arm)
        configs.setdefault(spec["config"], None if spec["policy"] == arm else arm)
    ok = True
    for cfg_name, arm_for_cfg in sorted(configs.items(), key=lambda kv: kv[0]):
        cok, lines = eng.preflight(strict=True, arm=arm_for_cfg)
        print(f"  --- pre-flight for {cfg_name}"
              + (f" (arm {arm_for_cfg})" if arm_for_cfg else " (frozen default)"))
        for ln in lines:
            print("  " + ln)
        ok = ok and cok
    if not ok and not force:
        sys.exit("\nAborting: local pre-flight failed. Re-run with --force to override.")

    sok, slines = eng.ssh_preflight()
    bok, blines = eng.build_preflight()
    for ln in slines + blines:
        print("  " + ln)
    if not (sok and bok) and not force:
        sys.exit("\nAborting: cluster pre-flight failed. Re-run with --force to override.")

    rows, deficit = build_plan(target_clean, scope)
    if deficit == 0:
        print(f"\nNothing to do — every targeted cell already has {target_clean} clean reps.")
        return
    write_group_docs(target_clean, rows, scope)

    commit = eng.git_commit()
    down = set()
    attempts = []
    eng.driver_log(f"=== POWER INFILL ({GROUP}) scope {scope}: {deficit} clean runs needed, "
                   f"git {commit} ===")

    wave_idx = 0
    while True:
        wave = pending_cells(target_clean, wave_idx, scope)
        if not wave:
            break
        eng.driver_log(f"INFILL wave {wave_idx + 1}: {len(wave)} run(s) — order "
                       + ", ".join(f"{c['policy']}@{eng.block_name(c['conc'], c['workload'])}"
                                   for c in wave))
        wave_idx += 1
        stop = False
        for cell in wave:
            need, have = cell.pop("_need"), cell.pop("_clean")
            arm = cell.pop("_arm")
            # Replica names come from THIS arm's config (wlc_conc uses the
            # re-weighted toml), matching run_3e's own single-cell path.
            replica_names, _, _, _ = eng.parse_toml_frozen(eng.arm_spec(arm)["config"])
            key = (arm, cell["workload"], cell["conc"])
            SESSION_ATTEMPTS[key] = SESSION_ATTEMPTS.get(key, 0) + 1
            eng.driver_log(
                f"INFILL {arm} {eng.block_name(cell['conc'], cell['workload'])} "
                f"rep{cell['rep']} (clean {have}/{target_clean}, {need} still needed, "
                f"attempt {SESSION_ATTEMPTS[key]}/{SESSION_ATTEMPTS.get(('budget',)+key, '?')})")
            verdict = eng.run_cell(cell, commit, down, replica_names)
            attempts.append((cell, verdict, arm))
            append_manifest({
                "arm": arm, "policy": cell["policy"], "conc": cell["conc"],
                "workload": cell["workload"], "rep": cell["rep"], "verdict": verdict,
                "group": GROUP, "scope": scope,
                "clean_before": have, "target_clean": target_clean,
                "reason": "power infill: rep count could not reach alpha=0.05",
                "git": commit, "ts": eng.now_iso(),
            })
            if verdict == "skipped" and down:
                eng.driver_log(f"STOP infill: node(s) down {sorted(down)}; resumable after recovery")
                stop = True
                break
        if stop:
            break

    # refresh the README with final counts
    rows, _ = build_plan(target_clean, scope)
    write_group_docs(target_clean, rows, scope)
    print_final(attempts, target_clean, scope)


def print_final(attempts, target_clean, scope=DEFAULT_SCOPE):
    print("\n" + "=" * 92)
    print(f"POWER INFILL COMPLETE — {len(attempts)} run(s) attempted")
    print("=" * 92)
    for cell, verdict, arm in attempts:
        print(f"  {verdict:8s} {arm:28s} "
              f"{eng.block_name(cell['conc'], cell['workload']):10s} rep{cell['rep']}")
    rows, deficit = build_plan(target_clean, scope)

    if SUSPECT_LIMITED:
        print(f"\nSUSPECT-LIMITED ({len(SUSPECT_LIMITED)} cells) — the attempt budget ran out "
              f"before reaching {target_clean} clean reps.")
        print("  This is NOT a power problem to keep grinding at: the policy is unreliable at")
        print("  that load, and the suspect rate is itself a reportable reliability result.")
        for (arm, w, c), info in sorted(SUSPECT_LIMITED.items()):
            print(f"    {eng.block_name(c, w):10s} {arm:28s} clean={info['clean']} "
                  f"still_needed={info['need']} attempts_this_session={info['attempts_this_session']}"
                  f"/{info['budget']}")
        print("  Re-running the script grants a fresh budget if you want to push further.")

    still = [r for r in rows if r["need"] > 0]
    if still:
        print(f"\nSTILL SHORT ({deficit} clean runs): re-run this script to continue.")
        for r in still:
            print(f"  {eng.block_name(r['conc'], r['workload']):10s} {r['arm']:28s} "
                  f"clean={r['clean']} need={r['need']}")
    elif not SUSPECT_LIMITED:
        print(f"\nAll targeted cells now have >= {target_clean} clean reps. "
              f"Re-run:  python3 analyze_3e.py")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true",
                    help="execute the infill (default is plan-only, no cluster contact)")
    ap.add_argument("--force", action="store_true", help="proceed even if pre-flight fails")
    ap.add_argument("--scope", choices=sorted(SCOPES), default=DEFAULT_SCOPE,
                    help="; ".join(f"{k}={v[1]}" for k, v in sorted(SCOPES.items()))
                         + f" (default {DEFAULT_SCOPE})")
    ap.add_argument("--target", type=int, default=DEFAULT_TARGET_CLEAN,
                    help=f"clean reps per arm to reach (default {DEFAULT_TARGET_CLEAN})")
    args = ap.parse_args()

    if args.run:
        do_run(args.target, args.force, args.scope)
    else:
        print_plan(args.target, args.scope)


if __name__ == "__main__":
    main()
