"""Driver for the ShareGPT concurrency 4 and 12 runs of the baseline policies.

Without flags it only prints the plan; --run executes it on the cluster.
"""

import argparse
import fcntl
import hashlib
import importlib.util
import json
import socket
from pathlib import Path

HERE = Path(__file__).resolve().parent
GROUP = "sharegpt_c4_c12_v1"
ARMS = ("weighted_least_connections", "state_aware", "weighted_round_robin")
CONCURRENCIES = (4, 12)


def load_engine():
    spec = importlib.util.spec_from_file_location("run_3e", HERE / "run_3e.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inventory(root, arm, concurrency):
    """Count only this extension; reserve ALL existing repetition directories."""
    directory = root / arm / f"c{concurrency}_sharegpt"
    measured = clean = suspect = highest = 0
    if directory.exists():
        for rep in directory.glob("rep*"):
            if not rep.is_dir() or not rep.name[3:].isdigit():
                continue
            highest = max(highest, int(rep.name[3:]))
            meta_path = rep / "run_meta.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text())
            if meta.get("extension_group") != GROUP:
                continue
            summary = meta.get("bench_summary") or {}
            result = meta.get("bench_result_file")
            if (summary.get("num_prompts") == 100 and result
                    and (rep / result).is_file()
                    and meta.get("verdict") in ("clean", "suspect")):
                measured += 1
                clean += meta["verdict"] == "clean"
                suspect += meta["verdict"] == "suspect"
    return {"measured": measured, "clean": clean, "suspect": suspect,
            "next_rep": highest + 1}


def plan(root, repetitions):
    rows = []
    for concurrency in CONCURRENCIES:
        for arm in ARMS:
            row = inventory(root, arm, concurrency)
            row.update(arm=arm, concurrency=concurrency)
            row["remaining"] = max(0, repetitions - row["measured"])
            rows.append(row)
    return rows


def next_cell(root, repetitions):
    # Resume the earliest unfinished round. Across six rounds, each of the
    # six permutations occurs once, and which concurrency goes first alternates.
    rows = plan(root, repetitions)
    pending = [row for row in rows if row["remaining"]]
    if not pending:
        return None
    round_index = min(row["measured"] for row in pending)
    offset = round_index % len(ARMS)
    direction = 1 if (round_index // len(ARMS)) % 2 == 0 else -1
    order = [ARMS[(offset + direction * i) % len(ARMS)] for i in range(len(ARMS))]
    conc_order = CONCURRENCIES if round_index % 2 == 0 else CONCURRENCIES[::-1]
    lookup = {(row["arm"], row["concurrency"]): row for row in pending}
    for concurrency in conc_order:
        for arm in order:
            row = lookup.get((arm, concurrency))
            if row and row["measured"] == round_index:
                return {"policy": arm, "arm": arm, "conc": concurrency,
                        "workload": "sharegpt", "rep": row["next_rep"],
                        "group": GROUP}
    raise RuntimeError("Could not schedule pending run")


def show_plan(root, repetitions):
    rows = plan(root, repetitions)
    print(f"{GROUP}: {repetitions} measured runs per arm/cell; suspects count.")
    for row in rows:
        print(f"  c{row['concurrency']:2} {row['arm']:28} "
              f"measured={row['measured']} clean={row['clean']} "
              f"suspect={row['suspect']} remaining={row['remaining']}")
    remaining = sum(row["remaining"] for row in rows)
    print(f"Remaining: {remaining} runs / {remaining * 100} measured requests.")
    return remaining


def source_hashes(engine):
    files = [Path(__file__), HERE / "run_3e.py",
             engine.BENCH_DIR / engine.BENCH_SCRIPT,
             engine.ROUTER_DIR / engine.ROUTER_CFG, engine.ROUTER_BIN]
    return {str(path.relative_to(engine.THESIS)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files}


def journal(directory, engine, event, **details):
    with (directory / "manifest.jsonl").open("a") as handle:
        handle.write(json.dumps({"time": engine.now_iso(), "event": event,
                                 "group": GROUP, **details}) + "\n")


def execute(engine, repetitions, directory):
    # This lock prevents two copies of THIS scheduler from running together.
    # Other benchmark drivers must also be left idle during this experiment.
    with (directory / "runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another ShareGPT extension runner is already active.")
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", 3000)) == 0:
                raise SystemExit("Port 3000 is occupied. Stop the existing router before starting.")

        for check in (lambda: engine.preflight(strict=True), engine.ssh_preflight,
                      engine.build_preflight, engine.thermal_preflight):
            ok, lines = check()
            for line in lines:
                print(line, flush=True)
            if not ok:
                raise SystemExit("Preflight failed; no experimental run launched.")

        settings = {"group": GROUP, "repetitions": repetitions,
                    "counting_rule": "measured clean and suspect runs both count",
                    "arms": list(ARMS), "concurrencies": list(CONCURRENCIES),
                    "source_sha256": source_hashes(engine)}
        settings_path = directory / "settings.json"
        if settings_path.exists():
            if json.loads(settings_path.read_text()) != settings:
                raise SystemExit("Settings or source files changed since this extension began. "
                                 "Resolve the difference before resuming.")
        else:
            settings_path.write_text(json.dumps(settings, indent=2) + "\n")

        names, _, _, _ = engine.parse_toml_frozen()
        commit = engine.git_commit()
        down = set()
        while (cell := next_cell(engine.OUT_ROOT, repetitions)) is not None:
            before = inventory(engine.OUT_ROOT, cell["arm"], cell["conc"])["measured"]
            journal(directory, engine, "started", cell=cell, git=commit)
            try:
                verdict = engine.run_cell(cell, commit, down, names, cooldown=True)
            except BaseException as exc:
                journal(directory, engine, "interrupted_or_error", cell=cell,
                        error=f"{type(exc).__name__}: {exc}")
                raise
            journal(directory, engine, "finished", cell=cell, verdict=verdict)
            after = inventory(engine.OUT_ROOT, cell["arm"], cell["conc"])["measured"]
            if after != before + 1:
                raise SystemExit("Run did not produce a complete benchmark result. Stopping "
                                 "for inspection; rerun the same command after resolving it.")
            show_plan(engine.OUT_ROOT, repetitions)
        print("Extension complete. All measured runs, including suspects, were retained.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="execute; otherwise offline plan only")
    parser.add_argument("--reps", type=int, default=6, help="fixed measured runs per arm/cell (default 6)")
    args = parser.parse_args()
    if args.reps < 1:
        parser.error("--reps must be positive")
    engine = load_engine()
    if not show_plan(engine.OUT_ROOT, args.reps):
        return
    if not args.run:
        print("Plan only: no cluster contact and no files written. Add --run to execute.")
        return
    directory = engine.OUT_ROOT / GROUP
    directory.mkdir(parents=True, exist_ok=True)
    execute(engine, args.reps, directory)


if __name__ == "__main__":
    main()
