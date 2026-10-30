"""Driver for the router-tracked load experiment (four arms, bimodal concurrency 8).

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
GROUP = "freshness_extension_v1"
ROOT = HERE / GROUP
ARMS = ("state_aware", "weighted_least_connections", "state_aware_tracked", "wlc_tracked")
WORKLOAD, CONC = "bimodal", 8
ROUTER_CONFIG = "router_3nodes.toml"
WEIGHTS = [3.1, 1.0, 1.0]
DATASET = "local_datasets/bimodal_50short_50long_qwen_tokens.json"
# Balanced Latin square for four arms (Williams design).
LATIN = ((0, 1, 3, 2), (1, 2, 0, 3), (2, 3, 1, 0), (3, 0, 2, 1))


def load_engine():
    spec = importlib.util.spec_from_file_location("run_3e", HERE / "run_3e.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT_ROOT = ROOT
    module.DRIVER_LOG = ROOT / "driver.log"
    module.WORKLOAD_ARGS = {**module.WORKLOAD_ARGS,
                            WORKLOAD: ["--dataset-name", "local", "--dataset-path", DATASET]}
    for arm in ARMS:
        module.ARMS[f"{arm}__fresh"] = {
            "policy": arm, "config": ROUTER_CONFIG, "weights": WEIGHTS,
            "note": f"{GROUP}: {arm}, bimodal c8, Qwen partitioned 4096",
        }
    return module


def arm_name(arm):
    return f"{arm}__fresh"


def dataset_blockers(engine):
    path = engine.BENCH_DIR / DATASET
    if not path.exists():
        return [f"tokenized dataset missing: {path} (run benchmarks/tokenize_dataset.py)"]
    data = json.loads(path.read_text())
    original = json.loads(engine.BIMODAL_DATASET.read_text())
    reasons = []
    if [{k: v for k, v in d.items() if k != "prompt_tokens"} for d in data] != original:
        reasons.append("tokenized dataset differs from the main bimodal dataset "
                       "apart from prompt_tokens")
    if not all(isinstance(d.get("prompt_tokens"), int) and d["prompt_tokens"] > 0 for d in data):
        reasons.append("tokenized dataset has items without a positive prompt_tokens")
    return reasons


def inventory(arm):
    directory = ROOT / arm_name(arm) / f"c{CONC}_{WORKLOAD}"
    measured = highest = 0
    if directory.exists():
        for rep in directory.glob("rep*"):
            if not rep.is_dir() or not rep.name[3:].isdigit():
                continue
            highest = max(highest, int(rep.name[3:]))
            meta_path = rep / "run_meta.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text())
            summary = meta.get("bench_summary") or {}
            result = meta.get("bench_result_file")
            if (meta.get("extension_group") == GROUP
                    and summary.get("num_prompts") == 100 and result
                    and (rep / result).is_file()
                    and meta.get("verdict") in ("clean", "suspect")):
                measured += 1
    return measured, highest + 1


def schedule(counts, reps):
    """Remaining order: block b runs every arm with fewer than b+1 measured
    runs, in Latin-square row b mod 4."""
    counts = dict(counts)
    order = []
    while any(counts[a] < reps for a in ARMS):
        block = min(counts[a] for a in ARMS if counts[a] < reps)
        for i in LATIN[block % 4]:
            arm = ARMS[i]
            if counts[arm] == block and counts[arm] < reps:
                order.append(arm)
                counts[arm] += 1
    return order


def source_hashes(engine):
    files = [Path(__file__), HERE / "run_3e.py", engine.BENCH_DIR / engine.BENCH_SCRIPT,
             engine.ROUTER_BIN, engine.ROUTER_DIR / ROUTER_CONFIG, engine.BENCH_DIR / DATASET]
    return {str(path.relative_to(engine.THESIS)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files}


def journal(engine, event, **details):
    with (ROOT / "manifest.jsonl").open("a") as handle:
        handle.write(json.dumps({"time": engine.now_iso(), "event": event,
                                 "group": GROUP, **details}) + "\n")


def show_plan(engine, reps):
    counts = {a: inventory(a)[0] for a in ARMS}
    print(f"{GROUP}: bimodal c{CONC}, {reps} measured runs per arm; suspects count.")
    print(f"  router config {ROUTER_CONFIG}, weights {WEIGHTS}, dataset {DATASET}")
    print(f"  llama-server {engine.llama_server_args(engine.DEFAULT_SERVER_CONFIG)}")
    for reason in dataset_blockers(engine):
        print(f"  BLOCKED: {reason}")
    for arm in ARMS:
        print(f"  {arm:27} measured={counts[arm]}/{reps}")
    order = schedule(counts, reps)
    print(f"\nRemaining order ({len(order)} runs):")
    for i, arm in enumerate(order, 1):
        print(f"  {i:3} {arm}")
    return order


def execute(engine, reps):
    ROOT.mkdir(exist_ok=True)
    with (ROOT / "runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another freshness extension runner is already active.")
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", 3000)) == 0:
                raise SystemExit("Port 3000 is occupied. Stop the existing router first.")

        checks = [lambda arm=arm_name(a): engine.preflight(strict=True, arm=arm) for a in ARMS]
        for check in checks + [engine.ssh_preflight, engine.build_preflight,
                               engine.thermal_preflight]:
            ok, lines = check()
            for line in lines:
                print(line, flush=True)
            if not ok:
                raise SystemExit("Preflight failed; no experimental run launched.")

        settings = {"group": GROUP, "repetitions": reps, "arms": list(ARMS),
                    "cell": [WORKLOAD, CONC], "router_config": ROUTER_CONFIG,
                    "weights": WEIGHTS, "dataset": DATASET,
                    "counting_rule": "measured clean and suspect runs both count",
                    "order": "balanced Latin-square blocks",
                    "source_sha256": source_hashes(engine)}
        settings_path = ROOT / "settings.json"
        if settings_path.exists():
            if json.loads(settings_path.read_text()) != settings:
                raise SystemExit("Design or source files changed since this extension began. "
                                 "Resolve the difference before resuming.")
        else:
            settings_path.write_text(json.dumps(settings, indent=2) + "\n")

        ok, lines = engine.apply_server_config(engine.DEFAULT_SERVER_CONFIG)
        for line in lines:
            print(line, flush=True)
        journal(engine, "server_config", config="default", ok=ok, detail=lines)
        if not ok:
            raise SystemExit("Could not put all nodes on the default configuration. Stopping.")

        names, _, _, _ = engine.parse_toml_frozen()
        commit = engine.git_commit()
        down = set()
        while order := schedule({a: inventory(a)[0] for a in ARMS}, reps):
            arm = order[0]
            before, next_rep = inventory(arm)
            cell = {"policy": arm, "arm": arm_name(arm), "conc": CONC, "workload": WORKLOAD,
                    "rep": next_rep, "group": GROUP}
            journal(engine, "started", cell=cell, git=commit)
            try:
                verdict = engine.run_cell(cell, commit, down, names, cooldown=True)
            except BaseException as exc:
                journal(engine, "interrupted_or_error", cell=cell,
                        error=f"{type(exc).__name__}: {exc}")
                raise
            journal(engine, "finished", cell=cell, verdict=verdict)
            if inventory(arm)[0] != before + 1:
                raise SystemExit("Run did not produce a complete benchmark result. Stopping "
                                 "for inspection; rerun the same command after resolving it.")
        print("Freshness extension complete. All measured runs, including suspects, were retained.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="store_true", help="execute; otherwise offline plan only")
    parser.add_argument("--reps", type=int, required=True,
                        help="fixed measured runs per arm, chosen before collection")
    args = parser.parse_args()
    if args.reps < 1:
        parser.error("--reps must be positive")

    engine = load_engine()
    order = show_plan(engine, args.reps)
    if not order:
        print("Nothing remaining.")
        return
    if not args.run:
        print("\nPlan only: no cluster contact and no files written. Add --run to execute.")
        return
    blocked = dataset_blockers(engine)
    if blocked:
        raise SystemExit("Refusing to run: " + "; ".join(blocked))
    execute(engine, args.reps)


if __name__ == "__main__":
    main()
