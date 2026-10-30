"""Driver for the follow-up campaign: other models and unified KV allocation.

Without flags it only prints the plan; --run executes it on the cluster.
"""

import argparse
import fcntl
import hashlib
import importlib.util
import json
import socket
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
GROUP = "model_kv_extension_v1"
ROOT = HERE / GROUP
CONFIGS_FILE = ROOT / "configs.json"
POLICIES = ("state_aware", "weighted_least_connections")
ROWS = (  # (configuration, workload, concurrency)
    ("qwen_p4096", "bimodal", 8),
    ("qwen_p4096", "sharegpt", 12),
    ("lfm2_p4096", "bimodal", 8),
    ("lfm2_p4096", "sharegpt", 12),
    ("openelm_p4096", "bimodal", 8),
    ("openelm_p4096", "sharegpt", 12),
    ("qwen_u4096", "bimodal", 8),
)
CONFIG_ORDER = ("qwen_p4096", "lfm2_p4096", "openelm_p4096", "qwen_u4096")
SERVER_KEYS = ("model_path", "model_name", "kv_mode", "ctx_size", "slots")


def load_engine():
    spec = importlib.util.spec_from_file_location("run_3e", HERE / "run_3e.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Isolate this extension's runs and log from the original sweep.
    module.OUT_ROOT = ROOT
    module.DRIVER_LOG = ROOT / "driver.log"
    return module


def load_configs():
    data = json.loads(CONFIGS_FILE.read_text())
    return {name: data[name] for name in CONFIG_ORDER}


def arm_name(policy, config):
    return f"{policy}__{config}"


def server_config(name, cfg):
    return {"name": name, **{key: cfg[key] for key in SERVER_KEYS}}


def blockers(engine, name, cfg):
    """Reasons a configuration cannot run yet (empty list = ready)."""
    reasons = []
    try:
        engine.validate_server_config(server_config(name, cfg))
    except ValueError as exc:
        reasons.append(str(exc))
    weights = cfg.get("weights")
    if not (isinstance(weights, list) and len(weights) == 3
            and all(isinstance(w, (int, float)) and w > 0 for w in weights)):
        reasons.append("weights not set (calibrate, then set them in configs.json "
                       "and in its router TOML)")
    toml_path = engine.ROUTER_DIR / cfg["router_config"]
    if not toml_path.exists():
        reasons.append(f"router config missing: {toml_path}")
    else:
        toml = tomllib.loads(toml_path.read_text())
        if toml.get("kv_mode", "partitioned") != cfg["kv_mode"]:
            reasons.append(f"{cfg['router_config']} kv_mode={toml.get('kv_mode', 'partitioned')!r} "
                           f"but configuration uses {cfg['kv_mode']!r}")
        budgets = {r.get("context_budget") for r in toml.get("replicas", [])}
        if budgets != {cfg["ctx_size"]}:
            reasons.append(f"{cfg['router_config']} context_budget={sorted(budgets)} "
                           f"but ctx_size={cfg['ctx_size']}")
    return reasons


def register_arms(engine, configs):
    for name, cfg in configs.items():
        for policy in POLICIES:
            engine.ARMS[arm_name(policy, name)] = {
                "policy": policy, "config": cfg["router_config"],
                "weights": cfg.get("weights"),
                "note": f"{GROUP}: {name} ({cfg['kv_mode']} KV, ctx {cfg['ctx_size']}, "
                        f"{Path(cfg['model_path'] or 'model TBD').name})",
            }


def inventory(arm, workload, concurrency):
    """Measured runs of THIS extension for one arm and cell."""
    directory = ROOT / arm / f"c{concurrency}_{workload}"
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


def units(selected):
    """Every (configuration, workload, concurrency, policy) to collect."""
    return [(c, w, n, p) for c, w, n in ROWS if c in selected for p in POLICIES]


def round_order(round_index, selected):
    """Visiting order within one round (deterministic, counterbalanced)."""
    configs = [c for c in CONFIG_ORDER if c in selected]
    k = len(configs)
    start = round_index % k
    rotated = configs[start:] + configs[:start]
    if (round_index // k) % 2:
        rotated = rotated[::-1]
    order = []
    for config in rotated:
        cells = [(w, n) for c, w, n in ROWS if c == config]
        if round_index % 2:
            cells = cells[::-1]
        policies = POLICIES if round_index % 2 == 0 else POLICIES[::-1]
        order += [(config, w, n, p) for w, n in cells for p in policies]
    return order


def next_unit(counts, reps, selected):
    """Earliest unfinished round; first pending unit in its order."""
    pending = [u for u in units(selected) if counts[u] < reps]
    if not pending:
        return None
    round_index = min(counts[u] for u in pending)
    for unit in round_order(round_index, selected):
        if counts[unit] == round_index:
            return unit
    raise RuntimeError("could not schedule a pending unit")


def schedule(counts, reps, selected):
    """Full remaining order, simulated from the current counts."""
    counts = dict(counts)
    order = []
    while (unit := next_unit(counts, reps, selected)) is not None:
        order.append(unit)
        counts[unit] += 1
    return order


def current_counts(selected):
    return {u: inventory(arm_name(u[3], u[0]), u[1], u[2])[0] for u in units(selected)}


def show_plan(engine, configs, selected, reps):
    print(f"{GROUP}: {reps} measured runs per policy and row; suspects count.\n")
    for name in CONFIG_ORDER:
        cfg = configs[name]
        tag = "selected" if name in selected else "not selected"
        reasons = blockers(engine, name, cfg)
        print(f"[{name}] ({tag}) {'READY' if not reasons else 'BLOCKED'}")
        print(f"  router config: {cfg['router_config']}  weights: {cfg.get('weights')}")
        if reasons:
            for reason in reasons:
                print(f"  BLOCKED: {reason}")
        else:
            print(f"  llama-server {engine.llama_server_args(server_config(name, cfg))}")
    counts = current_counts(selected)
    print("\nRows:")
    for config, workload, conc in ROWS:
        if config not in selected:
            continue
        for policy in POLICIES:
            print(f"  {config:11} {workload:8} c{conc:<2} {policy:27} "
                  f"measured={counts[(config, workload, conc, policy)]}/{reps}")
    order = schedule(counts, reps, selected)
    print(f"\nRemaining order ({len(order)} runs; '*' = configuration switch, "
          f"which restarts and verifies all nodes):")
    previous = None
    for i, (config, workload, conc, policy) in enumerate(order, 1):
        mark = "*" if config != previous else " "
        print(f"  {i:3} {mark} {config:11} {workload:8} c{conc:<2} {policy}")
        previous = config
    switches = sum(1 for a, b in zip([None] + order, order) if a is None or a[0] != b[0])
    print(f"\nConfiguration switches: {switches}")
    return order


def source_hashes(engine):
    files = [Path(__file__), HERE / "run_3e.py", engine.BENCH_DIR / engine.BENCH_SCRIPT,
             engine.ROUTER_BIN]
    return {str(path.relative_to(engine.THESIS)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files}


def freeze_config(engine, name, cfg):
    """Record a configuration's definition on first use; refuse later changes."""
    lock_path = ROOT / "config_lock.json"
    locks = json.loads(lock_path.read_text()) if lock_path.exists() else {}
    toml_path = engine.ROUTER_DIR / cfg["router_config"]
    record = {**cfg, "router_config_sha256": hashlib.sha256(toml_path.read_bytes()).hexdigest()}
    if name in locks and locks[name] != record:
        raise SystemExit(f"Configuration {name} changed after collection began "
                         f"(configs.json or {cfg['router_config']}). Stopping.")
    locks[name] = record
    lock_path.write_text(json.dumps(locks, indent=2) + "\n")


def journal(engine, event, **details):
    with (ROOT / "manifest.jsonl").open("a") as handle:
        handle.write(json.dumps({"time": engine.now_iso(), "event": event,
                                 "group": GROUP, **details}) + "\n")


def execute(engine, configs, selected, reps):
    with (ROOT / "runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another model/KV extension runner is already active.")
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", 3000)) == 0:
                raise SystemExit("Port 3000 is occupied. Stop the existing router first.")

        checks = [lambda arm=arm_name(p, c): engine.preflight(strict=True, arm=arm)
                  for c in selected for p in POLICIES]
        for check in checks + [engine.ssh_preflight, engine.build_preflight,
                               engine.thermal_preflight]:
            ok, lines = check()
            for line in lines:
                print(line, flush=True)
            if not ok:
                raise SystemExit("Preflight failed; no experimental run launched.")

        settings = {"group": GROUP, "repetitions": reps, "policies": list(POLICIES),
                    "rows": [list(r) for r in ROWS],
                    "counting_rule": "measured clean and suspect runs both count",
                    "source_sha256": source_hashes(engine)}
        settings_path = ROOT / "settings.json"
        if settings_path.exists():
            if json.loads(settings_path.read_text()) != settings:
                raise SystemExit("Design or source files changed since this extension began. "
                                 "Resolve the difference before resuming.")
        else:
            settings_path.write_text(json.dumps(settings, indent=2) + "\n")

        names, _, _, _ = engine.parse_toml_frozen()
        commit = engine.git_commit()
        down = set()
        active = None
        while (unit := next_unit(current_counts(selected), reps, selected)) is not None:
            config, workload, conc, policy = unit
            if config != active:
                freeze_config(engine, config, configs[config])
                ok, lines = engine.apply_server_config(server_config(config, configs[config]))
                for line in lines:
                    print(line, flush=True)
                journal(engine, "server_config", config=config, ok=ok, detail=lines)
                if not ok:
                    raise SystemExit(f"Could not switch all nodes to {config}. Stopping.")
                active = config
            arm = arm_name(policy, config)
            before, next_rep = inventory(arm, workload, conc)
            cell = {"policy": policy, "arm": arm, "conc": conc, "workload": workload,
                    "rep": next_rep, "group": GROUP}
            journal(engine, "started", cell=cell, config=config, git=commit)
            try:
                verdict = engine.run_cell(cell, commit, down, names, cooldown=True)
            except BaseException as exc:
                journal(engine, "interrupted_or_error", cell=cell,
                        error=f"{type(exc).__name__}: {exc}")
                raise
            journal(engine, "finished", cell=cell, verdict=verdict)
            if inventory(arm, workload, conc)[0] != before + 1:
                raise SystemExit("Run did not produce a complete benchmark result. Stopping "
                                 "for inspection; rerun the same command after resolving it.")
        print("Extension complete for the selected configurations. "
              "All measured runs, including suspects, were retained.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="store_true", help="execute; otherwise offline plan only")
    parser.add_argument("--reps", type=int, default=6,
                        help="fixed measured runs per policy and row (default 6)")
    parser.add_argument("--configs", default=",".join(CONFIG_ORDER),
                        help="comma-separated subset of configurations to schedule")
    args = parser.parse_args()
    selected = tuple(c for c in CONFIG_ORDER if c in args.configs.split(","))
    unknown = set(args.configs.split(",")) - set(CONFIG_ORDER)
    if unknown or not selected:
        parser.error(f"unknown or empty --configs: {sorted(unknown) or args.configs}")
    if args.reps < 1:
        parser.error("--reps must be positive")

    engine = load_engine()
    configs = load_configs()
    register_arms(engine, configs)
    order = show_plan(engine, configs, selected, args.reps)
    if not order:
        print("Nothing remaining for the selected configurations.")
        return
    if not args.run:
        print("\nPlan only: no cluster contact and no files written. Add --run to execute.")
        return
    blocked = {c: blockers(engine, c, configs[c]) for c in selected}
    blocked = {c: r for c, r in blocked.items() if r}
    if blocked:
        raise SystemExit("Refusing to run: blocked configurations selected: "
                         + ", ".join(sorted(blocked))
                         + ". Fix them or pass --configs with ready ones only.")
    execute(engine, configs, selected, args.reps)


if __name__ == "__main__":
    main()
