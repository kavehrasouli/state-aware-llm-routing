"""Experiment driver for the main routing grid.

Without flags it only prints the plan; --run executes it on the cluster.
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

# Paths
THESIS = Path(__file__).resolve().parents[1]  # repository root
ROUTER_DIR = THESIS / "routers" / "router"
ROUTER_BIN = ROUTER_DIR / "target" / "release" / "router"
ROUTER_CFG = "router_3nodes.toml"  # relative to ROUTER_DIR (frozen — never modified)
BENCH_DIR = THESIS / "benchmarks"
BENCH_PY = BENCH_DIR / ".venv" / "bin" / "python3"
BENCH_SCRIPT = "bench.py"
RESULTS_DIR = BENCH_DIR / "results"
BIMODAL_DATASET = BENCH_DIR / "local_datasets" / "bimodal_50short_50long.json"
OUT_ROOT = THESIS / "3e_results"
DRIVER_LOG = OUT_ROOT / "driver.log"
README = OUT_ROOT / "README.md"
BACKUP_MIN_RUN_META = 127

# ───────────────────────── Cluster ─────────────────────────
# (name, ip, node-number-for-logfile)
NODES = [
    ("pi-node-1", "192.168.2.2", 1),  # Pi 5 (8GB)
    ("pi-node-2", "192.168.2.3", 2),  # Pi 4B (4GB)
    ("pi-node-3", "192.168.2.5", 3),  # Pi 4B (8GB)
]
SSH_USER = "admin"
MODEL_PATH = "/opt/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
LLAMA_BIN = "/home/admin/llama.cpp-a935fbffe/build/bin/llama-server"
LLAMA_BUILD = "9960 / a935fbffe"

# Inference-server configuration. Wrappers switch it with set_server_config(), so every launch
# and restart uses the configuration of the run in progress.
KV_FLAGS = {"partitioned": "--no-kv-unified", "unified": "--kv-unified"}
DEFAULT_SERVER_CONFIG = {
    "name": "qwen0.5b_partitioned_4096",
    "model_path": MODEL_PATH,
    "model_name": "qwen2.5-0.5b",
    "kv_mode": "partitioned",
    "ctx_size": 4096,
    "slots": 4,
}
SERVER_CONFIG = dict(DEFAULT_SERVER_CONFIG)


def validate_server_config(cfg):
    """Raise ValueError unless cfg is complete and launchable."""
    missing = [k for k in DEFAULT_SERVER_CONFIG if cfg.get(k) in (None, "")]
    if missing:
        raise ValueError(f"server config {cfg.get('name')!r} is missing {missing}")
    if cfg["kv_mode"] not in KV_FLAGS:
        raise ValueError(f"unknown kv_mode {cfg['kv_mode']!r}")
    for key in ("ctx_size", "slots"):
        if not isinstance(cfg[key], int) or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer, got {cfg[key]!r}")


def set_server_config(cfg):
    """Make cfg the configuration used by launches, restarts, bench and meta."""
    global SERVER_CONFIG
    validate_server_config(cfg)
    SERVER_CONFIG = dict(cfg)


def llama_server_args(cfg=None):
    """llama-server arguments for cfg (default: the active configuration)."""
    cfg = cfg or SERVER_CONFIG
    return (f"--model {cfg['model_path']} --host 0.0.0.0 --port 8080 "
            f"--ctx-size {cfg['ctx_size']} -np {cfg['slots']} "
            f"{KV_FLAGS[cfg['kv_mode']]} --metrics")

# ───────────────────────── Grid ─────────────────────────
BASELINE_POLICIES = [
    "random",
    "round_robin",
    "least_connections",
    "weighted_round_robin",
    "weighted_least_connections",
    "state_aware",
]
NEW_POLICIES = ["state_aware_v2", "cost_predict_v3"]
POLICIES = BASELINE_POLICIES + NEW_POLICIES

# Arms: a (policy, router config) pair. Every policy is its own arm with the main config.
# wlc_conc is WLC with Pi 5 weight 6.0, which matches cost_predict_v3's routing share
# (higher weights give the same split, because the Pi 5's four slots fill up).
ARMS = {
    "wlc_conc": {
        "policy": "weighted_least_connections",
        "config": "router_3nodes_conc.toml",
        "weights": [6.0, 1.0, 1.0],
        "note": "concentration control: WLC re-weighted to match v3's Pi-5 share",
    },
}


def arm_spec(arm):
    """Resolve an arm name to {policy, config, weights, note}."""
    if arm in ARMS:
        return dict(ARMS[arm])
    return {"policy": arm, "config": ROUTER_CFG,
            "weights": list(FROZEN["weights"]), "note": None}


def arm_label(cell):
    """Results-directory name: the arm, falling back to the bare policy."""
    return cell.get("arm") or cell["policy"]
CONC = [4, 8, 12]
MAIN_WORKLOADS = ["uniform", "bimodal"]
SHAREGPT_CONC = 8
EXTENSION_REP_START = 4
NEW_POLICY_TARGET_REPS = 5
INFILL_TARGET_CLEAN = 5
INFILL_ATTEMPT_CAP = 8
INFILL_CONC = [8, 12]

WORKLOAD_ARGS = {
    "uniform": ["--dataset-name", "random", "--random-input-len", "200", "--random-output-len", "150"],
    "bimodal": ["--dataset-name", "local", "--dataset-path", "local_datasets/bimodal_50short_50long.json"],
    "sharegpt": ["--dataset-name", "sharegpt"],
}

# Frozen config, recorded verbatim into every run_meta.json + the README. The
# driver cross-checks router_3nodes.toml against these before an overnight run.
FROZEN = {
    "weights": [3.1, 1.0, 1.0],
    "poll_interval_ms": 350,
    "poll_timeout_ms": 300,
    "dispatch_eviction_threshold": 3,
    "max_state_age_ms": 1000,
    "connect_timeout_ms": 1000,
    "upstream_timeout_ms": 120000,
    "stream_idle_timeout_ms": 120000,
    "llama_build": LLAMA_BUILD,
    "model": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
    "num_prompts": 100,
    "request_rate": 2,
    "seed": 42,
    "cost_coefficients": {
        "pi-node-1": {"c0": 87.5638, "c1": 0.0, "c2": 0.013507},
        "pi-node-2": {"c0": 53.0, "c1": 47.0, "c2": 0.070},
        "pi-node-3": {"c0": 53.0, "c1": 47.0, "c2": 0.070},
    },
    "bench_timeout_s": 2400,
    "minimum_cooldown_s": 60,
    "thermal_start_max_c": 60.0,
    "thermal_stable_readings": 2,
    "thermal_poll_s": 10,
    "thermal_wait_timeout_s": 1200,
}

# ───────────────────────── Thresholds / timing ─────────────────────────
FAILURE_RATE_MAX = 0.10       # verdict gate
COOLDOWN_S = 60               # between runs; servers stay up
COLD_START_TIMEOUT_S = 90     # llama-server cold start after a restart
ELIGIBLE_TIMEOUT_S = 30       # router must report all nodes eligible within this
BENCH_TIMEOUT_S = 2400        # hard cap on one bench run
THERMAL_MAX_C = 60.0          # every node must be at/below this before a cell
THERMAL_STABLE_READINGS = 2   # avoid starting on a single transient reading
THERMAL_POLL_S = 10
THERMAL_WAIT_TIMEOUT_S = 1200
MAX_RESTARTS = 2              # per node per health-gate (hung recovery)
HEARTBEAT_SECS = 2            # router --heartbeat-secs; STDERR only, never the JSONL


# ═════════════════════════ Grid construction ═════════════════════════
def blocks():
    """(workload, concurrency) blocks, in fixed within-rep order."""
    b = [(wl, c) for wl in MAIN_WORKLOADS for c in CONC]
    b.append(("sharegpt", SHAREGPT_CONC))
    return b


def rotated(rep):
    """Counterbalanced eight-policy order for extension rep numbers.

    A step of three is coprime to eight, so rep4..rep11 use every policy as the
    first policy exactly once. Direction alternates so the two new policies do
    not always execute in the same relative order. Cells filter this order to
    policies they need.
    """
    index = rep - EXTENSION_REP_START
    offset = (3 * index) % len(POLICIES)
    direction = 1 if index % 2 == 0 else -1
    return [
        POLICIES[(offset + direction * position) % len(POLICIES)]
        for position in range(len(POLICIES))
    ]


def cell_group(policy, conc, workload):
    if policy == "state_aware_v2":
        return "state_aware_v2_full"
    if policy == "cost_predict_v3":
        return "cost_predict_v3_full"
    if policy in BASELINE_POLICIES and workload == "bimodal" and conc in INFILL_CONC:
        return "baseline_bimodal_infill"
    return None


def block_name(conc, workload):
    return f"c{conc}_{workload}"


def rep_dir(cell):
    return OUT_ROOT / arm_label(cell) / block_name(cell["conc"], cell["workload"]) / f"rep{cell['rep']}"


def policy_metadata(policy):
    if policy == "cost_predict_v3":
        return {
            "experimental": True,
            "predictive": True,
            "request_aware": True,
            "description": "experimental predictive per-request cost policy",
        }
    if policy == "state_aware_v2":
        return {
            "experimental": True,
            "predictive": False,
            "request_aware": False,
            "description": "experimental staged observed-state policy",
        }
    return {
        "experimental": False,
        "predictive": False,
        "request_aware": False,
    }


REP_DIR_RE = re.compile(r"^rep([0-9]+)$")


def cell_root(policy, conc, workload):
    return OUT_ROOT / policy / block_name(conc, workload)


def rep_number(path):
    match = REP_DIR_RE.match(path.name)
    return int(match.group(1)) if match else None


def load_run_meta(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def cell_status(policy, conc, workload):
    """Inventory a cell without changing it.

    All valid metadata contributes to the clean count. Only measured rep4+
    attempts (`clean` or `suspect`) consume an extension attempt; a health-gate
    `skipped` result remains pending after node recovery. Merely existing rep
    directories still reserve their number, which prevents an interrupted
    attempt from ever being overwritten.
    """
    root = cell_root(policy, conc, workload)
    rep_dirs = []
    if root.exists():
        for path in root.iterdir():
            number = rep_number(path) if path.is_dir() else None
            if number is not None:
                rep_dirs.append((number, path))

    metas = []
    for number, path in rep_dirs:
        meta = load_run_meta(path / "run_meta.json")
        if meta is not None:
            metas.append((number, meta))

    return {
        "clean_total": sum(1 for _, meta in metas if meta.get("verdict") == "clean"),
        "extension_attempts": sum(
            1
            for number, meta in metas
            if number >= EXTENSION_REP_START
            and meta.get("verdict") in ("clean", "suspect")
        ),
        "max_rep": max((number for number, _ in rep_dirs), default=EXTENSION_REP_START - 1),
        "metas": metas,
    }


def cell_needs_run(policy, conc, workload):
    group = cell_group(policy, conc, workload)
    if group is None:
        return False
    status = cell_status(policy, conc, workload)
    if group == "baseline_bimodal_infill":
        return (
            status["clean_total"] < INFILL_TARGET_CLEAN
            and status["extension_attempts"] < INFILL_ATTEMPT_CAP
        )
    return status["extension_attempts"] < NEW_POLICY_TARGET_REPS


def next_rep_number(policy, conc, workload):
    return max(EXTENSION_REP_START, cell_status(policy, conc, workload)["max_rep"] + 1)


def extension_specs():
    specs = []
    for policy in BASELINE_POLICIES:
        for conc in INFILL_CONC:
            specs.append((policy, conc, "bimodal"))
    for policy in NEW_POLICIES:
        for workload, conc in blocks():
            specs.append((policy, conc, workload))
    return specs


def remaining_plan():
    """Return the current append-only work estimate.

    Infill has a range because suspect attempts do not increase the clean count:
    `minimum` assumes every new attempt is clean, while `maximum` runs each
    underpowered cell to its remaining attempt cap.
    """
    infill_minimum = 0
    infill_maximum = 0
    underpowered = []
    for policy in BASELINE_POLICIES:
        for conc in INFILL_CONC:
            status = cell_status(policy, conc, "bimodal")
            clean_deficit = max(0, INFILL_TARGET_CLEAN - status["clean_total"])
            attempts_left = max(0, INFILL_ATTEMPT_CAP - status["extension_attempts"])
            if clean_deficit and attempts_left:
                infill_minimum += min(clean_deficit, attempts_left)
                infill_maximum += attempts_left
                underpowered.append({
                    "policy": policy,
                    "conc": conc,
                    "clean": status["clean_total"],
                    "attempts": status["extension_attempts"],
                    "minimum": min(clean_deficit, attempts_left),
                    "maximum": attempts_left,
                    "next_rep": next_rep_number(policy, conc, "bimodal"),
                })

    new_remaining = {}
    for policy in NEW_POLICIES:
        remaining = 0
        for workload, conc in blocks():
            status = cell_status(policy, conc, workload)
            remaining += max(0, NEW_POLICY_TARGET_REPS - status["extension_attempts"])
        new_remaining[policy] = remaining

    return {
        "infill_minimum": infill_minimum,
        "infill_maximum": infill_maximum,
        "underpowered": underpowered,
        "new_remaining": new_remaining,
        "minimum_total": infill_minimum + sum(new_remaining.values()),
        "maximum_total": infill_maximum + sum(new_remaining.values()),
    }


def verified_backups():
    backups = []
    for path in sorted(THESIS.glob("3e_results_backup_*")):
        if not path.is_dir():
            continue
        meta_count = sum(1 for _ in path.rglob("run_meta.json"))
        structure_ok = (path / "run_3e.py").exists() and (path / "README.md").exists()
        if structure_ok and meta_count >= BACKUP_MIN_RUN_META:
            backups.append((path, meta_count))
    return backups


# ═════════════════════════ Small utilities ═════════════════════════
def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def driver_log(msg):
    line = f"{now_iso()}  {msg}"
    print(line, flush=True)
    try:
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        with open(DRIVER_LOG, "a") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def git_commit():
    try:
        r = subprocess.run(
            ["git", "-C", str(THESIS), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def parse_toml_frozen(config=None):
    """Parse a router config, including v3 cost coefficients."""
    data = tomllib.loads((ROUTER_DIR / (config or ROUTER_CFG)).read_text())
    replicas = data.get("replicas", [])
    names = [replica.get("name") for replica in replicas]
    weights = [replica.get("weight") for replica in replicas]
    scalar_keys = (
        "poll_interval_ms",
        "poll_timeout_ms",
        "dispatch_eviction_threshold",
        "max_state_age_ms",
        "connect_timeout_ms",
        "upstream_timeout_ms",
        "stream_idle_timeout_ms",
    )
    scalars = {key: data.get(key) for key in scalar_keys}
    coefficients = {
        replica.get("name"): {
            "c0": replica.get("c0"),
            "c1": replica.get("c1"),
            "c2": replica.get("c2"),
        }
        for replica in replicas
        if replica.get("name")
    }
    return names, weights, scalars, coefficients


def coefficients_match(actual, expected):
    for name, expected_values in expected.items():
        actual_values = actual.get(name, {})
        for key, expected_value in expected_values.items():
            value = actual_values.get(key)
            if not isinstance(value, (int, float)) or abs(value - expected_value) > 1e-9:
                return False
    return set(actual) == set(expected)


def preflight(strict, arm=None):
    """Local-only checks (no cluster contact). Returns (ok, lines).

    `arm` selects which router config is gated and against which weights. The
    default (None) reproduces the original behaviour exactly: router_3nodes.toml
    checked against FROZEN weights 3.1/1.0/1.0.
    """
    spec = arm_spec(arm) if arm else {"config": ROUTER_CFG,
                                      "weights": list(FROZEN["weights"])}
    cfg_name = spec["config"]
    want_weights = spec["weights"]
    lines = []
    ok = True

    def check(cond, msg):
        nonlocal ok
        lines.append(("PASS" if cond else "FAIL") + "  " + msg)
        ok = ok and cond
        return cond

    check(ROUTER_BIN.exists(), f"router binary present: {ROUTER_BIN}")
    check((ROUTER_DIR / cfg_name).exists(), f"router config present: {ROUTER_DIR/cfg_name}")
    check(BENCH_PY.exists(), f"bench venv python present: {BENCH_PY}")
    check((BENCH_DIR / BENCH_SCRIPT).exists(), f"bench script present: {BENCH_DIR/BENCH_SCRIPT}")
    check(BIMODAL_DATASET.exists(), f"bimodal dataset present: {BIMODAL_DATASET}")
    RESULTS_DIR.mkdir(exist_ok=True)
    check(RESULTS_DIR.exists(), f"results dir present: {RESULTS_DIR}")
    backups = verified_backups()
    if backups:
        backup, meta_count = backups[-1]
        check(True, f"verified backup exists: {backup} ({meta_count} run_meta.json files)")
    else:
        check(False, f"verified 3e_results backup with >= {BACKUP_MIN_RUN_META} run metadata files")

    # Refuse to run unless the config has the three replicas and weights 3.1/1.0/1.0.
    try:
        names, weights, scalars, coefficients = parse_toml_frozen(cfg_name)
        cfg_ok = names == ["pi-node-1", "pi-node-2", "pi-node-3"]
        w_ok = (
            len(weights) == 3
            and all(isinstance(weight, (int, float)) for weight in weights)
            and all(abs(weight - expected) < 1e-9
                    for weight, expected in zip(weights, want_weights))
        )
        s_ok = all(scalars.get(key) == FROZEN[key] for key in scalars)
        c_ok = coefficients_match(coefficients, FROZEN["cost_coefficients"])
        want_str = "/".join(str(w) for w in want_weights)
        check(cfg_ok, f"config has 3 replicas [pi-node-1/2/3]: found {names}")
        check(w_ok, f"{cfg_name} weights == {want_str}: found {weights}")
        check(s_ok, f"config tuning scalars == frozen: found {scalars}")
        check(c_ok, f"config c0/c1/c2 coefficients == frozen: found {coefficients}")
    except Exception as e:  # noqa: BLE001
        check(False, f"could not parse {cfg_name}: {e}")

    if strict and not ok:
        lines.append("")
        lines.append("Pre-flight FAILED. Fix the above (or pass --force to override) before --run.")
    return ok, lines


# ═════════════════════════ Cluster interaction (only in --run) ═════════════════════════
def health_ok(ip):
    try:
        r = subprocess.run(
            ["curl", "--max-time", "3", "-s", f"http://{ip}:8080/health"],
            capture_output=True, text=True, timeout=6,
        )
        return '"status"' in r.stdout and "ok" in r.stdout
    except Exception:
        return False


def can_ssh(ip):
    try:
        r = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=6", f"{SSH_USER}@{ip}", "true"],
            capture_output=True, timeout=15,
        )
        return r.returncode == 0
    except Exception:
        return False


def ssh_preflight():
    """Verify the driver's non-interactive SSH contract on every node."""
    lines = []
    ok = True
    for name, ip, _ in NODES:
        node_ok = can_ssh(ip)
        lines.append(
            ("PASS" if node_ok else "FAIL")
            + f"  {name} ({ip}) key-based SSH (BatchMode=yes)"
        )
        ok = ok and node_ok
    if not ok:
        lines.append("")
        lines.append("SSH FAILURE: recovery cannot run unattended until key-based SSH works.")
    return ok, lines


def ping_ok(ip):
    try:
        r = subprocess.run(["ping", "-c", "1", "-t", "2", ip], capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


def ssh_exec(ip, remote_cmd, timeout=40):
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", f"{SSH_USER}@{ip}", remote_cmd],
        capture_output=True, text=True, timeout=timeout,
    )


def node_temperature_c(ip):
    """Read the CPU thermal-zone temperature over SSH. Returns (temp_c, error)."""
    try:
        r = ssh_exec(ip, "cat /sys/class/thermal/thermal_zone0/temp", timeout=15)
    except Exception as e:  # noqa: BLE001
        return None, str(e)
    if r.returncode != 0:
        return None, (r.stderr or f"ssh exited {r.returncode}").strip()
    try:
        raw = float(r.stdout.strip().splitlines()[-1])
        temp_c = raw / 1000.0 if abs(raw) > 1000 else raw
    except (ValueError, IndexError):
        return None, f"unexpected thermal reading: {r.stdout!r}"
    if not 0.0 <= temp_c <= 150.0:
        return None, f"implausible thermal reading: {temp_c:.1f} C"
    return temp_c, None


def wait_for_thermal_ready():
    """Wait until all nodes are cool for consecutive readings.

    Returns (ok, details, reason). The details are persisted in run_meta.json so
    every measured cell records its thermal starting condition.
    """
    started = time.monotonic()
    deadline = started + THERMAL_WAIT_TIMEOUT_S
    initial = {}
    final = {}
    stable = 0
    samples = 0
    last_log_at = None

    while True:
        samples += 1
        temps = {}
        errors = {}
        for name, ip, _ in NODES:
            temp_c, error = node_temperature_c(ip)
            if error:
                errors[name] = error
            else:
                temps[name] = round(temp_c, 2)
                initial.setdefault(name, round(temp_c, 2))
        final = temps

        elapsed = time.monotonic() - started
        all_ready = (
            len(temps) == len(NODES)
            and max(temps.values()) <= THERMAL_MAX_C
        )
        stable = stable + 1 if all_ready else 0

        should_log = (
            samples == 1
            or stable > 0
            or last_log_at is None
            or elapsed - last_log_at >= 60
        )
        if should_log:
            readings = ", ".join(
                f"{name}={temps[name]:.1f}C" if name in temps else f"{name}=unavailable"
                for name, _, _ in NODES
            )
            suffix = (
                f" (stable {stable}/{THERMAL_STABLE_READINGS})"
                if all_ready else f" (waiting for <= {THERMAL_MAX_C:.1f}C)"
            )
            driver_log(f"  thermal gate: {readings}{suffix}")
            last_log_at = elapsed

        details = {
            "threshold_c": THERMAL_MAX_C,
            "stable_readings_required": THERMAL_STABLE_READINGS,
            "poll_interval_s": THERMAL_POLL_S,
            "timeout_s": THERMAL_WAIT_TIMEOUT_S,
            "wait_s": round(elapsed, 3),
            "samples": samples,
            "initial_c": initial,
            "final_c": final,
            "read_errors": errors,
        }
        if stable >= THERMAL_STABLE_READINGS:
            return True, details, None
        if time.monotonic() >= deadline:
            reason = (
                f"thermal gate timed out after {THERMAL_WAIT_TIMEOUT_S}s; "
                f"need every node <= {THERMAL_MAX_C:.1f}C"
            )
            return False, details, reason
        time.sleep(THERMAL_POLL_S)


def thermal_preflight():
    """Confirm every node exposes a readable CPU temperature sensor."""
    lines = []
    ok = True
    for name, ip, _ in NODES:
        temp_c, error = node_temperature_c(ip)
        if error:
            lines.append(f"FAIL  {name} ({ip}) temperature: {error}")
            ok = False
        else:
            lines.append(f"PASS  {name} ({ip}) temperature: {temp_c:.1f} C")
    return ok, lines


# Required build markers; checked against the running server, not the binary on disk.
BUILD_NUM = "9960"
BUILD_COMMIT = "a935fbffe"


def node_build_ok(ip):
    """Confirm the server ACTUALLY RUNNING on :8080 is build 9960/a935fbffe.

    1) HTTP /props — if its payload contains the build number or commit, done
       (no SSH needed).
    2) Fallback: SSH and read the binary path of the process bound to :8080 via
       /proc/<pid>/cmdline. A stale old-build server would show a different build
       directory here, which is exactly the silent-wrong-build case.
    Returns (ok, detail)."""
    body = ""
    try:
        r = subprocess.run(
            ["curl", "--max-time", "3", "-s", f"http://{ip}:8080/props"],
            capture_output=True, text=True, timeout=6,
        )
        body = r.stdout or ""
    except Exception:  # noqa: BLE001
        body = ""
    if BUILD_COMMIT in body or f"b{BUILD_NUM}" in body or BUILD_NUM in body:
        return True, f"/props reports build {BUILD_NUM}/{BUILD_COMMIT}"

    cmdline = ""
    try:
        remote = ("pid=$(ss -ltnp 2>/dev/null | grep ':8080 ' | grep -oP 'pid=\\K[0-9]+' | head -1); "
                  "if [ -n \"$pid\" ]; then tr '\\0' ' ' < /proc/$pid/cmdline; fi")
        rr = ssh_exec(ip, f'bash -lc "{remote}"', timeout=20)
        cmdline = (rr.stdout or "").strip()
    except Exception:  # noqa: BLE001
        cmdline = ""
    if f"llama.cpp-{BUILD_COMMIT}" in cmdline:
        running_bin = cmdline.split()[0] if cmdline else ""
        return True, f"running :8080 binary confirms build: {running_bin}"

    snippet = " ".join(body.split())[:160]
    return False, (f"could NOT confirm build {BUILD_NUM}/{BUILD_COMMIT} — "
                   f"/props='{snippet or '(no response)'}' running_cmdline='{cmdline[:160] or '(none)'}'")


def build_preflight():
    """Query all nodes' running build. Cluster-contacting — only for --run,
    --preflight, --single-cell. Returns (ok, lines)."""
    lines = []
    ok = True
    for name, ip, _ in NODES:
        node_ok, detail = node_build_ok(ip)
        lines.append(("PASS" if node_ok else "FAIL") + f"  {name} ({ip}) build: {detail}")
        ok = ok and node_ok
    if not ok:
        lines.append("")
        lines.append("BUILD MISMATCH: at least one node is not running 9960/a935fbffe. A wrong build "
                     "silently invalidates the sweep. Fix the node(s) above before proceeding.")
    return ok, lines


def restart_llama(name, ip, n):
    """Kill whatever holds :8080, relaunch with the active server configuration
    (the frozen command by default), wait for /health. Returns True on success."""
    driver_log(f"    restart {name} ({ip}): killing :8080 and relaunching")
    # No passwordless sudo on the nodes: llama-server runs as the SSH user, so
    # ss shows its PID and a plain kill suffices (graceful, then forced).
    pids = "$(ss -ltnp 2>/dev/null | grep ':8080 ' | grep -oP 'pid=\\K[0-9]+' | sort -u)"
    kill = (f"for p in {pids}; do kill $p; done; sleep 3; "
            f"for p in {pids}; do kill -9 $p; done; sleep 1; true")
    try:
        ssh_exec(ip, kill)
    except Exception as e:  # noqa: BLE001
        driver_log(f"    restart {name}: kill step errored: {e}")
    time.sleep(2)
    launch = (
        f"nohup {LLAMA_BIN} {llama_server_args()} "
        f"> llama_node{n}_a935fbffe.log 2>&1 & disown"
    )
    try:
        ssh_exec(ip, f'bash -lc "{launch}"')
    except Exception as e:  # noqa: BLE001
        driver_log(f"    restart {name}: launch step errored: {e}")
        return False
    deadline = time.time() + COLD_START_TIMEOUT_S
    while time.time() < deadline:
        if health_ok(ip):
            driver_log(f"    restart {name}: /health ok")
            return True
        time.sleep(3)
    driver_log(f"    restart {name}: still unhealthy after {COLD_START_TIMEOUT_S}s")
    return False


def ensure_nodes_healthy(down):
    """Health-gate all nodes. `down` is a persistent set of node names already
    ruled dead/unrecoverable (skipped without retry). Returns (ok, reason)."""
    for name, ip, n in NODES:
        if name in down:
            return False, f"{name} previously marked unrecoverable/dead"
        if health_ok(ip):
            continue
        # /health failed — distinguish hung (recoverable) from dead.
        reachable = can_ssh(ip)
        if reachable:
            driver_log(f"  {name} /health FAILED but SSH answers -> hung; restarting (max {MAX_RESTARTS})")
            recovered = False
            for attempt in range(1, MAX_RESTARTS + 1):
                driver_log(f"  {name} restart attempt {attempt}/{MAX_RESTARTS}")
                if restart_llama(name, ip, n):
                    recovered = True
                    break
            if not recovered:
                down.add(name)
                return False, f"{name} unrecoverable after {MAX_RESTARTS} restart attempts"
        else:
            alive = ping_ok(ip)
            down.add(name)
            how = "pings but no SSH (cannot restart)" if alive else "no ping, no SSH (genuinely dead)"
            driver_log(f"  {name} /health FAILED and {how} -> marking down, will skip cells needing it")
            return False, f"{name} unreachable: {how}"
    return True, None


def running_server_cmdline(ip):
    """Command line of the process bound to :8080 on a node ('' if none).

    Sent straight to the remote login shell (no nested `bash -lc "..."`), so
    $pid is expanded on the node after it is set. The nested form expanded it
    early to an empty string and read /proc//cmdline, i.e. the kernel's boot
    parameters instead of the server's command line."""
    remote = ("pid=$(ss -ltnp 2>/dev/null | grep ':8080 ' | grep -oP 'pid=\\K[0-9]+' | head -1); "
              "if [ -n \"$pid\" ]; then tr '\\0' ' ' < /proc/$pid/cmdline; fi")
    try:
        rr = ssh_exec(ip, remote, timeout=20)
        return (rr.stdout or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def node_server_config_ok(ip, cfg=None):
    """Confirm the RUNNING server matches cfg (model, ctx, slots, KV mode).

    Reads the live process's command line, so a node still serving a previous
    configuration is caught. Returns (ok, detail)."""
    cfg = cfg or SERVER_CONFIG
    tokens = running_server_cmdline(ip).split()
    if not tokens:
        return False, "no process bound to :8080"

    def value(flag):
        return tokens[tokens.index(flag) + 1] if flag in tokens[:-1] else None

    expected = {"--model": cfg["model_path"], "--ctx-size": str(cfg["ctx_size"]),
                "-np": str(cfg["slots"])}
    wrong = [f"{flag}={value(flag)!r} (want {want!r})"
             for flag, want in expected.items() if value(flag) != want]
    kv_flag = KV_FLAGS[cfg["kv_mode"]]
    other_flags = [f for mode, f in KV_FLAGS.items() if mode != cfg["kv_mode"]]
    if kv_flag not in tokens or any(f in tokens for f in other_flags):
        wrong.append(f"KV flag (want {kv_flag})")
    if wrong:
        return False, "running server differs: " + "; ".join(wrong)
    return True, f"running {cfg['name']}: {os.path.basename(cfg['model_path'])}, " \
                 f"ctx {cfg['ctx_size']}, -np {cfg['slots']}, {kv_flag}"


def apply_server_config(cfg):
    """Switch every node to cfg and verify it. Cluster-contacting (--run only).

    Nodes already running cfg are left alone; others are restarted (up to
    MAX_RESTARTS) with cfg. Returns (ok, lines)."""
    set_server_config(cfg)
    lines = []
    ok = True
    for name, ip, n in NODES:
        match, detail = node_server_config_ok(ip, cfg)
        if not (match and health_ok(ip)):
            driver_log(f"  {name}: switching to {cfg['name']} ({detail})")
            for attempt in range(1, MAX_RESTARTS + 1):
                if restart_llama(name, ip, n):
                    break
                driver_log(f"  {name}: restart attempt {attempt}/{MAX_RESTARTS} failed")
            match, detail = node_server_config_ok(ip, cfg)
        node_ok = match and health_ok(ip)
        lines.append(("PASS" if node_ok else "FAIL") + f"  {name} ({ip}): {detail}")
        ok = ok and node_ok
    return ok, lines


def start_router(policy, log_path, console_path, config=None):
    console = open(console_path, "w")
    proc = subprocess.Popen(
        [str(ROUTER_BIN), "--config", config or ROUTER_CFG, "--policy", policy,
         "--log", str(log_path), "--heartbeat-secs", str(HEARTBEAT_SECS)],
        cwd=str(ROUTER_DIR), stdout=console, stderr=subprocess.STDOUT,
    )
    return proc, console


def wait_router_eligible(console_path, replica_names):
    """Poll the router console (heartbeat lines, STDERR) until every replica
    reports elig=1, or timeout."""
    deadline = time.time() + ELIGIBLE_TIMEOUT_S
    while time.time() < deadline:
        try:
            text = Path(console_path).read_text()
        except OSError:
            text = ""
        elig = {}
        for m in re.finditer(r"\[hb\] (\S+) elig=(\d)", text):
            elig[m.group(1)] = m.group(2) == "1"
        if replica_names and all(elig.get(nm, False) for nm in replica_names):
            return True
        time.sleep(1)
    return False


def stop_router(proc, console):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
    try:
        console.flush()
        console.close()
    except Exception:  # noqa: BLE001
        pass


def run_bench(workload, conc, console_path):
    args = [str(BENCH_PY), BENCH_SCRIPT,
            "--base-url", "http://127.0.0.1:3000", "--model", SERVER_CONFIG["model_name"]]
    args += WORKLOAD_ARGS[workload]
    args += ["--num-prompts", "100", "--max-concurrency", str(conc),
             "--request-rate", "2", "--seed", "42", "--save-result"]
    with open(console_path, "w") as bc:
        try:
            r = subprocess.run(args, cwd=str(BENCH_DIR), stdout=bc,
                               stderr=subprocess.STDOUT, timeout=BENCH_TIMEOUT_S)
            return r.returncode, None
        except subprocess.TimeoutExpired:
            return None, f"bench timed out after {BENCH_TIMEOUT_S}s"
        except Exception as e:  # noqa: BLE001
            return None, f"bench errored: {e}"


# ═════════════════════════ Verdict ═════════════════════════
def compute_verdict(router_log_path):
    """Parse the moved router JSONL: drop reachability_event lines, group request
    records (one run_id per run), and apply the clean/suspect gate."""
    records, events = [], []
    try:
        with open(router_log_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("type") == "reachability_event":
                    events.append(r)
                else:
                    records.append(r)
    except OSError as e:
        return "suspect", f"router log unreadable: {e}", {}, []

    run_ids = sorted({r.get("run_id") for r in records if r.get("run_id") is not None})
    n = len(records)
    no_healthy = sum(1 for r in records if r.get("outcome") == "no_healthy_replica")
    n_fail = sum(1 for r in records if r.get("outcome") != "success")
    fail_rate = (n_fail / n) if n else 1.0
    evictions = sum(1 for e in events if e.get("action") == "evicted")

    stale_dist, chosen = {}, {}
    for r in records:
        m = r.get("stale_mode", "unknown")
        stale_dist[m] = stale_dist.get(m, 0) + 1
        c = r.get("chosen_replica")
        if c is not None:
            chosen[c] = chosen.get(c, 0) + 1

    reasons = []
    if n == 0:
        reasons.append("no request records in router log")
    if no_healthy > 0:
        reasons.append(f"no_healthy_replica={no_healthy}")
    if fail_rate > FAILURE_RATE_MAX:
        reasons.append(f"failure_rate={fail_rate:.1%} > {FAILURE_RATE_MAX:.0%}")
    if evictions > 0:
        reasons.append(f"mid-run eviction events={evictions}")
    if len(run_ids) > 1:
        reasons.append(f"multiple run_ids in one run log: {run_ids}")

    verdict = "clean" if not reasons else "suspect"
    metrics = {
        "run_ids": run_ids,
        "router_request_records": n,
        "no_healthy_replica": no_healthy,
        "failure_rate": round(fail_rate, 4),
        "eviction_events": evictions,
        "stale_mode_distribution": stale_dist,
        "chosen_replica_counts": chosen,
    }
    return verdict, ("; ".join(reasons) if reasons else None), metrics, events


def bench_summary(bench_json_path):
    try:
        d = json.loads(Path(bench_json_path).read_text())
    except Exception:  # noqa: BLE001
        return {}
    return {k: d.get(k) for k in
            ("num_prompts", "num_warmup", "completed", "failed", "outcome_counts",
             "throughput_req_s", "duration_s")}


def write_meta(cell, rd, verdict, reason, start_ts, end_ts, commit,
               bench_file, router_file, metrics, bench_meta, events,
               thermal_gate=None):
    spec = arm_spec(arm_label(cell))
    # Record the weights actually loaded (an arm may use different weights).
    frozen = dict(FROZEN)
    frozen["weights"] = spec["weights"]
    frozen["router_config_file"] = spec["config"]
    frozen["model"] = os.path.basename(SERVER_CONFIG["model_path"])
    meta = {
        "policy": cell["policy"],
        "arm": arm_label(cell),
        "arm_note": spec.get("note"),
        "concurrency": cell["conc"],
        "workload": cell["workload"],
        "rep": cell["rep"],
        "extension_group": cell.get("group") or cell_group(
            cell["policy"], cell["conc"], cell["workload"]
        ),
        "policy_metadata": policy_metadata(cell["policy"]),
        "experimental_predictive_policy": cell["policy"] == "cost_predict_v3",
        "start_ts": start_ts,
        "end_ts": end_ts,
        "router_git_commit": commit,
        "frozen_config": frozen,
        "bench_result_file": bench_file,
        "router_log_file": router_file,
        "verdict": verdict,
        "verdict_reason": reason,
        "router_metrics": metrics,
        "bench_summary": bench_meta,
        "reachability_events": events,
        "thermal_gate": thermal_gate,
        "server_config": dict(SERVER_CONFIG),
    }
    (rd / "run_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


# ═════════════════════════ One cell ═════════════════════════
def existing_verdict(rd):
    mp = rd / "run_meta.json"
    if not mp.exists():
        return None
    try:
        return json.loads(mp.read_text()).get("verdict")
    except Exception:  # noqa: BLE001
        return None


def create_rep_dir_exclusive(rd):
    """Create a new append-only rep directory; never reuse an old number."""
    rd.parent.mkdir(parents=True, exist_ok=True)
    try:
        rd.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(
            f"refusing to overwrite existing run directory: {rd}; "
            "the scheduler must allocate a higher rep number"
        ) from exc


def run_cell(cell, commit, down, replica_names, cooldown=True):
    rd = rep_dir(cell)
    tag = f"{cell['policy']} {block_name(cell['conc'], cell['workload'])} rep{cell['rep']}"
    start_ts = now_iso()
    driver_log(f"RUN  {tag}")
    create_rep_dir_exclusive(rd)

    # 1–2. Health gate (+ hung recovery / dead skip).
    ok, reason = ensure_nodes_healthy(down)
    if not ok:
        driver_log(f"SKIP {tag}: {reason}")
        write_meta(cell, rd, "skipped", reason, start_ts, now_iso(), commit,
                   None, None, {}, {}, [])
        return "skipped"

    # 3. Standardize the thermal starting condition across every policy/cell.
    thermal_ok, thermal_gate, thermal_reason = wait_for_thermal_ready()
    if not thermal_ok:
        driver_log(f"SUSPECT {tag}: {thermal_reason}")
        write_meta(cell, rd, "suspect", thermal_reason, start_ts, now_iso(), commit,
                   None, None, {}, {}, [], thermal_gate)
        if cooldown:
            time.sleep(COOLDOWN_S)
        return "suspect"

    stage_log = ROUTER_DIR / f".3e_stage_{arm_label(cell)}_c{cell['conc']}_{cell['workload']}_rep{cell['rep']}.jsonl"
    if stage_log.exists():
        stage_log.unlink()
    router_console = rd / "router_console.log"
    bench_console = rd / "bench_console.log"

    # 4–5. Start router, wait for all-eligible.
    proc, console = start_router(cell["policy"], stage_log, router_console,
                                 config=arm_spec(arm_label(cell))["config"])
    try:
        router_eligible = wait_router_eligible(router_console, replica_names)
    except KeyboardInterrupt:
        stop_router(proc, console)
        if stage_log.exists():
            shutil.move(str(stage_log), str(rd / "router_log.jsonl"))
        driver_log(f"INTERRUPTED {tag}: preserved incomplete rep; resume will use a higher number")
        raise
    if not router_eligible:
        stop_router(proc, console)
        reason = f"router did not report all nodes eligible within {ELIGIBLE_TIMEOUT_S}s"
        driver_log(f"SUSPECT {tag}: {reason}")
        if stage_log.exists():
            shutil.move(str(stage_log), str(rd / "router_log.jsonl"))
        write_meta(cell, rd, "suspect", reason, start_ts, now_iso(), commit,
                   None, "router_log.jsonl" if (rd / "router_log.jsonl").exists() else None,
                   {}, {}, [], thermal_gate)
        if cooldown:
            time.sleep(COOLDOWN_S)
        return "suspect"

    # 6. Bench.
    before = set(glob.glob(str(RESULTS_DIR / "*.json")))
    try:
        rc, berr = run_bench(cell["workload"], cell["conc"], bench_console)
    except KeyboardInterrupt:
        stop_router(proc, console)
        if stage_log.exists():
            shutil.move(str(stage_log), str(rd / "router_log.jsonl"))
        driver_log(f"INTERRUPTED {tag}: preserved incomplete rep; resume will use a higher number")
        raise
    else:
        # 7. Stop router (before moving its log).
        stop_router(proc, console)

    # 6. Move router log + detect/move bench result.
    router_file = None
    if stage_log.exists():
        shutil.move(str(stage_log), str(rd / "router_log.jsonl"))
        router_file = "router_log.jsonl"

    bench_file = None
    after = set(glob.glob(str(RESULTS_DIR / "*.json")))
    new = [Path(p) for p in (after - before)]
    if new:
        src = max(new, key=lambda p: p.stat().st_mtime)
        shutil.move(str(src), str(rd / src.name))
        bench_file = src.name

    # 9. Verdict.
    if berr or rc not in (0, None):
        reason = berr or f"bench exited with code {rc}"
        if router_file:
            v, vr, metrics, events = compute_verdict(rd / "router_log.jsonl")
            reason = f"{reason}; router-log check: {vr or 'clean'}"
            metrics = metrics
        else:
            metrics, events = {}, []
        verdict = "suspect"
    elif bench_file is None:
        verdict, reason, metrics, events = "suspect", "bench produced no results JSON", {}, []
    elif router_file is None:
        verdict, reason, metrics, events = "suspect", "router produced no log", {}, []
    else:
        verdict, reason, metrics, events = compute_verdict(rd / "router_log.jsonl")

    bench_meta = bench_summary(rd / bench_file) if bench_file else {}
    write_meta(cell, rd, verdict, reason, start_ts, now_iso(), commit,
               bench_file, router_file, metrics, bench_meta, events, thermal_gate)
    driver_log(f"{verdict.upper():7s} {tag}"
               + (f": {reason}" if reason else "")
               + (f"  chosen={metrics.get('chosen_replica_counts')}" if metrics else ""))

    # 8. Cooldown (servers stay up).
    if cooldown:
        time.sleep(COOLDOWN_S)
    return verdict


# ═════════════════════════ Scaffold + README ═════════════════════════
def make_scaffold():
    """Create only missing parent folders; never create or alter a rep folder."""
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for policy, conc, workload in extension_specs():
        cell_root(policy, conc, workload).mkdir(parents=True, exist_ok=True)
    README.write_text(readme_text())


def folder_tree_str():
    lines = [
        "3e_results/",
        "  <each existing baseline policy>/",
        "    c8_bimodal/    rep1/ rep2/ rep3/  [preserved]  rep4+ [new infill]",
        "    c12_bimodal/   rep1/ rep2/ rep3/  [preserved]  rep4+ [new infill]",
    ]
    for policy in NEW_POLICIES:
        lines.append(f"  {policy}/")
        for workload, conc in blocks():
            lines.append(
                f"    {block_name(conc, workload)}/   rep4/ rep5/ rep6/ rep7/ rep8/"
            )
    lines.extend([
        "",
        "Each new rep contains: <bench>.json, router_log.jsonl, run_meta.json,",
        "router_console.log, and bench_console.log.",
    ])
    return "\n".join(lines)


def example_meta():
    return {
        "policy": "cost_predict_v3",
        "concurrency": 8,
        "workload": "bimodal",
        "rep": 4,
        "extension_group": "cost_predict_v3_full",
        "policy_metadata": policy_metadata("cost_predict_v3"),
        "experimental_predictive_policy": True,
        "start_ts": "2026-07-28T02:14:30Z",
        "end_ts": "2026-07-28T02:16:41Z",
        "router_git_commit": git_commit(),
        "frozen_config": FROZEN,
        "bench_result_file": "bimodal_rate2_c8_20260725_021441.json",
        "router_log_file": "router_log.jsonl",
        "verdict": "clean",
        "verdict_reason": None,
        "router_metrics": {
            "run_ids": ["1784900000000000000-51234"],
            "router_request_records": 103,
            "no_healthy_replica": 0,
            "failure_rate": 0.0,
            "eviction_events": 0,
            "stale_mode_distribution": {"live": 98, "some_stale_excluded": 5},
            "chosen_replica_counts": {"pi-node-1": 61, "pi-node-2": 21, "pi-node-3": 21},
        },
        "bench_summary": {
            "num_prompts": 100, "num_warmup": 3, "completed": 100, "failed": 0,
            "outcome_counts": {"ok": 100}, "throughput_req_s": 1.94, "duration_s": 53.2,
        },
        "reachability_events": [],
        "thermal_gate": {
            "threshold_c": THERMAL_MAX_C,
            "stable_readings_required": THERMAL_STABLE_READINGS,
            "wait_s": 10.1,
            "initial_c": {"pi-node-1": 54.2, "pi-node-2": 51.8, "pi-node-3": 52.4},
            "final_c": {"pi-node-1": 53.9, "pi-node-2": 51.6, "pi-node-3": 52.1},
        },
    }


def readme_text():
    orders = "\n".join(
        f"- rep{rep}: {', '.join(rotated(rep))}"
        for rep in range(EXTENSION_REP_START, EXTENSION_REP_START + INFILL_ATTEMPT_CAP)
    )
    return f"""# §3e benchmark extension

Append-only extension of the original six-policy §3e sweep.

## New work
- Infill the six baseline policies at bimodal c8/c12 to at least
  {INFILL_TARGET_CLEAN} clean reps, with at most {INFILL_ATTEMPT_CAP} new
  attempts per cell.
- Run `state_aware_v2` over the full 7-cell grid with
  {NEW_POLICY_TARGET_REPS} reps per cell.
- Run experimental predictive `cost_predict_v3` over the same full grid with
  {NEW_POLICY_TARGET_REPS} reps per cell.
- Existing runs are immutable. All extension attempts use rep4 or higher.

## Folder scheme
`3e_results/<policy>/c<concurrency>_<workload>/rep<n>/`

The scheduler inventories all rep directories before every allocation. It
counts existing clean baseline reps toward the infill target and creates only
the next unused rep number. Suspect, skipped, and interrupted attempts are
retained and never overwritten.

## Counterbalancing
The eight-policy order advances its first position by three per extension rep
and alternates direction. Because three and eight are coprime, rep4..rep11 each
put a different policy first; alternating direction also reverses the relative
order of v2 and v3 across reps:
{orders}

## Frozen config (router_3nodes.toml — NOT modified by the driver)
- weights: pi-node-1 = {FROZEN['weights'][0]} (Pi 5), pi-node-2 = {FROZEN['weights'][1]} (Pi 4B/4GB),
  pi-node-3 = {FROZEN['weights'][2]} (Pi 4B/8GB)
- poll_interval_ms = {FROZEN['poll_interval_ms']}, poll_timeout_ms = {FROZEN['poll_timeout_ms']}
- dispatch_eviction_threshold = {FROZEN['dispatch_eviction_threshold']}, max_state_age_ms = {FROZEN['max_state_age_ms']}, connect_timeout_ms = {FROZEN['connect_timeout_ms']}
- llama.cpp build {FROZEN['llama_build']}, model {FROZEN['model']}
- bench: num-prompts {FROZEN['num_prompts']}, request-rate {FROZEN['request_rate']}, seed {FROZEN['seed']}, 3 warmup requests
- timing: bench timeout {BENCH_TIMEOUT_S}s; minimum cooldown {COOLDOWN_S}s between runs
- thermal gate: every node must report ≤ {THERMAL_MAX_C:.1f}°C for
  {THERMAL_STABLE_READINGS} consecutive readings ({THERMAL_POLL_S}s apart) before a measured cell starts
- cost coefficients: {json.dumps(FROZEN['cost_coefficients'], sort_keys=True)}

## Calibration provenance
Weighted policy weights are static and calibrated from the per-node solo
throughput sweep. The v3 coefficients are the frozen Pi 5 R-only model and Pi
4B linear models recorded in `router_3nodes.toml`; the pre-flight requires an
exact match before the extension can run.

## Verdict (per run, in run_meta.json)
`clean` iff (from the router JSONL, reachability_event lines excluded, grouped by
run_id): `no_healthy_replica == 0` AND failure rate ≤ {int(FAILURE_RATE_MAX*100)}% AND no mid-run
eviction events. Otherwise `suspect` with the specific reason. `skipped` = a
node was down and the cell could not run. run_meta also records the stale_mode
distribution and chosen_replica counts.

## Running
- `python3 run_3e.py` prints the remaining append-only plan and stops.
- `python3 run_3e.py --preflight` validates backup, config, key-based SSH,
  running build, and temperature sensors.
- `caffeinate -dimsu python3 run_3e.py --run` executes only currently needed
  extension attempts. It never re-runs or overwrites rep1..3.
"""


# ═════════════════════════ Modes ═════════════════════════
def print_extension_plan():
    plan = remaining_plan()
    backups = verified_backups()
    print("\n--- Backup confirmation ---")
    if backups:
        path, count = backups[-1]
        print(f"  VERIFIED: {path} exists with {count} run_meta.json files")
    else:
        print("  MISSING: no verified backup; --run will fail pre-flight")

    print("\n--- New-run count ---")
    print(
        "  (1) baseline bimodal c8/c12 infill: "
        f"{plan['infill_minimum']} minimum new runs if all clean; "
        f"{plan['infill_maximum']} maximum at per-cell caps"
    )
    print(
        "  (2) state_aware_v2 full grid: "
        f"{plan['new_remaining']['state_aware_v2']} remaining "
        f"(target 7 cells x {NEW_POLICY_TARGET_REPS} reps = 35)"
    )
    print(
        "  (3) cost_predict_v3 full grid: "
        f"{plan['new_remaining']['cost_predict_v3']} remaining "
        f"(target 7 cells x {NEW_POLICY_TARGET_REPS} reps = 35)"
    )
    print(
        f"  TOTAL: {plan['minimum_total']} minimum, "
        f"{plan['maximum_total']} maximum new runs from current state"
    )

    print("\n--- Baseline infill inventory ---")
    for cell in plan["underpowered"]:
        print(
            f"  {cell['policy']:<27} c{cell['conc']}_bimodal: "
            f"clean={cell['clean']}/5, extension_attempts={cell['attempts']}/8, "
            f"next=rep{cell['next_rep']}, remaining={cell['minimum']}..{cell['maximum']}"
        )

    print("\n--- Append-only folder tree ---")
    print(folder_tree_str())
    print("\n--- Counterbalanced eight-policy order ---")
    for rep in range(EXTENSION_REP_START, EXTENSION_REP_START + INFILL_ATTEMPT_CAP):
        print(f"  rep{rep}: {rotated(rep)}")


def do_plan():
    make_scaffold()

    print("=" * 78)
    print("§3e EXTENSION — PLAN (no cluster contact, no benchmarks executed)")
    print("=" * 78)
    print_extension_plan()

    print("\n--- Pre-flight (local checks) ---")
    _, lines = preflight(strict=False)
    for ln in lines:
        print("  " + ln)

    print("\nNOTHING WAS LAUNCHED.")
    print("After reviewing the plan, launch manually with:")
    print("  caffeinate -dimsu python3 run_3e.py --run")


def pending_wave():
    pending = []
    for policy, conc, workload in extension_specs():
        if cell_needs_run(policy, conc, workload):
            pending.append({
                "policy": policy,
                "conc": conc,
                "workload": workload,
                "rep": next_rep_number(policy, conc, workload),
                "group": cell_group(policy, conc, workload),
            })
    if not pending:
        return []

    wave_rep = min(cell["rep"] for cell in pending)
    eligible = {
        (cell["policy"], cell["conc"], cell["workload"]): cell
        for cell in pending
        if cell["rep"] == wave_rep
    }
    ordered = []
    for workload, conc in blocks():
        for policy in rotated(wave_rep):
            key = (policy, conc, workload)
            if key in eligible:
                ordered.append(eligible[key])
    return ordered


def do_run(force):
    ok, lines = preflight(strict=True)
    print("--- Pre-flight (local) ---")
    for ln in lines:
        print("  " + ln)
    print("--- Pre-flight (key-based SSH) ---")
    sok, slines = ssh_preflight()
    for ln in slines:
        print("  " + ln)
    print("--- Pre-flight (node build version) ---")
    bok, blines = build_preflight()
    for ln in blines:
        print("  " + ln)
    print("--- Pre-flight (node temperature sensors) ---")
    tok, tlines = thermal_preflight()
    for ln in tlines:
        print("  " + ln)
    if not ok:
        print("\nAborting (local frozen-config or backup pre-flight failed).")
        sys.exit(1)
    if not (sok and bok and tok) and not force:
        print("\nAborting (pre-flight failed). Re-run with --force to override.")
        sys.exit(1)

    make_scaffold()
    print_extension_plan()
    commit = git_commit()
    replica_names, _, _, _ = parse_toml_frozen()
    down = set()
    attempts = []

    plan = remaining_plan()
    driver_log(
        f"=== §3e extension start: {plan['minimum_total']}..{plan['maximum_total']} "
        f"new runs currently needed, git {commit} ==="
    )
    for rep in range(EXTENSION_REP_START, EXTENSION_REP_START + INFILL_ATTEMPT_CAP):
        driver_log(f"rep{rep} counterbalanced policy order: {rotated(rep)}")

    while True:
        wave = pending_wave()
        if not wave:
            break
        stop_for_down_node = False
        for cell in wave:
            verdict = run_cell(cell, commit, down, replica_names)
            attempts.append((cell, verdict))
            if verdict == "skipped" and down:
                driver_log(
                    "STOP extension: every cell requires all three nodes; "
                    f"pending work remains resumable after recovery of {sorted(down)}"
                )
                stop_for_down_node = True
                break
        if stop_for_down_node:
            break

    print_summary(attempts, down)


def print_summary(attempts, down):
    print("\n" + "=" * 78)
    print("§3e EXTENSION SUMMARY")
    print("=" * 78)
    header = f"{'policy':<27} {'block':<14} {'rep':<4} verdict"
    print(header)
    print("-" * len(header))
    counts = {}
    attention = []
    for cell, v in attempts:
        counts[v] = counts.get(v, 0) + 1
        print(f"{cell['policy']:<27} {block_name(cell['conc'], cell['workload']):<14} "
              f"{cell['rep']:<4} {v}")
        if v not in ("clean",):
            attention.append(f"{cell['policy']} {block_name(cell['conc'], cell['workload'])} rep{cell['rep']} -> {v}")

    print("\nTotals: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if down:
        print(f"Nodes marked down during sweep: {sorted(down)}")
    if attention:
        print("\nCELLS NEEDING ATTENTION (suspect / skipped / not-run):")
        for a in attention:
            print("  " + a)
    else:
        print("\nNo new suspect or skipped attempts.")

    print("\nFinal target status:")
    for policy, conc, workload in extension_specs():
        status = cell_status(policy, conc, workload)
        group = cell_group(policy, conc, workload)
        if group == "baseline_bimodal_infill":
            target = (
                f"clean={status['clean_total']}/{INFILL_TARGET_CLEAN}, "
                f"new_attempts={status['extension_attempts']}/{INFILL_ATTEMPT_CAP}"
            )
        else:
            target = f"new_attempts={status['extension_attempts']}/{NEW_POLICY_TARGET_REPS}"
        print(f"  {policy:<27} {block_name(conc, workload):<14} {target}")
    driver_log("=== §3e extension end ===")


def do_preflight():
    """Local checks plus node build and thermal checks. Exits nonzero on failure."""
    print("--- Pre-flight (local) ---")
    ok, lines = preflight(strict=False)
    for ln in lines:
        print("  " + ln)
    print("--- Pre-flight (key-based SSH — contacts all 3 nodes) ---")
    sok, slines = ssh_preflight()
    for ln in slines:
        print("  " + ln)
    print("--- Pre-flight (node build version — contacts all 3 nodes) ---")
    bok, blines = build_preflight()
    for ln in blines:
        print("  " + ln)
    print("--- Pre-flight (node temperature sensors) ---")
    tok, tlines = thermal_preflight()
    for ln in tlines:
        print("  " + ln)
    allok = ok and sok and bok and tok
    print(f"\nPRE-FLIGHT {'PASS — safe to --run' if allok else 'FAIL — do not --run until fixed'}.")
    sys.exit(0 if allok else 1)


def do_single_cell(policy, conc, workload, rep, force, arm=None):
    """Run exactly one append-only extension cell with production machinery."""
    make_scaffold()
    default = {
        "policy": "state_aware_v2",
        "conc": 4,
        "workload": "uniform",
    }
    # --arm names a (policy, config) variant; --policy alone is the plain case.
    selected_arm = arm or policy or default["policy"]
    spec = arm_spec(selected_arm)
    selected_policy = spec["policy"]
    if arm and policy and policy != selected_policy:
        sys.exit(f"--arm {arm} implies --policy {selected_policy}, "
                 f"but --policy {policy} was given")
    selected_conc = conc or default["conc"]
    selected_workload = workload or default["workload"]
    cell = {
        "policy": selected_policy,
        "arm": selected_arm,
        "conc": selected_conc,
        "workload": selected_workload,
        "rep": rep or next_rep_number(selected_arm, selected_conc, selected_workload),
        "group": cell_group(selected_policy, selected_conc, selected_workload) or "manual_smoke",
    }
    if cell["policy"] not in POLICIES:
        sys.exit(f"invalid policy {cell['policy']}; choose from {POLICIES}")
    if cell["workload"] not in WORKLOAD_ARGS:
        sys.exit(f"invalid --workload {cell['workload']}; choose from {list(WORKLOAD_ARGS)}")

    print("--- Pre-flight (local) ---")
    ok, lines = preflight(strict=True, arm=selected_arm)
    for ln in lines:
        print("  " + ln)
    print("--- Pre-flight (key-based SSH) ---")
    sok, slines = ssh_preflight()
    for ln in slines:
        print("  " + ln)
    print("--- Pre-flight (node build version) ---")
    bok, blines = build_preflight()
    for ln in blines:
        print("  " + ln)
    print("--- Pre-flight (node temperature sensors) ---")
    tok, tlines = thermal_preflight()
    for ln in tlines:
        print("  " + ln)
    if not ok:
        sys.exit("\nAborting single-cell test (local frozen-config or backup pre-flight failed).")
    if not (sok and bok and tok) and not force:
        sys.exit("\nAborting single-cell test (pre-flight failed). Use --force to override.")

    commit = git_commit()
    replica_names, _, _, _ = parse_toml_frozen(spec["config"])
    rd = rep_dir(cell)
    tag = f"{arm_label(cell)} {block_name(cell['conc'], cell['workload'])} rep{cell['rep']}"
    driver_log(f"=== SINGLE-CELL test: {tag} (git {commit}, cfg {spec['config']}, "
               f"weights {spec['weights']}) ===")
    verdict = run_cell(cell, commit, set(), replica_names, cooldown=False)

    print("\n" + "=" * 78)
    print(f"SINGLE-CELL RESULT: {tag} -> {verdict}")
    print("=" * 78)
    print(f"Rep folder: {rd}")
    if rd.exists():
        for p in sorted(rd.iterdir()):
            print(f"  {p.name}  ({p.stat().st_size} bytes)")
    meta_path = rd / "run_meta.json"
    if meta_path.exists():
        print("\n--- run_meta.json ---")
        print(meta_path.read_text())
    print("\nINSPECT: verdict == 'clean'; run_meta.json has real router_metrics "
          "(record count ~103, failure_rate 0, chosen_replica_counts across all 3 nodes) "
          "and a bench_summary (completed 100, failed 0); the bench JSON and router_log.jsonl "
          "are present in the rep folder.")


def do_test_recovery():
    """Exercise the hung/dead detection + SSH-restart path once, on demand. Kill
    llama-server on one node first (see the printed procedure), then run this."""
    print("--- Recovery-path self-check ---")
    print("This runs the SAME code the per-run health gate uses "
          "(ensure_nodes_healthy -> restart_llama). Kill llama-server on one node")
    print("BEFORE running this so there is something to recover.\n")
    down = set()
    driver_log("=== TEST-RECOVERY: running health gate + recovery ===")
    ok, reason = ensure_nodes_healthy(down)
    print()
    if ok:
        print("RESULT: all nodes healthy (recovery succeeded if a node had been killed).")
    else:
        print(f"RESULT: gate returned NOT-OK: {reason}")
        if down:
            print(f"Nodes classified down (no restart attempted — dead/ssh-unreachable): {sorted(down)}")
    print("\nPost-check /health:")
    for name, ip, _ in NODES:
        print(f"  {name} ({ip}): {'ok' if health_ok(ip) else 'DOWN'}")
    driver_log("=== TEST-RECOVERY: done ===")


def main():
    ap = argparse.ArgumentParser(
        description="§3e sweep driver. Default = PLAN (offline). See the mode flags below.")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="execute the full sweep")
    mode.add_argument("--preflight", action="store_true",
                      help="local checks + node build/temperature checks, then exit")
    mode.add_argument("--single-cell", action="store_true",
                      help="run ONE real cell end-to-end (default: first cell), then exit")
    mode.add_argument("--test-recovery", action="store_true",
                      help="exercise the hung-node detection + SSH-restart path once")
    ap.add_argument("--force", action="store_true", help="proceed even if pre-flight fails")
    # single-cell selectors (optional; default to the first execution cell)
    ap.add_argument("--policy", choices=POLICIES, help="single-cell: policy")
    ap.add_argument("--arm", choices=sorted(set(ARMS) | set(POLICIES)),
                    default=None,
                    help="single-cell: a policy name, or a named (policy, "
                         "config) variant from ARMS. Results land in "
                         "3e_results/<arm>/ and run_meta records the arm's "
                         "real weights. Accepts plain policy names so a "
                         "multi-arm loop can use one flag throughout.")
    ap.add_argument("--conc", type=int, help="single-cell: concurrency")
    ap.add_argument("--workload", choices=list(WORKLOAD_ARGS), help="single-cell: workload")
    ap.add_argument("--rep", type=int, help="single-cell: rep number")
    args = ap.parse_args()

    if args.run:
        do_run(args.force)
    elif args.preflight:
        do_preflight()
    elif args.single_cell:
        do_single_cell(args.policy, args.conc, args.workload, args.rep, args.force,
                       arm=args.arm)
    elif args.test_recovery:
        do_test_recovery()
    else:
        do_plan()


if __name__ == "__main__":
    main()
