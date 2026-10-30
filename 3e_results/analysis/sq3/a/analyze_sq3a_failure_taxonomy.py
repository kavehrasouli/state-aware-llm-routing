"""Classifies failed requests (deadline before/after first token, HTTP errors, disconnects)."""

import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA_ROOT = ROOT.parent  # real data root: 3e_results
sys.path.insert(0, str(ROOT))
import analyze_3e as A3
from scope import included_paths

CATEGORIES = [
    ("upstream deadline expiry, first token seen", "upstream_error", 200, True, True),
    ("upstream deadline expiry, no first token", "upstream_error", 200, False, True),
    ("no response headers within the timeout (504)", "upstream_timeout_headers", 504, False, True),
    ("immediate upstream connect failure (502)", "upstream_error", 502, False, True),
    ("client disconnect", "client_disconnect", 200, True, False),
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    durations = defaultdict(list)
    denominators = defaultdict(Counter)
    excluded = []
    warmups = 0
    bench_outcomes = Counter()
    seen = set()
    for path in included_paths():
        meta = json.loads(path.read_text())
        workload, concurrency = meta["workload"], meta["concurrency"]
        if not A3.is_grid_cell(workload, concurrency):
            excluded.append(str(path.relative_to(DATA_ROOT)))
            continue
        policy = meta.get("arm") or meta["policy"]
        cell = (policy, workload, concurrency)
        bench = json.loads((path.parent / meta["bench_result_file"]).read_text())
        raw = bench["raw"]
        offered = bench["num_prompts"]
        failed = sum(not r["success"] for r in raw)
        require(len(raw) == offered, f"Incomplete bench data: {path}")
        require(failed == bench["failed"] == meta["bench_summary"]["failed"],
                f"Driver failure mismatch: {path}")
        require(offered - failed == bench["completed"] == meta["bench_summary"]["completed"],
                f"Driver completion mismatch: {path}")
        bench_outcomes.update(r["outcome"] for r in raw if not r["success"])
        local = Counter()
        deadline = meta["frozen_config"]["upstream_timeout_ms"]
        for line in (path.parent / meta["router_log_file"]).read_text().splitlines():
            record = json.loads(line)
            if record.get("type") != "request":
                continue
            identity = (record["run_id"], record["request_id"])
            require(identity not in seen, f"Duplicate router request: {identity}")
            seen.add(identity)
            if record["request_id"] < bench["num_warmup"]:
                warmups += 1
                continue
            local["measured"] += 1
            if record["outcome"] == "success":
                local["success"] += 1
                continue
            first = record.get("ttft_ts_ns") is not None
            require(first == (record.get("ttft_ms_router_internal") is not None),
                    f"Inconsistent first-token fields: {identity}")
            key = (record["outcome"], record["http_status"], first)
            matches = [i for i, cat in enumerate(CATEGORIES) if cat[1:4] == key]
            require(len(matches) == 1, f"Unclassified outcome: {path}: {key}")
            category = matches[0]
            ms = record["total_ms"]
            require(math.isfinite(ms) and ms >= 0, f"Invalid duration: {identity}")
            if category in (0, 1):
                # Check the timing evidence behind the inferred deadline label.
                require(abs(ms - deadline) < 100, f"Stream error away from deadline: {identity}")
            if category == 3:
                require(ms < 10, f"502 is not immediate: {identity}")
            durations[cell, category].append(ms)
            local["failed" if CATEGORIES[category][4] else "disconnect"] += 1
        require(local["measured"] == offered, f"Router request-count mismatch: {path}")
        require(local["failed"] == failed, f"Router/driver failure mismatch: {path}")
        require(local["success"] + local["disconnect"] == offered - failed,
                f"Router/driver completion mismatch: {path}")
        denominators[cell].update(runs=1, offered=offered, failed=failed)

    rows = []

    def add_row(block, cell, category, values, denom):
        name, outcome, status, first, is_failure = category
        rows.append(dict(
            block=block, policy=cell[0], workload=cell[1], concurrency=cell[2],
            category=name, outcome=outcome, http_status=status, first_token_seen=first,
            counted_as_bench_failure=is_failure, count=len(values),
            median_total_ms=round(statistics.median(values), 6) if values else "",
            n_runs=denom["runs"], bench_offered=denom["offered"], bench_failed=denom["failed"],
            pct_of_bench_failures=round(100 * len(values) / denom["failed"], 6)
                if is_failure and denom["failed"] else "",
            pct_of_offered=round(100 * len(values) / denom["offered"], 6),
        ))

    cells = sorted(denominators, key=lambda c: (A3.POLICIES.index(c[0]), A3.CELL_COLS.index(c[1:])))
    for cell in cells:
        for i, category in enumerate(CATEGORIES):
            add_row("cell", cell, category, durations[cell, i], denominators[cell])
    total = sum(denominators.values(), Counter())
    all_failures = []
    for i, category in enumerate(CATEGORIES):
        values = [v for cell in cells for v in durations[cell, i]]
        add_row("grid_total", ("ALL", "ALL", "ALL"), category, values, total)
        if category[4]:
            all_failures.extend(values)
    require(len(all_failures) == total["failed"], "Grid failure reconciliation failed")
    add_row("grid_total", ("ALL", "ALL", "ALL"),
            ("bench_failures", "ALL_FAILURE_OUTCOMES", "", "", True), all_failures, total)
    output = HERE / "sq3a_failure_taxonomy.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output}: {len(cells)} cells, {len(rows)} rows")
    print(f"Reconciled every run: {dict(total)}; excluded {warmups} warmup requests")
    print(f"Off-grid runs excluded: {excluded}")
    print(f"Bench failure labels: {dict(bench_outcomes)}")
    for row in rows:
        if row["block"] == "grid_total":
            print(row["category"], row["count"], row["median_total_ms"],
                  row["pct_of_bench_failures"], row["pct_of_offered"])


if __name__ == "__main__":
    main()
