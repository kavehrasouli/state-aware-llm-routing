"""Where the slowest 5% of requests ran, and how often the Pi 5 had all slots busy (descriptive).

Output: sq3b_tail_node_composition.csv
"""
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMBINED = HERE.parent.parent
RESULTS = COMBINED.parent
INCLUSION = COMBINED / "run_inclusion.csv"
OUT = HERE / "sq3b_tail_node_composition.csv"

CELLS = [("sharegpt", "12")]
CELL_CODE = {"uniform": "uni", "bimodal": "bim", "sharegpt": "sgpt"}
ARMS = ("weighted_least_connections", "state_aware")
NODES = ("pi-node-1", "pi-node-2", "pi-node-3")
SLOTS = 4          # -np 4 per server
TAIL = 0.05


def retained_runs(arm, workload, conc):
    with INCLUSION.open(newline="") as f:
        for r in csv.DictReader(f):
            if (r["arm"] == arm and r["workload"] == workload and r["concurrency"] == conc
                    and r["included"] == "True" and r["verdict"] == "clean"):
                yield (RESULTS / r["metadata_path"]).parent


def measured_requests(run_dir):
    meta = json.loads((run_dir / "run_meta.json").read_text())
    num_warmup = meta.get("num_warmup", 3)
    reqs = []
    with (run_dir / "router_log.jsonl").open() as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("type") == "request" and isinstance(rec.get("request_id"), int) \
                    and rec["request_id"] >= num_warmup:
                reqs.append(rec)
    return reqs


def summarize(arm, workload, conc):
    sent = defaultdict(int)
    full = defaultdict(int)
    tail = defaultdict(int)
    e2e = defaultdict(list)
    ttft = defaultdict(list)
    n_runs = n_tail = 0
    for run_dir in retained_runs(arm, workload, conc):
        n_runs += 1
        reqs = measured_requests(run_dir)
        for r in reqs:
            node = r["chosen_replica"]
            sent[node] += 1
            state = {s["replica"]: s for s in r["state_vector"]}[node]
            if state["b_slots_active"] >= SLOTS:
                full[node] += 1
        ok = [r for r in reqs if r["outcome"] == "success"]
        for r in ok:
            e2e[r["chosen_replica"]].append(r["total_ms"] / 1000)
            ttft[r["chosen_replica"]].append(r["ttft_ms_router_internal"] / 1000)
        k = max(1, round(TAIL * len(ok)))
        for r in sorted(ok, key=lambda r: -r["total_ms"])[:k]:
            tail[r["chosen_replica"]] += 1
            n_tail += 1
    total = sum(sent.values())
    rows = []
    for node in NODES:
        rows.append({
            "arm": arm, "cell": f"c{conc}_{CELL_CODE[workload]}", "node": node,
            "n_runs": n_runs, "requests": sent[node],
            "share": round(sent[node] / total, 6),
            "sent_to_full_node": round(full[node] / sent[node], 6) if sent[node] else "",
            "median_e2el_s": round(statistics.median(e2e[node]), 3) if e2e[node] else "",
            "median_ttft_s": round(statistics.median(ttft[node]), 3) if ttft[node] else "",
            "tail_requests": tail[node], "tail_total": n_tail,
            "tail_share": round(tail[node] / n_tail, 6),
        })
    return rows


def main():
    rows = [row for workload, conc in CELLS for arm in ARMS
            for row in summarize(arm, workload, conc)]
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {OUT.name}: {len(rows)} rows.")
    for r in rows:
        print(f"  {r['arm']:27} {r['cell']:9} {r['node']}: share {100 * r['share']:5.1f}%, "
              f"full-node dispatch {100 * float(r['sent_to_full_node'] or 0):5.1f}%, "
              f"tail {r['tail_requests']}/{r['tail_total']}")


if __name__ == "__main__":
    main()
