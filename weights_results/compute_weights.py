"""Recomputes the capacity weights from the calibration runs (two repetitions per node).

Usage: python compute_weights.py
"""

import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
CONFIGURED_PI5_WEIGHT = 3.1  # weight used in all experiments
NODES = {"pi-node-1": "Pi 5 (8 GB)", "pi-node-2": "Pi 4B (4 GB)", "pi-node-3": "Pi 4B (8 GB)"}


def main():
    capacity = {}
    for node, label in NODES.items():
        runs = sorted((HERE / node).glob("*.json"))
        values = []
        for path in runs:
            result = json.loads(path.read_text())
            values.append(result["throughput_req_s"])
            print(f"{node} {label:13} {path.name}  completed {result['completed']}/"
                  f"{result['num_prompts']}  {result['throughput_req_s']:.4f} req/s")
        capacity[node] = mean(values)
        print(f"{node} {label:13} mean over {len(values)} runs: {capacity[node]:.4f} req/s\n")

    pi4b = mean([capacity["pi-node-2"], capacity["pi-node-3"]])
    ratio = capacity["pi-node-1"] / pi4b
    print(f"Pi 5 / mean Pi 4B capacity: {ratio:.2f}")
    print(f"Configured weights (router_3nodes.toml): {CONFIGURED_PI5_WEIGHT} : 1.0 : 1.0")


if __name__ == "__main__":
    main()
