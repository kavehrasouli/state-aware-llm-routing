"""Compares router-tracked prompt and output tokens with the engine's own counts.

Usage: python validate_tracking.py path/to/router_log.jsonl [...]
"""

import argparse
import json
import statistics
from pathlib import Path


def load_requests(paths):
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("type") == "request":
                yield rec


def summarize(name, diffs):
    if not diffs:
        print(f"{name}: no data")
        return
    exact = sum(d == 0 for d in diffs)
    absd = sorted(abs(d) for d in diffs)
    p95 = absd[min(len(absd) - 1, int(0.95 * len(absd)))]
    print(
        f"{name}: n={len(diffs)} exact={exact} ({100 * exact / len(diffs):.1f}%) "
        f"median diff={statistics.median(diffs):+g} p95 |diff|={p95:g} max |diff|={absd[-1]:g}"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("logs", nargs="+")
    p.add_argument("--budget", type=float, default=4096.0,
                   help="per-replica context budget c, to turn polled u back into tokens")
    args = p.parse_args()

    recs = list(load_requests(args.logs))
    tracked = [r for r in recs if r.get("tracked_prompt_source") is not None]
    print(f"{len(recs)} request records, {len(tracked)} with tracked fields")
    sources = {}
    for r in tracked:
        sources[r["tracked_prompt_source"]] = sources.get(r["tracked_prompt_source"], 0) + 1
    print(f"prompt token sources: {sources}")

    ok = [r for r in tracked if r.get("outcome") == "success"]
    summarize(
        "prompt tokens (tracked - usage)",
        [r["tracked_prompt_tokens"] - r["usage_prompt_tokens"]
         for r in ok if r.get("usage_prompt_tokens") is not None],
    )
    summarize(
        "output tokens (tracked - usage)",
        [r["tracked_output_tokens"] - r["usage_completion_tokens"]
         for r in ok if r.get("usage_completion_tokens") is not None],
    )

    tok_diffs, req_diffs = [], []
    for r in tracked:
        for sv in r.get("state_vector", []):
            if sv.get("stale") or sv.get("tracked_tokens") is None:
                continue
            tok_diffs.append(sv["tracked_tokens"] - round(sv["u"] * args.budget))
            req_diffs.append(sv["tracked_requests"] - (sv["b_metrics"] + sv["requests_deferred"]))
    summarize("decision-time tokens (tracked - polled u*c), fresh only", tok_diffs)
    summarize("decision-time requests (tracked - polled processing+deferred), fresh only", req_diffs)


if __name__ == "__main__":
    main()
