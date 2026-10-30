import json
import os
import random

from cluster import Request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
BIMODAL_JSON = os.path.join(
    REPO, "benchmarks", "local_datasets", "bimodal_50short_50long.json"
)
SHAREGPT_CACHE = os.path.join(HERE, "sharegpt_lengths.json")

# Tokenised prompt length for the bimodal classes. The dataset stores prompts
# as whitespace-joined filler words; the server reports 40 and 640 prompt
# tokens for the short and long classes respectively (actual_prompt_len in the
# logged runs), which is what matters for prefill cost.
BIMODAL_TOKENS = {"short": 40, "long": 640}


def _arrivals(n, rate, rng):
    """bench.py: cumulative expovariate gaps, first request at t=0."""
    out, t = [], 0.0
    for i in range(n):
        if i > 0:
            t += rng.expovariate(rate)
        out.append(t)
    return out


def uniform(n=100, rate=2.0, seed=42, input_len=128, output_len=128):
    rng = random.Random(seed)
    return [
        Request(i + 1, at, input_len, output_len, "uniform")
        for i, at in enumerate(_arrivals(n, rate, rng))
    ]


def bimodal(n=100, rate=2.0, seed=42):
    rng = random.Random(seed)
    spec = json.loads(open(BIMODAL_JSON).read())
    if n != len(spec):
        # keep the 50/50 mix if the caller asks for a different count
        spec = [spec[i % len(spec)] for i in range(n)]
    arrivals = _arrivals(n, rate, rng)
    return [
        Request(i + 1, at, BIMODAL_TOKENS[item["class"]],
                item["max_tokens"], item["class"])
        for i, (at, item) in enumerate(zip(arrivals, spec))
    ]


def sharegpt(n=100, rate=2.0, seed=42):
    if not os.path.exists(SHAREGPT_CACHE):
        raise SystemExit(
            f"{SHAREGPT_CACHE} missing -- run "
            "`python3 simulator/extract_sharegpt.py` to recover the (in, out) "
            "pairs from the logged 3e sharegpt runs."
        )
    pairs = json.loads(open(SHAREGPT_CACHE).read())["pairs"]
    rng = random.Random(seed)
    if n != len(pairs):
        pairs = [pairs[i % len(pairs)] for i in range(n)]
    arrivals = _arrivals(n, rate, rng)
    return [
        Request(i + 1, at, int(p["in"]), int(p["out"]), "sharegpt")
        for i, (at, p) in enumerate(zip(arrivals, pairs))
    ]


BUILDERS = {"uniform": uniform, "bimodal": bimodal, "sharegpt": sharegpt}


def build(workload, n=100, rate=2.0, seed=42):
    if workload not in BUILDERS:
        raise SystemExit(f"unknown workload {workload!r}; "
                         f"choose from {sorted(BUILDERS)}")
    return BUILDERS[workload](n=n, rate=rate, seed=seed)
