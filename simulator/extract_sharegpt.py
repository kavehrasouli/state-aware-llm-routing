import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(HERE, "sharegpt_lengths.json")


def bench_files():
    return sorted(glob.glob(os.path.join(
        REPO, "3e_results", "*", "c8_sharegpt", "rep*", "sharegpt_*.json"
    )))


def pairs_from(path):
    raw = json.loads(open(path).read()).get("raw", [])
    return [
        {"in": r["actual_prompt_len"], "out": r["requested_output_tokens"]}
        for r in raw
    ]


def clean(pairs):
    """bench.py writes actual_prompt_len = -1 for a request that never got a
    response, so runs with failures disagree on those slots. Only fully
    successful runs carry a complete length sequence."""
    return all(p["in"] > 0 for p in pairs)


def main():
    files = bench_files()
    if not files:
        raise SystemExit("no c8_sharegpt bench JSONs found under 3e_results/")
    complete = [f for f in files if clean(pairs_from(f))]
    if not complete:
        raise SystemExit("no c8_sharegpt run completed every request; "
                         "cannot recover a full length sequence")
    ref = pairs_from(complete[0])
    agree = sum(1 for f in complete if pairs_from(f) == ref)
    files = complete
    outs = [p["out"] for p in ref]
    ins = [p["in"] for p in ref]
    payload = {
        "_source": os.path.relpath(files[0], REPO),
        "_agreement": f"{agree}/{len(files)} logged sharegpt runs have an "
                      "identical length sequence (seed=42)",
        "n": len(ref),
        "in_tokens": {"min": min(ins), "max": max(ins),
                      "mean": round(sum(ins) / len(ins), 1)},
        "out_tokens": {"min": min(outs), "max": max(outs),
                       "mean": round(sum(outs) / len(outs), 1)},
        "pairs": ref,
    }
    with open(OUT, "w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    print(f"wrote {os.path.relpath(OUT, REPO)}: n={len(ref)}, "
          f"{agree}/{len(files)} runs agree")
    print(f"  in  tokens: {payload['in_tokens']}")
    print(f"  out tokens: {payload['out_tokens']}")
    if agree != len(files):
        print("  NOTE: not all runs agree -- check seeds before trusting the "
              "sharegpt cell.")


if __name__ == "__main__":
    main()
