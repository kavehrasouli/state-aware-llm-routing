import argparse
import glob
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
NODE_DIRS = {
    "pi-node-1": "cost_model_results_pi-node-1_Pi5-8GB",
    "pi-node-2": "cost_model_results_pi-node-2_4B-4g",
    "pi-node-3": "cost_model_results_pi-node-3_4B-8g",
}
OUT = os.path.join(HERE, "service_model.json")


def sweep_dir(node):
    return os.path.join(REPO, "benchmarks", NODE_DIRS[node])


CACHE_HIT_FACTOR = 2.0   # implied rate this far above median => prompt-cache hit


def prefill_samples(node):
    """[(total_prompt_tokens, ttft_s, label)] over the controlled sweep."""
    out = []
    for path in sorted(glob.glob(os.path.join(sweep_dir(node), "raw_results", "*.json"))):
        d = json.loads(open(path).read())
        raw = [r for r in d.get("raw", []) if r.get("success")]
        if not raw:
            continue
        conc = d.get("max_concurrency") or 1
        plen = d.get("actual_prompt_len", {}).get("mean")
        if not plen:
            continue
        ttft_s = st.median(r["ttft_ms"] for r in raw) / 1000.0
        if ttft_s > 0:
            # a -np>=conc batch prefills together, so the node cleared
            # conc*plen prompt tokens over one ttft.
            out.append((conc * plen, ttft_s, os.path.basename(path)))
    return out


def prefill_rate(node, verbose=False):
    """Aggregate prompt tokens/s, as a proportional model ttft = tokens / rate.

    llama.cpp reuses cached prompt prefixes across repeated identical prompts,
    which shows up as isolated runs an order of magnitude faster than siblings
    with the same (concurrency, prompt_len). Those are dropped: they measure
    the cache, not the hardware.
    """
    samples = prefill_samples(node)
    if not samples:
        raise SystemExit(f"no prefill samples for {node}")
    rough = st.median(tok / t for tok, t, _ in samples)
    kept = [s for s in samples if (s[0] / s[1]) <= CACHE_HIT_FACTOR * rough]
    dropped = [s for s in samples if s not in kept]
    rate = st.median(tok / t for tok, t, _ in kept)
    if verbose:
        rates = sorted(tok / t for tok, t, _ in kept)
        resid = [100 * ((tok / rate) - t) / t for tok, t, _ in kept]
        worst = max(resid, key=abs)
        print(f"  {node} prefill: {rate:.1f} tok/s from n={len(kept)} "
              f"(range [{rates[0]:.1f}, {rates[-1]:.1f}]), "
              f"worst residual {worst:+.0f}%")
        for tok, t, name in dropped:
            print(f"    dropped cache hit: {name} -> {tok / t:.0f} tok/s "
                  f"({tok:.0f} tok in {t:.2f}s)")
    return rate


def decode_grid(node, verbose=False):
    """{B: {R: tpot_ms}} from the backbone cells, averaged over reps."""
    import csv
    path = os.path.join(sweep_dir(node), "sweep_cost_model_summary.csv")
    cells = {}
    for row in csv.DictReader(open(path)):
        if row.get("block") != "backbone":
            continue
        # rep-1 of the Pi 5 'extreme' cell is a documented 10x outlier, but it
        # lives in the asymmetric block and is already excluded by the filter.
        key = (int(row["B"]), int(float(row["R_actual"])))
        cells.setdefault(key, []).append(float(row["tpot_mean"]))
    grid = {}
    for (b, r), vals in sorted(cells.items()):
        grid.setdefault(str(b), {})[str(r)] = round(st.mean(vals), 3)
    if verbose:
        for b in sorted(grid, key=int):
            pts = ", ".join(f"R={r}:{grid[b][r]:.0f}ms" for r in sorted(grid[b], key=int))
            print(f"  {node} decode B={b}: {pts}")
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    model = {
        "_source": "benchmarks/cost_model_results_*/ (controlled sweep, 3 reps/cell)",
        "_note": "prefill_rate_tok_s is aggregate across the prefill batch; "
                 "decode_tpot is per-request ms/token at batch depth B, context R.",
        "nodes": {},
    }
    for node in NODE_DIRS:
        if args.verbose:
            print(f"{node}:")
        model["nodes"][node] = {
            "prefill_rate_tok_s": round(prefill_rate(node, args.verbose), 2),
            "decode_tpot": decode_grid(node, args.verbose),
        }
    with open(OUT, "w") as fh:
        json.dump(model, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"\nwrote {os.path.relpath(OUT, REPO)}")
    for node, m in model["nodes"].items():
        print(f"  {node}: prefill {m['prefill_rate_tok_s']} tok/s, "
              f"{sum(len(v) for v in m['decode_tpot'].values())} decode cells")


if __name__ == "__main__":
    main()
