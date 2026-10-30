"""Generates the bimodal dataset: 50 short (40/10) and 50 long (640/160) requests."""

import argparse
import json
import random
from pathlib import Path

from bench import _BENCH_WORDS


def make_prompt(rng: random.Random, n_words: int) -> str:
    return " ".join(rng.choice(_BENCH_WORDS) for _ in range(n_words))


def parse_args():
    p = argparse.ArgumentParser(description="Generate synthetic bimodal benchmark JSON.")
    p.add_argument("--num-prompts", type=int, default=100)
    p.add_argument("--long-fraction", type=float, default=0.5)
    p.add_argument("--short-input-len", type=int, default=40)
    p.add_argument("--short-output-len", type=int, default=10)
    p.add_argument("--long-input-len", type=int, default=640)
    p.add_argument("--long-output-len", type=int, default=160)
    p.add_argument("--seed", type=int, default=20260723)
    p.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "local_datasets" / "bimodal_50short_50long.json",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if not 0.0 <= args.long_fraction <= 1.0:
        raise SystemExit("--long-fraction must be between 0 and 1")

    n_long = round(args.num_prompts * args.long_fraction)
    n_short = args.num_prompts - n_long
    rng = random.Random(args.seed)

    samples = []
    for _ in range(n_short):
        samples.append({
            "class": "short",
            "prompt": make_prompt(rng, args.short_input_len),
            "max_tokens": args.short_output_len,
        })
    for _ in range(n_long):
        samples.append({
            "class": "long",
            "prompt": make_prompt(rng, args.long_input_len),
            "max_tokens": args.long_output_len,
        })

    rng.shuffle(samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(samples, indent=2))

    print(f"wrote {args.output}")
    print(f"requests: total={len(samples)} short={n_short} long={n_long}")
    print(
        "short: "
        f"prompt_words={args.short_input_len} max_tokens={args.short_output_len} "
        f"approx_total={args.short_input_len + args.short_output_len}"
    )
    print(
        "long:  "
        f"prompt_words={args.long_input_len} max_tokens={args.long_output_len} "
        f"approx_total={args.long_input_len + args.long_output_len}"
    )


if __name__ == "__main__":
    main()
