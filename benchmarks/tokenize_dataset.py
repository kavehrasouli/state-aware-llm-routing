"""Adds exact prompt token counts to a dataset using a llama-server's /tokenize endpoint.

Usage: python tokenize_dataset.py --server URL --input IN.json --output OUT.json
"""

import argparse
import json
import urllib.request
from pathlib import Path


def count_tokens(server: str, prompt: str) -> int:
    req = urllib.request.Request(
        f"{server.rstrip('/')}/tokenize",
        data=json.dumps({"content": prompt, "add_special": True}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return len(json.load(resp)["tokens"])


def main():
    p = argparse.ArgumentParser(description="Add prompt_tokens to a local dataset via /tokenize.")
    p.add_argument("--server", required=True, help="llama-server base URL (same model as the experiment)")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    data = json.loads(args.input.read_text())
    for item in data:
        if not isinstance(item, dict):
            raise SystemExit("dataset items must be objects with a 'prompt' field")
        item["prompt_tokens"] = count_tokens(args.server, item["prompt"])

    args.output.write_text(json.dumps(data, indent=2))
    counts = sorted(item["prompt_tokens"] for item in data)
    print(f"wrote {args.output}: {len(data)} prompts, tokens min={counts[0]} max={counts[-1]}")


if __name__ == "__main__":
    main()
