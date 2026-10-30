"""Generates the cost-model sweep cells (80% prompt, 20% output) and their manifest."""
import json
import random
from pathlib import Path

from bench import _BENCH_WORDS

OUT_DIR = Path(__file__).parent / "grid_cells"
OUT_DIR.mkdir(exist_ok=True)

SEED = 20260718  # fixed and documented -- each cell's prompt text is
                  # deterministic and reused across all 3 of its reps.

# (block, label, B, R_target, [(prompt_len, output_len), ...] one per slot, note)
CELLS = [
    ("backbone", "R200_B1", 1, 200, [(160, 40)], ""),
    ("backbone", "R200_B2", 2, 200, [(80, 20), (80, 20)], ""),
    ("backbone", "R200_B4", 4, 200, [(40, 10)] * 4,
     "LOW_ITL_GAPS: only ~9 ITL gaps/rep -- noisiest cell by design, do not drop"),
    ("backbone", "R800_B1", 1, 800, [(640, 160)], ""),
    ("backbone", "R800_B2", 2, 800, [(320, 80), (320, 80)], ""),
    ("backbone", "R800_B4", 4, 800, [(160, 40)] * 4, ""),
    ("backbone", "R1600_B2", 2, 1600, [(640, 160), (640, 160)], ""),
    ("backbone", "R1600_B4", 4, 1600, [(320, 80)] * 4, ""),
    ("backbone", "R3200_B4", 4, 3200, [(640, 160)] * 4, ""),
    ("asymmetric", "onebig", 4, 800, [(400, 100), (80, 20), (80, 20), (80, 20)], ""),
    ("asymmetric", "extreme", 4, 800, [(544, 136), (32, 8), (32, 8), (32, 8)], ""),
]


def make_prompt(rng, n_words):
    return " ".join(rng.choice(_BENCH_WORDS) for _ in range(n_words))


def main():
    rng = random.Random(SEED)
    manifest = []
    tsv_lines = []

    for block, label, B, r_target, slots, note in CELLS:
        assert len(slots) == B, f"{label}: {len(slots)} slot specs but B={B}"
        assert sum(p + o for p, o in slots) == r_target, f"{label}: slots don't sum to R_target"

        samples = [{"prompt": make_prompt(rng, p), "max_tokens": o} for p, o in slots]
        out_path = OUT_DIR / f"{block}_{label}.json"
        out_path.write_text(json.dumps(samples, indent=2))

        manifest.append({
            "block": block, "label": label, "B": B, "R_target": r_target,
            "slots": slots, "path": str(out_path), "note": note,
        })
        tsv_lines.append(f"{block}\t{label}\t{B}\t{r_target}\t{out_path}\t{note}")
        print(f"wrote {out_path}  (B={B}, R_target={r_target}, slots={slots})")

    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (OUT_DIR / "manifest.tsv").write_text("\n".join(tsv_lines) + "\n")
    print(f"\n{len(CELLS)} cell files + manifest.json + manifest.tsv written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
