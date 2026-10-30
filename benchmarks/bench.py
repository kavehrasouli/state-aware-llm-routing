"""Streaming benchmark client for llama-server: Poisson arrivals, fixed seed, per-request timings.

TTFT is measured from HTTP dispatch; time waiting under the concurrency limit is recorded separately.
"""

import argparse
import asyncio
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import aiohttp
import numpy as np

# Short common words (~1 token each) give a controlled prompt length for any tokenizer.
_BENCH_WORDS = [
    "the", "a", "is", "of", "and", "to", "in", "that", "it", "with",
    "for", "on", "are", "as", "at", "be", "this", "from", "or", "an",
    "by", "not", "but", "what", "all", "were", "when", "we", "there",
    "can", "your", "has", "more", "one", "if", "out", "so", "up", "no",
]


# --------------------------------------------------------------------------- #
# Request / result dataclasses
# --------------------------------------------------------------------------- #

@dataclass
class Sample:
    prompt: str
    max_tokens: int
    request_class: str | None = None


@dataclass
class RequestResult:
    request_class: str | None # optional workload class label from local datasets
    prompt_len: int           # word count of sent prompt (approximate token count)
    actual_prompt_len: int    # usage.prompt_tokens from server response; -1 if unavailable
    output_len: int
    ttft: float               # seconds to first token, measured from actual HTTP dispatch
    tpot: float               # mean seconds per output token after first
    itl: list                 # inter-token latencies in seconds
    e2el: float               # end-to-end latency in seconds, measured from HTTP dispatch
    queue_delay: float        # seconds client waited on semaphore before dispatch
    success: bool
    requested_output_tokens: int = 0    # max_tokens asked for
    generated_output_tokens: int = 0    # usage.completion_tokens; falls back to counted stream tokens
    finish_reason: str = ""             # OpenAI finish_reason from the final chunk ("stop" / "length" / "")
    truncated: int = 0                  # 1 iff finish_reason=="length" AND generated < requested — see send_request
    outcome: str = "success"            # "success" | "timeout" | "connection_error" | "http_error" | "empty_response" | "error"
    error: str = ""



# Dataset helpers
def make_random_samples(num_prompts, input_len, output_len, seed):
    """Return list of Sample objects using random common words.

    Prompt length is approximate (~input_len tokens); actual tokenized length
    is captured from usage.prompt_tokens in the server response.
    """
    rng = random.Random(seed)
    samples = []
    for _ in range(num_prompts):
        words = [rng.choice(_BENCH_WORDS) for _ in range(input_len)]
        samples.append(Sample(" ".join(words), output_len))
    return samples


def make_sharegpt_samples(num_prompts, seed, max_input_len=400, max_output_len=400):
    """Return list of Sample objects from ShareGPT52K."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError(
            "ShareGPT requires: pip install datasets\n"
            "Or use --dataset-name oasst1."
        )

    print("Loading ShareGPT52K from HuggingFace (cached after first run)…")
    try:
        ds = load_dataset("RyokoAI/ShareGPT52K", split="train")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load ShareGPT52K: {exc}\n"
            "Dataset may be gated or removed.\n"
            "Use --dataset-name oasst1, or supply --dataset-path <file.json>."
        ) from exc

    rng = random.Random(seed)
    rows = list(ds)
    rng.shuffle(rows)

    samples = []
    for row in rows:
        if len(samples) >= num_prompts:
            break
        turns = row.get("conversations", [])
        human = [t["value"] for t in turns if t.get("from") == "human"]
        gpt   = [t["value"] for t in turns if t.get("from") == "gpt"]
        if not human or not gpt:
            continue
        prompt_len_est = len(human[0]) // 4
        output_len_est = len(gpt[0])   // 4
        if not (4 <= prompt_len_est <= max_input_len):
            continue
        output_len = max(1, min(output_len_est, max_output_len))
        samples.append(Sample(human[0], output_len))

    if not samples:
        raise RuntimeError(
            "Zero usable ShareGPT samples after filtering.\n"
            "Try --dataset-name oasst1 or --dataset-path <file.json>."
        )
    if len(samples) < num_prompts:
        print(f"Warning: only {len(samples)} usable ShareGPT samples (wanted {num_prompts})")
    return samples[:num_prompts]


def make_oasst1_samples(num_prompts, seed, output_len, max_input_len=400):
    """Return Sample objects from OpenAssistant/oasst1 (matches Nezami dataset).

    output_len is a fixed controlled value (from --oasst-output-len), not derived from
    the prompt text. This keeps oasst1 runs comparable to Random sweeps.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("oasst1 requires: pip install datasets")

    print("Loading OpenAssistant/oasst1 from HuggingFace (cached after first run)…")
    try:
        ds = load_dataset("OpenAssistant/oasst1", split="train")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load oasst1: {exc}\n"
            "Supply --dataset-path <file.json> instead."
        ) from exc

    rng = random.Random(seed)
    rows = [r for r in ds if r.get("role") == "prompter"]
    rng.shuffle(rows)

    samples = []
    for row in rows:
        if len(samples) >= num_prompts:
            break
        text = row.get("text", "")
        prompt_len_est = len(text) // 4
        if not (4 <= prompt_len_est <= max_input_len):
            continue
        samples.append(Sample(text, output_len))  # fixed output_len, not derived from prompt

    if not samples:
        raise RuntimeError("Zero usable oasst1 samples after filtering.")
    if len(samples) < num_prompts:
        print(f"Warning: only {len(samples)} usable oasst1 samples (wanted {num_prompts})")
    return samples[:num_prompts]


def make_local_samples(path, num_prompts, seed, max_output_len=128):
    """Load from a local JSON file: list of str, or list of {prompt, max_tokens, class?}."""
    data = json.loads(Path(path).read_text())
    rng = random.Random(seed)
    rng.shuffle(data)
    samples = []
    for item in data[:num_prompts]:
        if isinstance(item, str):
            samples.append(Sample(item, max_output_len))
        elif isinstance(item, dict):
            cls = item.get("class")
            samples.append(Sample(
                item["prompt"],
                item.get("max_tokens", max_output_len),
                str(cls) if cls is not None else None,
            ))
    if not samples:
        raise RuntimeError(f"Zero samples loaded from {path}")
    return samples


# Single streaming request
async def send_request(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    request_class: str | None,
    t_send: float,       # perf_counter() at HTTP dispatch (after semaphore acquisition)
    queue_delay: float,  # seconds spent waiting on semaphore before dispatch
    ignore_eos: bool,    # True forces generation to run to max_tokens (feasibility/stress sweeps);
                         # False lets the model stop naturally (real experiments)
) -> RequestResult:
    url = f"{base_url}/v1/completions"
    payload = {
        "model":          model,
        "prompt":         prompt,
        "max_tokens":     max_tokens,
        "temperature":    0.0,
        "stream":         True,
        "ignore_eos":     ignore_eos,
        "cache_prompt":   False,  # reuse would skip prefill and distort TTFT
        "stream_options": {"include_usage": True},   # capture actual prompt/completion token counts
    }

    token_times: list = []
    output_tokens = 0
    actual_prompt_len = -1
    generated_output_tokens = -1
    finish_reason = ""

    def _fail(outcome: str, error: str) -> RequestResult:
        # Every failure path returns here, so timeouts, connection errors and HTTP errors stay distinct.
        return RequestResult(
            request_class=request_class,
            prompt_len=len(prompt.split()), actual_prompt_len=-1, output_len=0,
            ttft=0.0, tpot=0.0, itl=[], e2el=0.0, queue_delay=queue_delay,
            success=False, requested_output_tokens=max_tokens, generated_output_tokens=0,
            outcome=outcome, error=error,
        )

    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                body = await resp.text()
                return _fail("http_error", f"HTTP {resp.status}: {body[:200]}")

            async for raw in resp.content:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Usage appears in the last data chunk when stream_options.include_usage is set
                if chunk.get("usage"):
                    actual_prompt_len = chunk["usage"].get("prompt_tokens", -1)
                    generated_output_tokens = chunk["usage"].get("completion_tokens", -1)

                choices = chunk.get("choices", [])
                if not choices:
                    continue
                fr = choices[0].get("finish_reason")
                if fr:
                    finish_reason = fr
                text = choices[0].get("text", "")
                if text:
                    token_times.append(time.perf_counter())
                    output_tokens += 1

    except asyncio.TimeoutError:
        return _fail("timeout", "request exceeded client-side timeout")
    except aiohttp.ClientError as exc:
        return _fail("connection_error", str(exc))
    except Exception as exc:
        return _fail("error", str(exc))

    if not token_times:
        return _fail("empty_response", "no tokens received")

    if generated_output_tokens < 0:
        generated_output_tokens = output_tokens  # usage missing from server; fall back to counted stream tokens

    # finish_reason "length" with fewer tokens than requested means the slot's context limit cut generation short.
    truncated = 1 if (finish_reason == "length" and generated_output_tokens < max_tokens) else 0

    t_end  = time.perf_counter()
    ttft   = token_times[0] - t_send   # from actual dispatch, not intended arrival
    e2el   = t_end - t_send
    itl    = [token_times[i] - token_times[i - 1] for i in range(1, len(token_times))]
    # Denominator is the server's token count, as in vLLM's serving benchmark. The
    # streamed-chunk count (output_tokens) undercounts: llama.cpp merges tokens of
    # incomplete multi-byte text into one chunk and sends EOS with no text.
    tpot   = (e2el - ttft) / (generated_output_tokens - 1) if generated_output_tokens > 1 else 0.0

    return RequestResult(
        request_class=request_class,
        prompt_len=len(prompt.split()),
        actual_prompt_len=actual_prompt_len,
        output_len=output_tokens,
        ttft=ttft,
        tpot=tpot,
        itl=itl,
        e2el=e2el,
        queue_delay=queue_delay,
        success=True,
        requested_output_tokens=max_tokens,
        generated_output_tokens=generated_output_tokens,
        finish_reason=finish_reason,
        truncated=truncated,
        outcome="success",
    )


# Benchmark loop — Poisson arrivals + optional concurrency cap
async def run_benchmark(samples, base_url, model, request_rate, max_concurrency, seed, ignore_eos, request_timeout):
    arrival_rng = random.Random(seed + 1)
    sem = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    connector = aiohttp.TCPConnector(limit=0)
    timeout   = aiohttp.ClientTimeout(total=request_timeout)
    tasks     = []

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:

        async def do_request(sample, intended_arrival):
            # t_send is recorded after semaphore acquisition so that
            # queue_delay = t_send - intended_arrival captures all client-side
            # backpressure and never contaminates TTFT.
            if sem:
                async with sem:
                    t_send = time.perf_counter()
                    return await send_request(
                        session, base_url, model, sample.prompt, sample.max_tokens,
                        sample.request_class,
                        t_send, t_send - intended_arrival, ignore_eos,
                    )
            t_send = time.perf_counter()
            return await send_request(
                session, base_url, model, sample.prompt, sample.max_tokens,
                sample.request_class,
                t_send, t_send - intended_arrival, ignore_eos,
            )

        t0 = time.perf_counter()

        for i, sample in enumerate(samples):
            if request_rate != float("inf") and i > 0:
                await asyncio.sleep(arrival_rng.expovariate(request_rate))
            intended_arrival = time.perf_counter()
            tasks.append(asyncio.create_task(do_request(sample, intended_arrival)))

        results = list(await asyncio.gather(*tasks))
        t1 = time.perf_counter()

    return results, t0, t1


# Statistics + output
def _pct(vals, p):
    return float(np.percentile(vals, p)) if vals else 0.0

def _mean(vals):
    return float(np.mean(vals)) if vals else 0.0


def compute_summary(results, duration, args):
    ok     = [r for r in results if r.success]
    failed = len(results) - len(ok)

    outcome_counts = {}
    for r in results:
        outcome_counts[r.outcome] = outcome_counts.get(r.outcome, 0) + 1

    # Three outcomes among successes — only kv_slot_truncated disqualifies a config.
    finish_reason_breakdown = {
        "stop_normal":       sum(1 for r in ok if r.truncated == 0 and r.finish_reason == "stop"),
        "length_expected":   sum(1 for r in ok if r.truncated == 0 and r.finish_reason == "length"),
        "kv_slot_truncated": sum(1 for r in ok if r.truncated == 1),
    }

    ttfts        = [r.ttft        * 1000 for r in ok]
    tpots        = [r.tpot        * 1000 for r in ok]
    itls         = [x             * 1000 for r in ok for x in r.itl]
    e2els        = [r.e2el        * 1000 for r in ok]
    queue_delays = [r.queue_delay * 1000 for r in ok]

    total_out_tokens = sum(r.generated_output_tokens for r in ok)  # server count, not chunks

    apl = [r.actual_prompt_len for r in ok if r.actual_prompt_len >= 0]
    actual_prompt_len_stats = {
        "min":       int(min(apl))               if apl else -1,
        "mean":      round(float(np.mean(apl)), 1) if apl else -1,
        "max":       int(max(apl))               if apl else -1,
        "n_missing": len(ok) - len(apl),         # requests where server didn't return usage
    }

    def stats(vals):
        return {
            "mean_ms": round(_mean(vals), 3),
            "p50_ms":  round(_pct(vals, 50), 3),
            "p95_ms":  round(_pct(vals, 95), 3),
        }

    class_metrics = {}
    class_labels = sorted({r.request_class for r in results if r.request_class is not None})
    for cls in class_labels:
        cls_results = [r for r in results if r.request_class == cls]
        cls_ok = [r for r in cls_results if r.success]
        cls_outcomes = {}
        for r in cls_results:
            cls_outcomes[r.outcome] = cls_outcomes.get(r.outcome, 0) + 1
        class_metrics[cls] = {
            "total": len(cls_results),
            "completed": len(cls_ok),
            "failed": len(cls_results) - len(cls_ok),
            "outcome_counts": cls_outcomes,
            "ttft": stats([r.ttft * 1000 for r in cls_ok]),
            "tpot": stats([r.tpot * 1000 for r in cls_ok]),
            "e2el": stats([r.e2el * 1000 for r in cls_ok]),
        }

    summary = {
        "benchmark_version":  "bench.py/2.0",
        "prompt_len_note":    "actual_prompt_len is the real tokenized input length (usage.prompt_tokens from server); use it for analysis, not prompt_len (word-count estimate)",
        "date":               datetime.now().isoformat(),
        "base_url":           args.base_url,
        "model":              args.model,
        "dataset_name":       args.dataset_name,
        "random_input_len":   args.random_input_len,
        "random_output_len":  args.random_output_len,
        "num_prompts":        args.num_prompts,
        "num_warmup":         args.num_warmup,
        "request_rate":       args.request_rate,
        "max_concurrency":    args.max_concurrency,
        "load_mode":          args.load_mode,
        "seed":               args.seed,
        "duration_s":         round(duration, 3),
        "completed":          len(ok),
        "failed":             failed,
        "outcome_counts":     outcome_counts,
        "finish_reason_breakdown": finish_reason_breakdown,
        "throughput_req_s":   round(len(ok)           / duration, 4),
        "throughput_tok_s":   round(total_out_tokens  / duration, 4),
        "ttft":               stats(ttfts),
        "tpot":               stats(tpots),
        "itl":                stats(itls),
        "e2el":               stats(e2els),
        "queue_delay":        stats(queue_delays),
        "actual_prompt_len":  actual_prompt_len_stats,
    }
    if class_metrics:
        summary["class_metrics"] = class_metrics
    return summary


def print_summary(s):
    qd = s["queue_delay"]
    bottleneck = qd["mean_ms"] > 50
    load_mode = s.get("load_mode", "open-loop")

    print(f"\n{'='*62}")
    print(f"  dataset={s['dataset_name']}  rate={s['request_rate']} req/s  "
          f"concurrency={s['max_concurrency']}  mode={load_mode}")
    print(f"  completed={s['completed']}/{s['num_prompts']}  "
          f"failed={s['failed']}  duration={s['duration_s']:.1f}s")
    if s["failed"]:
        breakdown = ", ".join(f"{k}={v}" for k, v in s["outcome_counts"].items() if k != "success")
        print(f"  failure breakdown: {breakdown}")
    print(f"  throughput: {s['throughput_req_s']:.3f} req/s  "
          f"{s['throughput_tok_s']:.1f} tok/s")
    print(f"{'─'*62}")
    print(f"  {'metric':<12}  {'mean':>9}  {'p50':>9}  {'p95':>9}  (ms)")
    print(f"  {'─'*47}")
    for key in ("ttft", "tpot", "itl", "e2el"):
        d = s[key]
        print(f"  {key:<12}  {d['mean_ms']:>9.1f}  {d['p50_ms']:>9.1f}  {d['p95_ms']:>9.1f}")
    if bottleneck and load_mode == "open-loop":
        qd_note = "  *** CLIENT BOTTLENECK"
    elif bottleneck and load_mode == "closed-loop":
        qd_note = "  (closed-loop backpressure)"
    else:
        qd_note = ""
    print(f"  {'queue_delay':<12}  {qd['mean_ms']:>9.1f}  {qd['p50_ms']:>9.1f}  {qd['p95_ms']:>9.1f}{qd_note}")
    apl = s["actual_prompt_len"]
    if apl["n_missing"] == 0:
        apl_note = f"min={apl['min']}  mean={apl['mean']}  max={apl['max']}  (tokens, from server usage)"
    else:
        apl_note = f"min={apl['min']}  mean={apl['mean']}  max={apl['max']}  ({apl['n_missing']} missing — usage not returned)"
    print(f"  actual_input_len: {apl_note}")
    frb = s["finish_reason_breakdown"]
    print(f"  finish_reason:    stop={frb['stop_normal']}  "
          f"length(expected)={frb['length_expected']}  "
          f"KV-SLOT-TRUNCATED={frb['kv_slot_truncated']}")

    class_metrics = s.get("class_metrics")
    if class_metrics:
        print(f"{'─'*62}")
        print("  per-class success metrics")
        print(f"  {'class':<10}  {'done':>9}  {'fail':>5}  {'metric':<6}  {'mean':>9}  {'p50':>9}  {'p95':>9}  (ms)")
        print(f"  {'─'*66}")
        for cls in sorted(class_metrics):
            cm = class_metrics[cls]
            done = f"{cm['completed']}/{cm['total']}"
            for i, key in enumerate(("ttft", "tpot", "e2el")):
                d = cm[key]
                cls_label = cls if i == 0 else ""
                done_label = done if i == 0 else ""
                fail_label = str(cm["failed"]) if i == 0 else ""
                print(f"  {cls_label:<10}  {done_label:>9}  {fail_label:>5}  "
                      f"{key:<6}  {d['mean_ms']:>9.1f}  {d['p50_ms']:>9.1f}  {d['p95_ms']:>9.1f}")
    print(f"{'='*62}\n")

    if frb["kv_slot_truncated"] > 0:
        print(f"*** CONFIG INVALID: {frb['kv_slot_truncated']} request(s) hit the KV slot ceiling")
        print("    (finish_reason=length but stopped short of requested max_tokens) —")
        print("    this config truncates under this workload and is disqualified.\n")

    if bottleneck and load_mode == "open-loop":
        print("WARNING: mean queue_delay > 50 ms — the client semaphore is the bottleneck.")
        print("         TTFT/E2EL are measured after dispatch and do not include queue_delay.")
        print("         Interpret this as capped-arrival load, not a pure open-loop rate run.")
        print("         Lower --request-rate or raise --max-concurrency to remove client backpressure.\n")
    elif bottleneck and load_mode == "closed-loop":
        print("INFO: mean queue_delay > 50 ms — expected for closed-loop fixed-concurrency load.")
        print("      TTFT/E2EL are measured after dispatch and do not include queue_delay.\n")


def result_path(args) -> Path:
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.dataset_name == "random":
        tag = (f"random_in{args.random_input_len}_out{args.random_output_len}"
               f"_rate{args.request_rate}_c{args.max_concurrency}_{ts}")
    else:
        tag = f"{args.dataset_name}_rate{args.request_rate}_c{args.max_concurrency}_{ts}"
    return out_dir / f"{tag}.json"


# CLI
def parse_args():
    p = argparse.ArgumentParser(
        description="Streaming benchmark client for llama-server: Poisson arrivals, TTFT/TPOT/ITL/E2EL.",
    )
    p.add_argument("--base-url",          required=True,
                   help="e.g. http://192.168.2.2:8080")
    p.add_argument("--model",             required=True,
                   help="model name or alias as configured in llama-server")
    p.add_argument("--dataset-name",      default="random",
                   choices=["random", "sharegpt", "oasst1", "local"])
    p.add_argument("--dataset-path",      default=None,
                   help="path to local JSON file (required for --dataset-name local)")
    p.add_argument("--random-input-len",  type=int, default=128)
    p.add_argument("--random-output-len", type=int, default=128)
    p.add_argument("--oasst-output-len",  type=int, default=128,
                   help="fixed output token target for oasst1 runs (ignored for other datasets)")
    p.add_argument("--num-prompts",       type=int, default=10)
    p.add_argument("--request-rate",      type=float, default=float("inf"),
                   help="req/s; inf = fire all immediately")
    p.add_argument("--max-concurrency",   type=int, default=None,
                   help="max in-flight requests; None = unlimited")
    p.add_argument("--load-mode",         choices=["open-loop", "closed-loop"], default="open-loop",
                   help="label/gating for interpretation: open-loop treats client queue_delay as a warning; "
                        "closed-loop treats fixed-concurrency backpressure as expected. Scheduling is unchanged.")
    p.add_argument("--num-warmup",        type=int, default=3,
                   help="warmup requests sent before benchmark (discarded from results)")
    p.add_argument("--seed",              type=int, default=42)
    p.add_argument("--save-result",       action="store_true",
                   help="write JSON to benchmarks/results/")
    p.add_argument("--ignore-eos",        action="store_true",
                   help="force generation to run to max_tokens instead of stopping at EOS. "
                        "ON for feasibility/stress sweeps (controlled length), OFF (default) "
                        "for real experiments where natural stopping matters.")
    p.add_argument("--request-timeout",   type=float, default=600.0,
                   help="client-side per-request timeout in seconds (default: 600). "
                        "Timeouts are recorded as their own outcome, never silently dropped.")
    return p.parse_args()


async def amain():
    args = parse_args()

    if args.num_warmup > 0:
        print(f"Warming up ({args.num_warmup} requests, results discarded)…")
        warmup_samples = make_random_samples(args.num_warmup, 32, 32, args.seed + 999)
        await run_benchmark(
            warmup_samples, args.base_url, args.model,
            float("inf"), args.max_concurrency, args.seed + 999,
            args.ignore_eos, args.request_timeout,
        )
        print("Warmup complete.")

    print(f"Building dataset: {args.dataset_name}")
    if args.dataset_name == "random":
        samples = make_random_samples(
            args.num_prompts, args.random_input_len, args.random_output_len, args.seed,
        )
    elif args.dataset_name == "sharegpt":
        samples = make_sharegpt_samples(args.num_prompts, args.seed)
    elif args.dataset_name == "oasst1":
        samples = make_oasst1_samples(args.num_prompts, args.seed, args.oasst_output_len)
    elif args.dataset_name == "local":
        if not args.dataset_path:
            raise SystemExit("--dataset-name local requires --dataset-path <file.json>")
        samples = make_local_samples(
            args.dataset_path, args.num_prompts, args.seed,
            max_output_len=args.random_output_len,
        )
    else:
        raise SystemExit(f"Unknown dataset: {args.dataset_name}")

    print(f"Sending {len(samples)} requests → {args.base_url}")
    print(f"rate={args.request_rate} req/s  concurrency={args.max_concurrency}  "
          f"mode={args.load_mode}  seed={args.seed}")

    results, t0, t1 = await run_benchmark(
        samples, args.base_url, args.model,
        args.request_rate, args.max_concurrency, args.seed,
        args.ignore_eos, args.request_timeout,
    )

    summary = compute_summary(results, t1 - t0, args)
    print_summary(summary)

    if args.save_result:
        path = result_path(args)
        summary["raw"] = [
            {
                "ttft_ms":           r.ttft        * 1000,
                "tpot_ms":           r.tpot        * 1000,
                "itl_ms":            [x * 1000 for x in r.itl],
                "e2el_ms":           r.e2el        * 1000,
                "queue_delay_ms":    r.queue_delay * 1000,
                "class":             r.request_class,
                "prompt_len":        r.prompt_len,
                "actual_prompt_len": r.actual_prompt_len,
                "input_tokens":      r.actual_prompt_len,
                "output_len":        r.output_len,
                "requested_output_tokens": r.requested_output_tokens,
                "generated_output_tokens": r.generated_output_tokens,
                "finish_reason":     r.finish_reason,
                "truncated":         r.truncated,
                "outcome":           r.outcome,
                "success":           r.success,
                "error":             r.error,
            }
            for r in results
        ]
        path.write_text(json.dumps(summary, indent=2))
        print(f"Saved → {path}")


if __name__ == "__main__":
    asyncio.run(amain())
