import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "service_model.json")
EPS = 1e-9


def load_service_model(path=DEFAULT_MODEL):
    if not os.path.exists(path):
        raise SystemExit(
            f"{path} missing -- run `python3 simulator/calibrate.py` first."
        )
    return json.loads(open(path).read())


class Request:
    __slots__ = ("rid", "arrival_t", "prompt_tokens", "max_tokens", "cls",
                 "dispatch_t", "node", "prompt_left", "tokens_left",
                 "first_token_t", "finish_t", "outcome")

    def __init__(self, rid, arrival_t, prompt_tokens, max_tokens, cls):
        self.rid = rid
        self.arrival_t = arrival_t
        self.prompt_tokens = prompt_tokens
        self.max_tokens = max_tokens
        self.cls = cls
        self.dispatch_t = None
        self.node = None
        self.prompt_left = float(prompt_tokens)
        self.tokens_left = float(max_tokens)
        self.first_token_t = None
        self.finish_t = None
        self.outcome = None

    @property
    def context(self):
        return self.prompt_tokens + (self.max_tokens - self.tokens_left)

    def e2el_ms(self):
        if self.finish_t is None or self.dispatch_t is None:
            return None
        return (self.finish_t - self.dispatch_t) * 1000.0

    def ttft_ms(self):
        if self.first_token_t is None or self.dispatch_t is None:
            return None
        return (self.first_token_t - self.dispatch_t) * 1000.0

    def tpot_ms(self):
        if self.finish_t is None or self.first_token_t is None:
            return None
        generated = self.max_tokens - self.tokens_left
        if generated <= 1:
            return None
        return (self.finish_t - self.first_token_t) * 1000.0 / (generated - 1)


def _interp(points, x):
    """points: sorted [(x, y)]. Linear interpolation, clamped at the ends."""
    if len(points) == 1:
        return points[0][1]
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= x <= x1:
            if x1 == x0:
                return y0
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return points[-1][1]


class Node:
    """One llama.cpp server."""

    def __init__(self, name, spec, np_slots=4):
        self.name = name
        self.prefill_rate = spec["prefill_rate_tok_s"]
        self.np = np_slots
        # {B: [(R, tpot_ms)]} sorted by R
        self.grid = {
            int(b): sorted((int(r), v) for r, v in cells.items())
            for b, cells in spec["decode_tpot"].items()
        }
        self._bs = sorted(self.grid)
        self.running = []   # occupying a slot
        self.queue = []     # dispatched, waiting for a slot
        # What the ROUTER assumes about in-flight work it cannot inspect: an
        # EWMA of output lengths it has seen complete on this node. A real Rust
        # router can maintain exactly this. Seeded at 128 (the uniform cell's
        # output length) so the first few decisions are not wild.
        self.avg_tokens_hint = 128.0

    def tpot_ms(self, batch, context):
        """Per-request ms/token at batch depth `batch`, context `context`.

        The measured B-effect is non-monotonic (B=2 is slower than B=4 on all
        three nodes), so this interpolates the measured grid rather than
        assuming any monotone shape.
        """
        batch = max(1, min(batch, self._bs[-1]))
        per_b = [(b, _interp(self.grid[b], context)) for b in self._bs]
        return _interp(per_b, batch)

    # state the router can observe
    def observed_b(self):
        return len(self.running) + len(self.queue)

    def observed_u(self):
        """KV-slot utilisation, the signal state_aware reads."""
        return min(1.0, len(self.running) / float(self.np))

    # simulation
    def admit(self, req, now):
        if len(self.running) < self.np:
            self.running.append(req)
        else:
            self.queue.append(req)

    def prefilling(self):
        return [r for r in self.running if r.prompt_left > EPS]

    def next_event_dt(self):
        """Time until this node's state next changes, or None if idle."""
        pre = self.prefilling()
        if pre:
            total = sum(r.prompt_left for r in pre)
            return total / self.prefill_rate
        if not self.running:
            return None
        batch = len(self.running)
        ctx = sum(r.context for r in self.running) / batch
        rate = 1000.0 / self.tpot_ms(batch, ctx)   # tokens/s per sequence
        return min(r.tokens_left / rate for r in self.running)

    def advance(self, dt, now):
        """Apply dt of progress. Returns requests that finished."""
        pre = self.prefilling()
        if pre:
            budget = self.prefill_rate * dt
            total = sum(r.prompt_left for r in pre)
            if total <= EPS:
                return []
            for r in pre:
                r.prompt_left -= budget * (r.prompt_left / total)
                if r.prompt_left <= EPS:
                    r.prompt_left = 0.0
                    if r.first_token_t is None:
                        r.first_token_t = now + dt
            return []
        if not self.running:
            return []
        batch = len(self.running)
        ctx = sum(r.context for r in self.running) / batch
        rate = 1000.0 / self.tpot_ms(batch, ctx)
        done = []
        for r in self.running:
            r.tokens_left -= rate * dt
            if r.tokens_left <= EPS:
                r.tokens_left = 0.0
                r.finish_t = now + dt
                r.outcome = "success"
                done.append(r)
        return done

    def release(self, req):
        if req in self.running:
            self.running.remove(req)
        elif req in self.queue:
            self.queue.remove(req)
        # the router learns the true output length once the stream ends
        self.avg_tokens_hint = 0.8 * self.avg_tokens_hint + 0.2 * req.max_tokens
        while self.queue and len(self.running) < self.np:
            self.running.append(self.queue.pop(0))


class Cluster:
    """Client + router + nodes."""

    def __init__(self, service_model, policy, *, nodes=None, np_slots=4,
                 request_rate=2.0, max_concurrency=8, poll_interval_ms=350,
                 upstream_timeout_s=120.0, seed=42):
        names = nodes or list(service_model["nodes"])
        self.nodes = [Node(n, service_model["nodes"][n], np_slots) for n in names]
        self.by_name = {n.name: n for n in self.nodes}
        self.policy = policy
        self.request_rate = request_rate
        self.max_concurrency = max_concurrency
        self.poll_interval = poll_interval_ms / 1000.0
        self.timeout = upstream_timeout_s
        self.rng = random.Random(seed)
        self.snapshot = None
        self.next_poll = 0.0

    def _refresh(self, now):
        """Take a new state snapshot; this is what the poller thread does."""
        self.snapshot = [
            {"replica": n.name, "b": n.observed_b(), "u": n.observed_u(),
             "running": len(n.running), "np": n.np, "age_s": 0.0}
            for n in self.nodes
        ]

    def _tick_poller(self, now):
        """Advance the poller to `now`, refreshing on its own fixed schedule.

        The poller runs independently of request dispatch, so its clock must
        advance with simulated time whether or not anything was routed. (Tying
        it to dispatches makes the next-event time collapse to zero and the
        event loop livelocks.)
        """
        if self.snapshot is None:
            self._refresh(now)
            self.next_poll = now + self.poll_interval
            return
        while now >= self.next_poll - EPS:
            self._refresh(now)
            self.next_poll += self.poll_interval

    def _view(self, now):
        """What the router sees: the last snapshot, stale by up to one poll."""
        if self.snapshot is None:
            self._tick_poller(now)
        return self.snapshot

    def run(self, requests):
        """Simulate `requests` (sorted by arrival_t). Returns the same list."""
        pending = list(requests)
        inflight = []          # holding a client semaphore slot
        t = 0.0
        idx = 0
        while idx < len(pending) or inflight:
            self._tick_poller(t)
            # 1. dispatch everything the semaphore and arrival clock allow
            while (idx < len(pending)
                   and len(inflight) < self.max_concurrency
                   and pending[idx].arrival_t <= t + EPS):
                req = pending[idx]
                idx += 1
                state = self._view(t)
                choice = self.policy.select(state, req, self)
                node = self.by_name[choice]
                req.node = choice
                req.dispatch_t = t
                node.admit(req, t)
                inflight.append(req)

            # 2. time of the next thing that can happen
            cands = []
            if idx < len(pending) and len(inflight) < self.max_concurrency:
                cands.append(max(0.0, pending[idx].arrival_t - t))
            for n in self.nodes:
                dt = n.next_event_dt()
                if dt is not None:
                    cands.append(max(dt, EPS))
            for r in inflight:
                cands.append(max(self.timeout - (t - r.dispatch_t), EPS))
            # the poller changes what the router sees even if nothing else moves
            if idx < len(pending):
                cands.append(max(self.next_poll - t, EPS))
            if not cands:
                break
            dt = min(cands)

            # 3. advance
            finished = []
            for n in self.nodes:
                finished.extend(n.advance(dt, t))
            t += dt

            # 4. deadline expiries
            for r in list(inflight):
                if r not in finished and (t - r.dispatch_t) >= self.timeout - EPS:
                    r.outcome = "deadline"
                    r.finish_t = r.dispatch_t + self.timeout
                    finished.append(r)

            for r in finished:
                self.by_name[r.node].release(r)
                if r in inflight:
                    inflight.remove(r)

        return requests
