import itertools
import random

WEIGHTS = {"pi-node-1": 3.1, "pi-node-2": 1.0, "pi-node-3": 1.0}
EPS = 1e-9


# forward projection through the service model
def project_completion(node, new_req_prompt, new_req_tokens, running, queue):
    """Seconds until a new request would finish on `node`.

    Replays the node's own service model over a snapshot of its work, with no
    future arrivals. `running` / `queue` are [(prompt_left, tokens_left,
    prompt_tokens, max_tokens)] tuples so the caller's real requests are never
    mutated.
    """
    run = [list(x) for x in running]
    wait = [list(x) for x in queue]
    new = [float(new_req_prompt), float(new_req_tokens),
           float(new_req_prompt), float(new_req_tokens)]
    if len(run) < node.np:
        run.append(new)
    else:
        wait.append(new)

    t = 0.0
    guard = 0
    while new in run or new in wait:
        guard += 1
        if guard > 10_000:          # cannot happen with finite work; be safe
            return float("inf")
        pre = [r for r in run if r[0] > EPS]
        if pre:
            total = sum(r[0] for r in pre)
            dt = total / node.prefill_rate
            for r in pre:
                r[0] = 0.0
            t += dt
            continue
        if not run:
            break
        batch = len(run)
        ctx = sum((r[2] + (r[3] - r[1])) for r in run) / batch
        rate = 1000.0 / node.tpot_ms(batch, ctx)
        dt = min(r[1] / rate for r in run)
        for r in run:
            r[1] -= rate * dt
        t += dt
        done = [r for r in run if r[1] <= EPS]
        for r in done:
            run.remove(r)
            if r is new:
                return t
        while wait and len(run) < node.np:
            run.append(wait.pop(0))
    return t


def _snapshot(node):
    running = [(r.prompt_left, r.tokens_left, r.prompt_tokens, r.max_tokens)
               for r in node.running]
    queue = [(r.prompt_left, r.tokens_left, r.prompt_tokens, r.max_tokens)
             for r in node.queue]
    return running, queue


def _stale_snapshot(node, entry):
    """Reconstruct plausible work from what the ROUTER can see.

    The router knows how many sequences are running and queued, not their
    remaining tokens. It assumes each in-flight sequence is half-done, which is
    the expectation under a uniform prior -- a real Rust implementation would
    do the same with the same information.
    """
    assumed_prompt, assumed_out = 0.0, node.avg_tokens_hint
    running = [(assumed_prompt, assumed_out / 2.0, assumed_out, assumed_out)
               for _ in range(min(entry["running"], node.np))]
    extra = max(0, entry["b"] - entry["running"])
    queue = [(assumed_prompt, assumed_out, assumed_out, assumed_out)
             for _ in range(extra)]
    return running, queue


# stateless / load-only policies (what the Rust router does today)
class Random:
    name = "random"

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def select(self, state, req, cluster):
        return self.rng.choice(state)["replica"]


class RoundRobin:
    name = "round_robin"

    def __init__(self):
        self.i = 0

    def select(self, state, req, cluster):
        choice = state[self.i % len(state)]["replica"]
        self.i += 1
        return choice


class WeightedRoundRobin:
    name = "weighted_round_robin"

    def __init__(self, weights=None):
        w = weights or WEIGHTS
        # smooth WRR: repeat each node in proportion to its weight
        scale = 10
        seq = []
        for node, weight in w.items():
            seq.extend([node] * max(1, int(round(weight * scale))))
        seq.sort(key=lambda n: -w[n])
        self.cycle = itertools.cycle(seq)

    def select(self, state, req, cluster):
        live = {e["replica"] for e in state}
        for _ in range(1000):
            pick = next(self.cycle)
            if pick in live:
                return pick
        return state[0]["replica"]


class LeastConnections:
    name = "least_connections"

    def select(self, state, req, cluster):
        return min(state, key=lambda e: (e["b"], e["replica"]))["replica"]


class WeightedLeastConnections:
    name = "weighted_least_connections"

    def __init__(self, weights=None):
        self.w = weights or WEIGHTS

    def select(self, state, req, cluster):
        return min(
            state, key=lambda e: (e["b"] / self.w[e["replica"]], e["replica"])
        )["replica"]


class StateAware:
    name = "state_aware"

    def __init__(self, weights=None):
        self.w = weights or WEIGHTS

    def select(self, state, req, cluster):
        return min(
            state, key=lambda e: (e["u"] / self.w[e["replica"]], e["replica"])
        )["replica"]


# length-aware: the policy this study is really proposing
class LengthAware:
    """Projected-completion argmin using max_tokens and the STALE snapshot.

    Implementable in the Rust router today: parse max_tokens from the request
    body, keep the existing 350ms poller, add the calibrated cost model.
    """
    name = "length_aware"

    def __init__(self, deadline_s=120.0):
        self.deadline = deadline_s

    def select(self, state, req, cluster):
        best, best_cost = None, None
        for entry in state:
            node = cluster.by_name[entry["replica"]]
            running, queue = _stale_snapshot(node, entry)
            eta = project_completion(
                node, req.prompt_tokens, req.max_tokens, running, queue
            )
            # a projected deadline breach is far worse than being slow
            cost = eta + (1e6 if eta > self.deadline else 0.0)
            if best_cost is None or cost < best_cost:
                best, best_cost = entry["replica"], cost
        return best


class LengthAwareCongestion:
    """length_aware + an explicit congestion penalty.

    oracle_greedy (perfect info, pure ETA-argmin) still loses to WLC on
    completion at high concurrency because minimising THIS request's ETA
    packs the fast node and starves later arrivals - the projected ETA
    already includes queueing delay, but a node can still look "best" while
    its queue is quietly building, because its throughput advantage
    outweighs the queue for a while. This adds an explicit per-excess-queued-
    request penalty (seconds), independent of the ETA projection, so a node
    gets less attractive as its queue builds even before ETA alone would
    reflect the full cost - the fairness/capacity term the README's finding
    says is missing. Same stale snapshot as length_aware (implementable
    today: parse max_tokens, keep the 350ms poller, add this term).
    """
    name = "length_aware_congestion"

    def __init__(self, deadline_s=120.0, congestion_weight=1.0, congestion_power=2.0):
        self.deadline = deadline_s
        self.congestion_weight = congestion_weight
        # power=1 (linear) penalizes a queue_depth of 1 as much per-unit as a
        # queue_depth of 5 -- that's what made c8_bimodal/c8_sharegpt pay for
        # caution they didn't need. power>1 makes small, normal queueing
        # nearly free while still strongly discouraging runaway pile-up.
        self.congestion_power = congestion_power

    def select(self, state, req, cluster):
        best, best_cost = None, None
        for entry in state:
            node = cluster.by_name[entry["replica"]]
            running, queue = _stale_snapshot(node, entry)
            eta = project_completion(
                node, req.prompt_tokens, req.max_tokens, running, queue
            )
            queue_depth = max(0, entry["b"] - entry["np"])
            cost = (eta
                    + self.congestion_weight * (queue_depth ** self.congestion_power)
                    + (1e6 if eta > self.deadline else 0.0))
            if best_cost is None or cost < best_cost:
                best, best_cost = entry["replica"], cost
        return best


# the bound
class OracleGreedy:
    """Exact node state + exact output length + true service model."""
    name = "oracle_greedy"

    def __init__(self, deadline_s=120.0):
        self.deadline = deadline_s

    def select(self, state, req, cluster):
        best, best_cost = None, None
        for node in cluster.nodes:
            running, queue = _snapshot(node)
            eta = project_completion(
                node, req.prompt_tokens, req.max_tokens, running, queue
            )
            cost = eta + (1e6 if eta > self.deadline else 0.0)
            if best_cost is None or cost < best_cost:
                best, best_cost = node.name, cost
        return best


class FixedAssignment:
    """Replay a precomputed per-request node assignment.

    Used by oracle_search.py: it lets an offline optimiser propose a whole
    assignment vector and score it through the same service model.
    """
    name = "fixed_assignment"

    def __init__(self, assignment):
        self.assignment = list(assignment)

    def select(self, state, req, cluster):
        return self.assignment[req.rid - 1]


REGISTRY = {
    "random": Random,
    "round_robin": RoundRobin,
    "weighted_round_robin": WeightedRoundRobin,
    "least_connections": LeastConnections,
    "weighted_least_connections": WeightedLeastConnections,
    "state_aware": StateAware,
    "length_aware": LengthAware,
    "length_aware_congestion": LengthAwareCongestion,
    "oracle_greedy": OracleGreedy,
}


def build(name, **kw):
    if name not in REGISTRY:
        raise SystemExit(f"unknown policy {name!r}; choose from {sorted(REGISTRY)}")
    cls = REGISTRY[name]
    if cls is LengthAwareCongestion:
        kwargs = {}
        if "deadline_s" in kw:
            kwargs["deadline_s"] = kw["deadline_s"]
        if "congestion_weight" in kw:
            kwargs["congestion_weight"] = kw["congestion_weight"]
        if "congestion_power" in kw:
            kwargs["congestion_power"] = kw["congestion_power"]
        return cls(**kwargs)
    if cls in (LengthAware, OracleGreedy) and "deadline_s" in kw:
        return cls(deadline_s=kw["deadline_s"])
    return cls()
