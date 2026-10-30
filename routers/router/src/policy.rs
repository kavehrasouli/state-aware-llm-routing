//! The routing policies. `select` reads only cached engine state and health, so it is unit-testable.
//! Selection is timed as one region; state-reading policies also load the state snapshot inside it.

use crate::cache::{Snapshot, StateCache};
use crate::reachability::Reachability;
use crate::replica::ReplicaState;
use crate::reservation::{PendingDispatches, ReservationGuard};
use crate::tracker::{LoadTracker, TrackGuard, TrackedLoad};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// How the decision handled state staleness. Logged per request so §3e runs
/// that rode stale state can be found and excluded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StaleMode {
    /// Stateless policy — staleness not applicable.
    Stateless,
    /// State policy; the chosen replica had fresh state.
    Live,
    /// Some eligible replicas were stale and excluded; a live one was chosen.
    SomeStaleExcluded,
    /// All eligible replicas stale but same-generation (age spread ≤ one poll
    /// interval): compared last-known values (apples-to-apples).
    AllStaleCompare,
    /// All eligible replicas stale with divergent ages: no fair comparison, so
    /// round-robin (blind).
    AllStaleRr,
    /// No eligible replicas at all.
    NoEligible,
    /// Router-tracked policy: decided on the router's own load count, which
    /// has no age. Eligibility is health-only.
    Tracked,
}

impl StaleMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            StaleMode::Stateless => "stateless",
            StaleMode::Live => "live",
            StaleMode::SomeStaleExcluded => "some_stale_excluded",
            StaleMode::AllStaleCompare => "all_stale_compare",
            StaleMode::AllStaleRr => "all_stale_rr",
            StaleMode::NoEligible => "no_eligible",
            StaleMode::Tracked => "tracked",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    Random,
    RoundRobin,
    LeastConnections,
    WeightedRoundRobin,
    WeightedLeastConnections,
    StateAware,
    CostModel,
    StateAwareV2,
    CostPredictV3,
    /// Experimental: like cost_predict_v3, but scores on the largest single
    /// busy slot's resident tokens (projected after admission), not just the
    /// summed total -- the "shape" signal v3 can't see.
    ShapeAware,
    /// Experimental: cost_predict_v3's prediction plus a squared penalty on queued requests
    /// (from the simulator). CONGESTION_WEIGHT is an untuned starting value.
    LengthAwareCongestion,
    /// state_aware's score on router-tracked context instead of polled
    /// resident context: argmin (tracked_tokens / c) / w.
    StateAwareTracked,
    /// weighted_least_connections on router-tracked outstanding requests
    /// instead of polled requests_processing: argmin tracked_requests / w.
    WlcTracked,
}

/// ms of penalty per (requests_deferred)^2. Untuned placeholder -- see the
/// LengthAwareCongestion variant doc. Needs its own calibration pass, the
/// same way c0/c1/c2 needed the offline sweep.
const CONGESTION_WEIGHT_MS: f64 = 30.0;

impl Policy {
    pub fn parse(s: &str) -> Result<Policy, String> {
        Ok(match s {
            "random" => Policy::Random,
            "round_robin" => Policy::RoundRobin,
            "least_connections" => Policy::LeastConnections,
            "weighted_round_robin" => Policy::WeightedRoundRobin,
            "weighted_least_connections" => Policy::WeightedLeastConnections,
            "state_aware" => Policy::StateAware,
            "cost_model" => Policy::CostModel,
            "state_aware_v2" => Policy::StateAwareV2,
            "cost_predict_v3" => Policy::CostPredictV3,
            "shape_aware" => Policy::ShapeAware,
            "length_aware_congestion" => Policy::LengthAwareCongestion,
            "state_aware_tracked" => Policy::StateAwareTracked,
            "wlc_tracked" => Policy::WlcTracked,
            other => return Err(format!("unknown policy '{other}'")),
        })
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Policy::Random => "random",
            Policy::RoundRobin => "round_robin",
            Policy::LeastConnections => "least_connections",
            Policy::WeightedRoundRobin => "weighted_round_robin",
            Policy::WeightedLeastConnections => "weighted_least_connections",
            Policy::StateAware => "state_aware",
            Policy::CostModel => "cost_model",
            Policy::StateAwareV2 => "state_aware_v2",
            Policy::CostPredictV3 => "cost_predict_v3",
            Policy::ShapeAware => "shape_aware",
            Policy::LengthAwareCongestion => "length_aware_congestion",
            Policy::StateAwareTracked => "state_aware_tracked",
            Policy::WlcTracked => "wlc_tracked",
        }
    }

    /// Whether this policy decides on router-tracked load (`select_tracked`).
    pub fn is_tracked(&self) -> bool {
        matches!(self, Policy::StateAwareTracked | Policy::WlcTracked)
    }

    /// Whether this policy consumes the live u/B snapshot to decide.
    pub fn reads_state(&self) -> bool {
        matches!(
            self,
            Policy::LeastConnections
                | Policy::WeightedLeastConnections
                | Policy::StateAware
                | Policy::CostModel
                | Policy::StateAwareV2
                | Policy::CostPredictV3
                | Policy::ShapeAware
                | Policy::LengthAwareCongestion
        )
    }

    /// Per-replica score for the state policies (lower = preferred). The SINGLE
    /// source of truth for scalar argmin objectives — both `select` and the
    /// `--verbose` diagnostic use it. None means stateless or a request/staged
    /// policy with its own selection path.
    pub fn score(&self, s: &ReplicaState) -> Option<f64> {
        match self {
            Policy::LeastConnections => Some(s.b_metrics), // argmin B_r
            Policy::WeightedLeastConnections => Some(s.b_metrics / s.weight), // argmin B_r / w_r
            Policy::StateAware => Some(s.u / s.weight),    // argmin u_r / w_r
            // argmin(c0 + c1·B + c2·R) with PER-NODE fitted coefficients and RAW
            // resident_tokens (not u). No /weight: the coefficients already
            // encode per-node speed; dividing would double-count capacity. No
            // interaction term (c3 ≈ 0 with CI crossing zero on every node).
            Policy::CostModel => Some(s.c0 + s.c1 * s.b_metrics + s.c2 * s.resident_tokens as f64),
            // v2 is a staged filter, not a scalar objective.
            Policy::StateAwareV2 => None,
            // v3 and shape_aware need request_tokens, so they use `select_predictive`.
            Policy::CostPredictV3 => None,
            Policy::ShapeAware => None,
            Policy::LengthAwareCongestion => None,
            _ => None,
        }
    }

    /// Choose a replica. See the module-level measurement contract.
    pub fn select(&self, cache: &StateCache, ctx: &SelectCtx) -> Decision {
        // Every policy pays exactly this one atomic reachability read (inside the
        // timed region). Stateless policies stop here; state policies load on top.
        let mask = ctx.reach.load();
        let n = ctx.n;

        match self {
            Policy::Random => Decision::stateless(ctx.pick_random_healthy(mask, n)),
            Policy::RoundRobin => Decision::stateless(ctx.pick_round_robin(mask, n)),
            Policy::WeightedRoundRobin => Decision::stateless(ctx.pick_wrr(mask, n)),

            Policy::StateAwareV2 => {
                let snap = cache.load_full();
                let now = Instant::now();
                let stale_candidates = ctx.stateful_candidates(&snap, mask, now);
                let (choice, mode, reservation, candidate_logs) =
                    ctx.state_aware_v2_choice(&snap, stale_candidates);
                Decision::v2(choice, snap, mode, reservation, candidate_logs)
            }

            Policy::CostPredictV3 => {
                panic!("cost_predict_v3 requires select_predictive(request_tokens)")
            }
            Policy::ShapeAware => {
                panic!("shape_aware requires select_predictive(request_tokens)")
            }
            Policy::LengthAwareCongestion => {
                panic!("length_aware_congestion requires select_predictive(request_tokens)")
            }
            Policy::StateAwareTracked | Policy::WlcTracked => {
                panic!("tracked policies require select_tracked(prompt_tokens)")
            }

            // These scalar state policies differ ONLY in `Policy::score`; the
            // snapshot load, `now`, and stale-aware fallback are identical.
            Policy::LeastConnections
            | Policy::WeightedLeastConnections
            | Policy::StateAware
            | Policy::CostModel => {
                let snap = cache.load_full();
                let now = Instant::now();
                let (choice, mode) = ctx
                    .stateful_choice(&snap, mask, now, |s| self.score(s).unwrap_or(f64::INFINITY));
                Decision::stateful(choice, snap, mode)
            }
        }
    }

    /// Request-aware selection for the experimental predictive policy. Prompt
    /// extraction happens in the caller; this timed path loads one state
    /// snapshot and uses the shared stale-state rules.
    pub fn select_predictive(
        &self,
        cache: &StateCache,
        ctx: &SelectCtx,
        request_tokens: u64,
    ) -> Decision {
        assert!(
            matches!(
                *self,
                Policy::CostPredictV3 | Policy::ShapeAware | Policy::LengthAwareCongestion
            ),
            "select_predictive is v3/shape_aware/length_aware_congestion-only"
        );
        let predict = |state: &ReplicaState| match self {
            Policy::ShapeAware => shape_prediction(state, request_tokens),
            Policy::LengthAwareCongestion => congestion_prediction(state, request_tokens),
            _ => v3_prediction(state, request_tokens),
        };
        let mask = ctx.reach.load();
        let snap = cache.load_full();
        let now = Instant::now();
        let mut candidate_logs: Vec<V3CandidateLog> =
            snap.iter().map(predict).collect();
        let (choice, mode) =
            ctx.stateful_choice(&snap, mask, now, |state| predict(state).predicted_tpot);
        if let Some(i) = choice {
            candidate_logs[i].chosen = true;
        }
        Decision::v3(choice, snap, mode, candidate_logs)
    }

    /// Selection on router-tracked load. Reads health and the tracker only, no
    /// polled snapshot; the chosen replica is tracked inside the same lock.
    /// Ties resolve to the lowest index, as in `argmin_over`.
    pub fn select_tracked(&self, ctx: &SelectCtx, prompt_tokens: u64) -> Decision {
        assert!(self.is_tracked(), "select_tracked is tracked-policy-only");
        let mask = ctx.reach.load();
        let mut choice = None;
        let (tracking, view) = ctx.tracker.select_and_track(prompt_tokens, |loads| {
            choice = argmin_tracked(*self, ctx, mask, loads);
            choice
        });
        Decision {
            choice,
            snapshot: None,
            stale_mode: if choice.is_some() {
                StaleMode::Tracked
            } else {
                StaleMode::NoEligible
            },
            reservation: None,
            v2_candidate_logs: None,
            v3_candidate_logs: None,
            tracking,
            tracked_view: Some(view),
        }
    }

    /// Tracked score for one replica (lower = preferred).
    fn tracked_score(&self, load: &TrackedLoad, context_budget: f64, weight: f64) -> f64 {
        match self {
            Policy::StateAwareTracked => load.tokens as f64 / context_budget / weight,
            Policy::WlcTracked => load.requests as f64 / weight,
            _ => f64::INFINITY,
        }
    }
}

fn argmin_tracked(
    policy: Policy,
    ctx: &SelectCtx,
    mask: u64,
    loads: &[TrackedLoad],
) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (i, load) in loads.iter().enumerate().take(ctx.n) {
        if !Reachability::bit(mask, i) {
            continue;
        }
        let sc = policy.tracked_score(load, ctx.context_budgets[i], ctx.weights[i]);
        if sc.is_nan() {
            continue;
        }
        if best.is_none_or(|(_, b)| sc < b) {
            best = Some((i, sc));
        }
    }
    best.map(|(i, _)| i)
}

/// Result of a selection. `snapshot` is the state the decision used (None for stateless policies,
/// which log a snapshot read afterwards, outside the timed region).
pub struct Decision {
    pub choice: Option<usize>,
    pub snapshot: Option<Arc<Snapshot>>,
    pub stale_mode: StaleMode,
    pub reservation: Option<ReservationGuard>,
    pub v2_candidate_logs: Option<Vec<V2CandidateLog>>,
    pub v3_candidate_logs: Option<Vec<V3CandidateLog>>,
    /// Tracked policies only: the tracker entry created inside selection, and
    /// the tracked view the decision saw. Other policies are tracked by the
    /// proxy after selection.
    pub tracking: Option<TrackGuard>,
    pub tracked_view: Option<Vec<TrackedLoad>>,
}

impl Decision {
    fn stateless(choice: Option<usize>) -> Decision {
        Decision {
            choice,
            snapshot: None,
            stale_mode: StaleMode::Stateless,
            reservation: None,
            v2_candidate_logs: None,
            v3_candidate_logs: None,
            tracking: None,
            tracked_view: None,
        }
    }
    fn stateful(choice: Option<usize>, snap: Arc<Snapshot>, stale_mode: StaleMode) -> Decision {
        Decision {
            choice,
            snapshot: Some(snap),
            stale_mode,
            reservation: None,
            v2_candidate_logs: None,
            v3_candidate_logs: None,
            tracking: None,
            tracked_view: None,
        }
    }
    fn v2(
        choice: Option<usize>,
        snap: Arc<Snapshot>,
        stale_mode: StaleMode,
        reservation: Option<ReservationGuard>,
        candidate_logs: Vec<V2CandidateLog>,
    ) -> Decision {
        Decision {
            choice,
            snapshot: Some(snap),
            stale_mode,
            reservation,
            v2_candidate_logs: Some(candidate_logs),
            v3_candidate_logs: None,
            tracking: None,
            tracked_view: None,
        }
    }
    fn v3(
        choice: Option<usize>,
        snap: Arc<Snapshot>,
        stale_mode: StaleMode,
        candidate_logs: Vec<V3CandidateLog>,
    ) -> Decision {
        Decision {
            choice,
            snapshot: Some(snap),
            stale_mode,
            reservation: None,
            v2_candidate_logs: None,
            v3_candidate_logs: Some(candidate_logs),
            tracking: None,
            tracked_view: None,
        }
    }
}

/// The exact v2 values used for one replica at decision time. These are copied
/// into the request's state_vector outside the timed selection region.
#[derive(Clone, Debug)]
pub struct V2CandidateLog {
    pub local_pending_dispatches: u64,
    pub b_eff: f64,
    pub weighted_load: f64,
    pub stage1_exclusion_reason: Option<&'static str>,
    pub stage1_guard_fallback: bool,
    pub shortlisted: bool,
}

/// Per-candidate projected values used by cost_predict_v3. The request-token
/// estimate itself is attached by the proxy because exact/approximate
/// provenance belongs to request parsing, not policy scoring.
#[derive(Clone, Debug)]
pub struct V3CandidateLog {
    pub b_proj: f64,
    pub r_proj_unclamped: u64,
    pub r_proj: u64,
    pub r_proj_clamped: bool,
    pub predicted_tpot: f64,
    pub chosen: bool,
}

fn v3_prediction(state: &ReplicaState, request_tokens: u64) -> V3CandidateLog {
    let b_proj = state.b_metrics + 1.0;
    let r_proj_unclamped = state.resident_tokens.saturating_add(request_tokens);
    let context_budget = state.context_budget.max(0.0) as u64;
    let r_proj = r_proj_unclamped.min(context_budget);
    let predicted_tpot = state.c0 + state.c1 * b_proj + state.c2 * r_proj as f64;
    V3CandidateLog {
        b_proj,
        r_proj_unclamped,
        r_proj,
        r_proj_clamped: r_proj != r_proj_unclamped,
        predicted_tpot,
        chosen: false,
    }
}

/// Experimental: v3's prediction plus a term for the largest slot after admission.
/// c3 comes from one node (pi-node-2) and is a heuristic, not a validated model.
fn shape_prediction(state: &ReplicaState, request_tokens: u64) -> V3CandidateLog {
    let b_proj = state.b_metrics + 1.0;
    let r_proj_unclamped = state.resident_tokens.saturating_add(request_tokens);
    let context_budget = state.context_budget.max(0.0) as u64;
    let r_proj = r_proj_unclamped.min(context_budget);
    let max_slot_proj = state.max_slot_tokens.max(request_tokens).min(context_budget);
    let predicted_tpot =
        state.c0 + state.c1 * b_proj + state.c2 * r_proj as f64 + state.c3 * max_slot_proj as f64;
    V3CandidateLog {
        b_proj,
        r_proj_unclamped,
        r_proj,
        r_proj_clamped: r_proj != r_proj_unclamped,
        predicted_tpot,
        chosen: false,
    }
}

/// Experimental: v3's prediction plus a penalty on requests_deferred^2 (untuned weight).
fn congestion_prediction(state: &ReplicaState, request_tokens: u64) -> V3CandidateLog {
    let b_proj = state.b_metrics + 1.0;
    let r_proj_unclamped = state.resident_tokens.saturating_add(request_tokens);
    let context_budget = state.context_budget.max(0.0) as u64;
    let r_proj = r_proj_unclamped.min(context_budget);
    let base = state.c0 + state.c1 * b_proj + state.c2 * r_proj as f64;
    let congestion_penalty = CONGESTION_WEIGHT_MS * state.requests_deferred.powi(2);
    let predicted_tpot = base + congestion_penalty;
    V3CandidateLog {
        b_proj,
        r_proj_unclamped,
        r_proj,
        r_proj_clamped: r_proj != r_proj_unclamped,
        predicted_tpot,
        chosen: false,
    }
}

enum StatefulCandidates {
    None(StaleMode),
    Compare(Vec<usize>, StaleMode),
    RoundRobin(Vec<usize>, StaleMode),
}

struct StatefulScope {
    eligible_mask: u64,
    live_mask: u64,
    any_stale: bool,
}

/// argmin over the given candidate indices. Ties resolve to the lowest index
/// (strict `<` replace) so runs are deterministic. NaN scores are skipped so
/// they can never "win".
fn argmin_over(
    snap: &Snapshot,
    candidates: impl Iterator<Item = usize>,
    score: &impl Fn(&ReplicaState) -> f64,
) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for i in candidates {
        let sc = score(&snap[i]);
        if sc.is_nan() {
            continue;
        }
        if best.is_none_or(|(_, b)| sc < b) {
            best = Some((i, sc));
        }
    }
    best.map(|(i, _)| i)
}

/// Mutable selection state shared across requests (RR cursor, smooth-WRR running
/// weights, RNG) plus the static weights, the shared reachability mask, and the
/// staleness thresholds.
pub struct SelectCtx {
    pub n: usize,
    weights: Vec<f64>,
    pub reach: Arc<Reachability>,
    /// state older than this is excluded from argmin (see `stateful_choice`).
    max_state_age: Duration,
    /// tolerance for the all-stale "same generation" compare — one poll cycle.
    poll_interval: Duration,
    rr: AtomicUsize,
    wrr_current: Mutex<Vec<f64>>,
    rng: Mutex<SmallRng>,
    pending_dispatches: Arc<PendingDispatches>,
    /// Router-tracked load, maintained for every policy.
    tracker: Arc<LoadTracker>,
    /// Per-replica context budget c, the tracked state-aware denominator.
    /// Defaults to 1.0 (argmin-neutral when budgets are equal).
    context_budgets: Vec<f64>,
}

impl SelectCtx {
    pub fn new(
        weights: Vec<f64>,
        reach: Arc<Reachability>,
        rng_seed: u64,
        max_state_age: Duration,
        poll_interval: Duration,
    ) -> SelectCtx {
        let n = weights.len();
        SelectCtx {
            n,
            wrr_current: Mutex::new(vec![0.0; n]),
            weights,
            reach,
            max_state_age,
            poll_interval,
            rr: AtomicUsize::new(0),
            rng: Mutex::new(SmallRng::seed_from_u64(rng_seed)),
            pending_dispatches: Arc::new(PendingDispatches::new(n)),
            tracker: Arc::new(LoadTracker::new(n)),
            context_budgets: vec![1.0; n],
        }
    }

    /// Set the per-replica context budgets used by `state_aware_tracked`.
    pub fn with_context_budgets(mut self, budgets: Vec<f64>) -> SelectCtx {
        assert_eq!(budgets.len(), self.n, "one context budget per replica");
        self.context_budgets = budgets;
        self
    }

    pub fn tracker(&self) -> Arc<LoadTracker> {
        self.tracker.clone()
    }

    pub fn pending_dispatches(&self) -> Arc<PendingDispatches> {
        self.pending_dispatches.clone()
    }

    /// Shared stale classification used by every state-reading policy. Bitmasks
    /// keep the original scalar policies allocation-free inside their timer.
    fn stateful_scope(&self, snap: &Snapshot, mask: u64, now: Instant) -> StatefulScope {
        let configured_mask = if snap.len() >= 64 {
            u64::MAX
        } else {
            (1_u64 << snap.len()) - 1
        };
        let eligible_mask = mask & configured_mask;
        let mut live_mask = 0_u64;
        for (i, state) in snap.iter().enumerate() {
            if Reachability::bit(eligible_mask, i)
                && now.saturating_duration_since(state.updated_at) <= self.max_state_age
            {
                live_mask |= 1_u64 << i;
            }
        }
        StatefulScope {
            eligible_mask,
            live_mask,
            any_stale: eligible_mask != live_mask,
        }
    }

    fn stale_age_spread(&self, snap: &Snapshot, eligible_mask: u64) -> Duration {
        let mut oldest: Option<Instant> = None;
        let mut newest: Option<Instant> = None;
        for (i, state) in snap.iter().enumerate() {
            if !Reachability::bit(eligible_mask, i) {
                continue;
            }
            oldest = Some(oldest.map_or(state.updated_at, |v| v.min(state.updated_at)));
            newest = Some(newest.map_or(state.updated_at, |v| v.max(state.updated_at)));
        }
        newest
            .zip(oldest)
            .map(|(newest, oldest)| newest.saturating_duration_since(oldest))
            .unwrap_or_default()
    }

    /// Stale-state rules: some fresh -> compare only fresh replicas; all stale within one poll
    /// interval of each other -> compare all; otherwise -> round-robin.
    fn stateful_candidates(&self, snap: &Snapshot, mask: u64, now: Instant) -> StatefulCandidates {
        let scope = self.stateful_scope(snap, mask, now);
        if scope.eligible_mask == 0 {
            return StatefulCandidates::None(StaleMode::NoEligible);
        }

        // Prefer live (fresh) eligible replicas.
        let live: Vec<usize> = (0..snap.len())
            .filter(|&i| Reachability::bit(scope.live_mask, i))
            .collect();
        if !live.is_empty() {
            return StatefulCandidates::Compare(
                live,
                if scope.any_stale {
                    StaleMode::SomeStaleExcluded
                } else {
                    StaleMode::Live
                },
            );
        }

        // All eligible replicas are stale. Compare only if same-generation.
        let eligible: Vec<usize> = (0..snap.len())
            .filter(|&i| Reachability::bit(scope.eligible_mask, i))
            .collect();
        if self.stale_age_spread(snap, scope.eligible_mask) <= self.poll_interval {
            StatefulCandidates::Compare(eligible, StaleMode::AllStaleCompare)
        } else {
            StatefulCandidates::RoundRobin(eligible, StaleMode::AllStaleRr)
        }
    }

    fn stateful_choice(
        &self,
        snap: &Snapshot,
        mask: u64,
        now: Instant,
        score: impl Fn(&ReplicaState) -> f64,
    ) -> (Option<usize>, StaleMode) {
        let scope = self.stateful_scope(snap, mask, now);
        if scope.eligible_mask == 0 {
            return (None, StaleMode::NoEligible);
        }

        // Preserve the original scalar-policy argmin path: no candidate Vec.
        let live = argmin_over(
            snap,
            (0..snap.len()).filter(|&i| Reachability::bit(scope.live_mask, i)),
            &score,
        );
        if let Some(i) = live {
            return (
                Some(i),
                if scope.any_stale {
                    StaleMode::SomeStaleExcluded
                } else {
                    StaleMode::Live
                },
            );
        }

        if self.stale_age_spread(snap, scope.eligible_mask) <= self.poll_interval {
            let choice = argmin_over(
                snap,
                (0..snap.len()).filter(|&i| Reachability::bit(scope.eligible_mask, i)),
                &score,
            );
            (choice, StaleMode::AllStaleCompare)
        } else {
            (
                self.pick_round_robin(scope.eligible_mask, self.n),
                StaleMode::AllStaleRr,
            )
        }
    }

    fn state_aware_v2_choice(
        &self,
        snap: &Snapshot,
        stale_candidates: StatefulCandidates,
    ) -> (
        Option<usize>,
        StaleMode,
        Option<ReservationGuard>,
        Vec<V2CandidateLog>,
    ) {
        let mode = match &stale_candidates {
            StatefulCandidates::None(mode)
            | StatefulCandidates::Compare(_, mode)
            | StatefulCandidates::RoundRobin(_, mode) => *mode,
        };

        let (reservation, (choice, logs)) = self.pending_dispatches.select_and_reserve(|pending| {
            let mut logs: Vec<V2CandidateLog> = snap
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    let b_eff = effective_b(s) + pending[i] as f64;
                    V2CandidateLog {
                        local_pending_dispatches: pending[i],
                        b_eff,
                        weighted_load: b_eff / s.weight,
                        stage1_exclusion_reason: None,
                        stage1_guard_fallback: false,
                        shortlisted: false,
                    }
                })
                .collect();

            let choice = match &stale_candidates {
                StatefulCandidates::None(_) => None,
                StatefulCandidates::RoundRobin(candidates, _) => {
                    self.pick_round_robin_candidates(candidates)
                }
                StatefulCandidates::Compare(candidates, _) => {
                    self.run_v2_stages(snap, candidates, &mut logs)
                }
            };
            (choice, (choice, logs))
        });

        (choice, mode, reservation, logs)
    }

    fn run_v2_stages(
        &self,
        snap: &Snapshot,
        candidates: &[usize],
        logs: &mut [V2CandidateLog],
    ) -> Option<usize> {
        if candidates.is_empty() {
            return None;
        }

        // Stage 1: use the guard only when at least one candidate has capacity.
        let guard_reasons: Vec<(usize, Option<&'static str>)> = candidates
            .iter()
            .map(|&i| (i, admission_guard_reason(&snap[i])))
            .collect();
        let has_capacity = guard_reasons.iter().any(|(_, reason)| reason.is_none());
        let admitted: Vec<usize> = if has_capacity {
            guard_reasons
                .iter()
                .filter_map(|&(i, reason)| {
                    if let Some(reason) = reason {
                        logs[i].stage1_exclusion_reason = Some(reason);
                        None
                    } else {
                        Some(i)
                    }
                })
                .collect()
        } else {
            for &(i, _) in &guard_reasons {
                logs[i].stage1_guard_fallback = true;
            }
            candidates.to_vec()
        };

        // Stage 2: candidate-specific one-request tolerance around min B_eff/w.
        let min_index = admitted
            .iter()
            .copied()
            .filter(|&i| logs[i].weighted_load.is_finite())
            .min_by(|&a, &b| logs[a].weighted_load.total_cmp(&logs[b].weighted_load))
            .or_else(|| admitted.first().copied())?;
        let min_load = logs[min_index].weighted_load;
        let mut shortlist: Vec<usize> = admitted
            .iter()
            .copied()
            .filter(|&i| logs[i].weighted_load <= min_load + (1.0 / snap[i].weight))
            .collect();
        if shortlist.is_empty() {
            // Defensive only: finite positive weights guarantee the min-load
            // candidate satisfies its own tolerance.
            shortlist.push(min_index);
        }
        for &i in &shortlist {
            logs[i].shortlisted = true;
        }

        // Stage 3: raw u, then weighted load, then RR among exact ties.
        let min_u = shortlist
            .iter()
            .copied()
            .filter(|&i| snap[i].u.is_finite())
            .min_by(|&a, &b| snap[a].u.total_cmp(&snap[b].u))
            .map(|i| snap[i].u);
        let u_tied: Vec<usize> = match min_u {
            Some(min_u) => shortlist
                .iter()
                .copied()
                .filter(|&i| snap[i].u == min_u)
                .collect(),
            None => shortlist,
        };
        let min_weighted = u_tied
            .iter()
            .copied()
            .min_by(|&a, &b| logs[a].weighted_load.total_cmp(&logs[b].weighted_load))
            .map(|i| logs[i].weighted_load)?;
        let exact_ties: Vec<usize> = u_tied
            .into_iter()
            .filter(|&i| logs[i].weighted_load == min_weighted)
            .collect();
        self.pick_round_robin_candidates(&exact_ties)
    }

    fn pick_random_healthy(&self, mask: u64, n: usize) -> Option<usize> {
        let healthy: Vec<usize> = (0..n).filter(|&i| Reachability::bit(mask, i)).collect();
        if healthy.is_empty() {
            return None;
        }
        let mut rng = self.rng.lock().unwrap();
        Some(healthy[rng.gen_range(0..healthy.len())])
    }

    fn pick_round_robin(&self, mask: u64, n: usize) -> Option<usize> {
        if mask == 0 {
            return None;
        }
        // Advance the shared cursor; skip unhealthy nodes (bounded by n).
        for _ in 0..n {
            let i = self.rr.fetch_add(1, Ordering::Relaxed) % n;
            if Reachability::bit(mask, i) {
                return Some(i);
            }
        }
        None
    }

    fn pick_round_robin_candidates(&self, candidates: &[usize]) -> Option<usize> {
        if candidates.is_empty() {
            return None;
        }
        for _ in 0..self.n {
            let i = self.rr.fetch_add(1, Ordering::Relaxed) % self.n;
            if candidates.contains(&i) {
                return Some(i);
            }
        }
        candidates.first().copied()
    }

    /// Smooth weighted round-robin (nginx-style) over the healthy subset, using
    /// the STATIC config weights. Deterministic, no schedule allocation.
    fn pick_wrr(&self, mask: u64, n: usize) -> Option<usize> {
        let mut cur = self.wrr_current.lock().unwrap();
        let mut total = 0.0;
        let mut best: Option<usize> = None;
        for i in 0..n {
            if !Reachability::bit(mask, i) {
                continue;
            }
            cur[i] += self.weights[i];
            total += self.weights[i];
            if best.is_none_or(|b| cur[i] > cur[b]) {
                best = Some(i);
            }
        }
        if let Some(b) = best {
            cur[b] -= total;
        }
        best
    }
}

fn effective_b(s: &ReplicaState) -> f64 {
    if !s.degraded_metrics_only && s.n_slots > 0 {
        s.b_metrics.max(s.b_slots_active as f64)
    } else {
        s.b_metrics
    }
}

fn admission_guard_reason(s: &ReplicaState) -> Option<&'static str> {
    let deferred = s.requests_deferred > 0.0;
    let slots_full = !s.degraded_metrics_only && s.n_slots > 0 && s.b_slots_active >= s.n_slots;
    match (deferred, slots_full) {
        (true, true) => Some("deferred_and_slots_full"),
        (true, false) => Some("requests_deferred"),
        (false, true) => Some("slots_full"),
        (false, false) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replica::ReplicaState;
    use std::time::Instant;

    fn state_aged(name: &str, weight: f64, u: f64, b: f64, age: Duration) -> ReplicaState {
        ReplicaState {
            name: name.into(),
            weight,
            u,
            resident_tokens: 0,
            max_slot_tokens: 0,
            context_budget: 4096.0,
            b_metrics: b,
            b_slots_active: 0,
            n_slots: 4,
            requests_deferred: 0.0,
            c0: 0.0,
            c1: 0.0,
            c2: 0.0,
            c3: 0.0,
            degraded_metrics_only: false,
            updated_at: Instant::now().checked_sub(age).unwrap_or_else(Instant::now),
        }
    }

    fn state(name: &str, weight: f64, u: f64, b: f64) -> ReplicaState {
        state_aged(name, weight, u, b, Duration::ZERO)
    }

    #[allow(clippy::too_many_arguments)]
    fn v2_state(
        name: &str,
        weight: f64,
        u: f64,
        b_metrics: f64,
        b_slots_active: u64,
        n_slots: u64,
        requests_deferred: f64,
        degraded_metrics_only: bool,
    ) -> ReplicaState {
        let mut s = state(name, weight, u, b_metrics);
        s.b_slots_active = b_slots_active;
        s.n_slots = n_slots;
        s.requests_deferred = requests_deferred;
        s.degraded_metrics_only = degraded_metrics_only;
        s
    }

    /// Builder for cost_model tests: sets raw B, raw resident_tokens R, and the
    /// per-node coefficients (weight is deliberately left at 1.0 and unused).
    #[allow(clippy::too_many_arguments)]
    fn cost_state_aged(
        name: &str,
        b: f64,
        r: u64,
        c0: f64,
        c1: f64,
        c2: f64,
        age: Duration,
    ) -> ReplicaState {
        let mut s = state_aged(name, 1.0, 0.0, b, age);
        s.resident_tokens = r;
        s.c0 = c0;
        s.c1 = c1;
        s.c2 = c2;
        s
    }

    fn cost_state(name: &str, b: f64, r: u64, c0: f64, c1: f64, c2: f64) -> ReplicaState {
        cost_state_aged(name, b, r, c0, c1, c2, Duration::ZERO)
    }

    /// max_state_age effectively infinite -> nothing is ever stale (existing
    /// argmin behavior unchanged); poll_interval arbitrary.
    fn setup(states: Vec<ReplicaState>, eligible: &[bool]) -> (StateCache, SelectCtx) {
        setup_ex(
            states,
            eligible,
            Duration::from_secs(3600),
            Duration::from_millis(350),
        )
    }

    fn setup_ex(
        states: Vec<ReplicaState>,
        eligible: &[bool],
        max_state_age: Duration,
        poll_interval: Duration,
    ) -> (StateCache, SelectCtx) {
        let weights: Vec<f64> = states.iter().map(|s| s.weight).collect();
        let reach = Arc::new(Reachability::new(states.len(), 3));
        for (i, &e) in eligible.iter().enumerate() {
            if e {
                reach.on_poll_success(i);
            }
        }
        let ctx = SelectCtx::new(weights, reach, 42, max_state_age, poll_interval);
        (StateCache::new(states), ctx)
    }

    #[test]
    fn least_connections_picks_fewest_inflight() {
        let (cache, ctx) = setup(
            vec![
                state("a", 1.0, 0.5, 3.0),
                state("b", 1.0, 0.5, 1.0),
                state("c", 1.0, 0.5, 2.0),
            ],
            &[true, true, true],
        );
        assert_eq!(
            Policy::LeastConnections.select(&cache, &ctx).choice,
            Some(1)
        );
    }

    #[test]
    fn state_aware_picks_lowest_u_over_w() {
        let (cache, ctx) = setup(
            vec![
                state("a", 1.0, 0.9, 0.0),
                state("b", 2.0, 0.9, 0.0), // u/w = 0.45 -> winner
                state("c", 1.0, 0.5, 0.0), // u/w = 0.5
            ],
            &[true, true, true],
        );
        assert_eq!(Policy::StateAware.select(&cache, &ctx).choice, Some(1));
    }

    #[test]
    fn state_aware_v2_is_parsed_and_reads_state() {
        assert_eq!(Policy::parse("state_aware_v2"), Ok(Policy::StateAwareV2));
        assert_eq!(Policy::StateAwareV2.as_str(), "state_aware_v2");
        assert!(Policy::StateAwareV2.reads_state());
    }

    #[test]
    fn cost_predict_v3_is_parsed_and_reads_state() {
        assert_eq!(Policy::parse("cost_predict_v3"), Ok(Policy::CostPredictV3));
        assert_eq!(Policy::CostPredictV3.as_str(), "cost_predict_v3");
        assert!(Policy::CostPredictV3.reads_state());
    }

    #[test]
    fn v2_stage1_excludes_deferred_and_full_nodes_when_capacity_exists() {
        let states = vec![
            v2_state("deferred", 1.0, 0.01, 1.0, 1, 4, 1.0, false),
            v2_state("full", 1.0, 0.02, 4.0, 4, 4, 0.0, false),
            v2_state("capacity", 1.0, 0.90, 2.0, 2, 4, 0.0, false),
        ];
        let (cache, ctx) = setup(states, &[true, true, true]);
        let d = Policy::StateAwareV2.select(&cache, &ctx);
        assert_eq!(d.choice, Some(2));
        let logs = d.v2_candidate_logs.as_ref().unwrap();
        assert_eq!(logs[0].stage1_exclusion_reason, Some("requests_deferred"));
        assert_eq!(logs[1].stage1_exclusion_reason, Some("slots_full"));
        assert_eq!(logs[2].stage1_exclusion_reason, None);
    }

    #[test]
    fn v2_stage1_falls_through_when_every_candidate_is_full() {
        let states = vec![
            v2_state("full-high-u", 1.0, 0.8, 4.0, 4, 4, 0.0, false),
            v2_state("full-low-u", 1.0, 0.2, 4.0, 4, 4, 0.0, false),
        ];
        let (cache, ctx) = setup(states, &[true, true]);
        let d = Policy::StateAwareV2.select(&cache, &ctx);
        assert_eq!(d.choice, Some(1));
        let logs = d.v2_candidate_logs.as_ref().unwrap();
        assert!(logs.iter().all(|log| log.stage1_guard_fallback));
        assert!(logs.iter().all(|log| log.stage1_exclusion_reason.is_none()));
    }

    #[test]
    fn v2_stage1_does_not_treat_slots_degraded_zero_as_full() {
        let states = vec![
            v2_state("degraded", 1.0, 0.1, 1.0, 0, 0, 0.0, true),
            v2_state("full", 1.0, 0.9, 4.0, 4, 4, 0.0, false),
        ];
        let (cache, ctx) = setup(states, &[true, true]);
        let d = Policy::StateAwareV2.select(&cache, &ctx);
        assert_eq!(d.choice, Some(0));
        let logs = d.v2_candidate_logs.as_ref().unwrap();
        assert_eq!(logs[0].stage1_exclusion_reason, None);
        assert_eq!(
            logs[0].b_eff, 1.0,
            "degraded mode must ignore b_slots_active"
        );
        assert_eq!(logs[1].stage1_exclusion_reason, Some("slots_full"));
    }

    #[test]
    fn v2_stage2_uses_each_candidates_weight_for_tolerance() {
        let states = vec![
            v2_state("min", 1.0, 0.4, 0.0, 0, 4, 0.0, false), // load 0
            v2_state("heavy", 4.0, 0.1, 4.0, 4, 8, 0.0, false), // load 1; tolerance .25 -> out
            v2_state("one-unit", 1.0, 0.8, 1.0, 1, 4, 0.0, false), // load 1; tolerance 1 -> in
        ];
        let (cache, ctx) = setup(states, &[true, true, true]);
        let d = Policy::StateAwareV2.select(&cache, &ctx);
        let logs = d.v2_candidate_logs.as_ref().unwrap();
        assert!(logs[0].shortlisted);
        assert!(!logs[1].shortlisted);
        assert!(logs[2].shortlisted);
        assert_eq!(d.choice, Some(0));
    }

    #[test]
    fn v2_b_eff_uses_max_of_metrics_and_valid_slot_count() {
        let states = vec![
            v2_state("slots-ahead", 1.0, 0.2, 1.0, 3, 4, 0.0, false),
            v2_state("metrics-ahead", 1.0, 0.8, 2.0, 1, 4, 0.0, false),
        ];
        let (cache, ctx) = setup(states, &[true, true]);
        let d = Policy::StateAwareV2.select(&cache, &ctx);
        let logs = d.v2_candidate_logs.as_ref().unwrap();
        assert_eq!(logs[0].b_eff, 3.0);
        assert_eq!(logs[1].b_eff, 2.0);
    }

    #[test]
    fn v2_stage3_uses_low_u_to_distinguish_short_from_long_load() {
        let states = vec![
            v2_state("four-shorts", 1.0, 0.20, 4.0, 4, 4, 0.0, false),
            v2_state("one-long", 1.0, 0.80, 4.0, 4, 4, 0.0, false),
        ];
        let (cache, ctx) = setup(states, &[true, true]);
        let d = Policy::StateAwareV2.select(&cache, &ctx);
        assert_eq!(d.choice, Some(0));
        assert!(d
            .v2_candidate_logs
            .as_ref()
            .unwrap()
            .iter()
            .all(|log| log.shortlisted));
    }

    #[test]
    fn v2_tie_breaks_by_weighted_load_then_round_robin() {
        let weighted_states = vec![
            v2_state("higher-load", 1.0, 0.5, 1.0, 1, 4, 0.0, false),
            v2_state("lower-load", 1.0, 0.5, 0.0, 0, 4, 0.0, false),
        ];
        let (cache, ctx) = setup(weighted_states, &[true, true]);
        assert_eq!(Policy::StateAwareV2.select(&cache, &ctx).choice, Some(1));

        let exact_ties = || {
            vec![
                v2_state("a", 1.0, 0.5, 0.0, 0, 4, 0.0, false),
                v2_state("b", 1.0, 0.5, 0.0, 0, 4, 0.0, false),
            ]
        };
        let (cache, ctx) = setup(exact_ties(), &[true, true]);
        let first = Policy::StateAwareV2.select(&cache, &ctx);
        let first_choice = first.choice;
        drop(first);
        let second = Policy::StateAwareV2.select(&cache, &ctx);
        assert_ne!(
            first_choice, second.choice,
            "exact ties must rotate, not favor index zero"
        );
    }

    #[test]
    fn concurrent_v2_selections_observe_each_others_reservations() {
        use std::sync::{mpsc, Barrier};
        use std::thread;

        const N: usize = 8;
        let states = vec![
            v2_state("preferred-low-u", 1.0, 0.0, 0.0, 0, 4, 0.0, false),
            v2_state("fallback-high-u", 1.0, 1.0, 0.0, 0, 4, 0.0, false),
        ];
        let (cache, ctx) = setup(states, &[true, true]);
        let cache = Arc::new(cache);
        let ctx = Arc::new(ctx);
        let start = Arc::new(Barrier::new(N + 1));
        let hold = Arc::new(Barrier::new(N + 1));
        let (tx, rx) = mpsc::channel();

        let workers: Vec<_> = (0..N)
            .map(|_| {
                let cache = cache.clone();
                let ctx = ctx.clone();
                let start = start.clone();
                let hold = hold.clone();
                let tx = tx.clone();
                thread::spawn(move || {
                    start.wait();
                    let decision = Policy::StateAwareV2.select(&cache, &ctx);
                    tx.send(decision.choice.unwrap()).unwrap();
                    hold.wait();
                    drop(decision);
                })
            })
            .collect();
        drop(tx);

        start.wait();
        let choices: Vec<usize> = (0..N).map(|_| rx.recv().unwrap()).collect();
        let counts = ctx.pending_dispatches().counts();
        assert_eq!(counts.iter().sum::<u64>(), N as u64);
        assert!(
            choices.contains(&1),
            "without mutually observed reservations every request would choose the low-u node"
        );
        hold.wait();
        for worker in workers {
            worker.join().unwrap();
        }
        assert_eq!(ctx.pending_dispatches().counts(), vec![0, 0]);
    }

    #[test]
    fn pending_dispatches_do_not_change_existing_policy_decisions() {
        let policies = [
            Policy::Random,
            Policy::RoundRobin,
            Policy::LeastConnections,
            Policy::WeightedRoundRobin,
            Policy::WeightedLeastConnections,
            Policy::StateAware,
            Policy::CostModel,
        ];
        for policy in policies {
            let states = vec![
                v2_state("a", 3.0, 0.1, 1.0, 1, 4, 0.0, false),
                v2_state("b", 1.0, 0.8, 2.0, 2, 4, 0.0, false),
            ];
            let (cache_pending, ctx_pending) = setup(states.clone(), &[true, true]);
            let (cache_clean, ctx_clean) = setup(states, &[true, true]);
            let pending = ctx_pending.pending_dispatches();
            let held = pending.select_and_reserve(|_| (Some(0), ())).0.unwrap();

            assert_eq!(
                policy.select(&cache_pending, &ctx_pending).choice,
                policy.select(&cache_clean, &ctx_clean).choice,
                "{policy:?} must not read v2 pending state"
            );
            drop(held);
        }
    }

    #[test]
    fn v2_excludes_stale_replica_and_uses_shared_all_stale_fallback() {
        let fresh = v2_state("fresh", 1.0, 0.9, 2.0, 2, 4, 0.0, false);
        let mut stale = v2_state("stale-low-u", 1.0, 0.1, 0.0, 0, 4, 0.0, false);
        stale.updated_at = Instant::now() - Duration::from_secs(10);
        let (cache, ctx) = setup_ex(
            vec![fresh, stale],
            &[true, true],
            Duration::from_secs(1),
            Duration::from_millis(350),
        );
        let d = Policy::StateAwareV2.select(&cache, &ctx);
        assert_eq!(d.choice, Some(0));
        assert_eq!(d.stale_mode, StaleMode::SomeStaleExcluded);

        let states = vec![
            state_aged("older", 1.0, 0.1, 0.0, Duration::from_secs(10)),
            state_aged("newer", 1.0, 0.9, 0.0, Duration::from_secs(2)),
        ];
        let (cache, ctx) = setup_ex(
            states,
            &[true, true],
            Duration::from_secs(1),
            Duration::from_millis(350),
        );
        let d = Policy::StateAwareV2.select(&cache, &ctx);
        assert_eq!(d.stale_mode, StaleMode::AllStaleRr);
        assert_eq!(
            d.choice,
            Some(0),
            "shared blind RR fallback starts at index zero"
        );
    }

    #[test]
    fn weighted_least_connections_uses_ratio() {
        let (cache, ctx) = setup(
            vec![
                state("a", 1.0, 0.0, 2.0), // 2/1 = 2
                state("b", 4.0, 0.0, 3.0), // 3/4 = 0.75 -> winner
                state("c", 2.0, 0.0, 2.0), // 2/2 = 1
            ],
            &[true, true, true],
        );
        assert_eq!(
            Policy::WeightedLeastConnections.select(&cache, &ctx).choice,
            Some(1)
        );
    }

    #[test]
    fn unhealthy_replicas_are_excluded() {
        // Best score is index 0, but it is unhealthy -> must not be chosen.
        let (cache, ctx) = setup(
            vec![
                state("a", 1.0, 0.1, 0.0),
                state("b", 1.0, 0.8, 5.0),
                state("c", 1.0, 0.3, 1.0),
            ],
            &[false, true, true],
        );
        let choice = Policy::StateAware.select(&cache, &ctx).choice.unwrap();
        assert_ne!(choice, 0);
        assert_eq!(choice, 2); // lowest u among healthy
                               // least_connections likewise avoids the unhealthy best.
        assert_eq!(
            Policy::LeastConnections.select(&cache, &ctx).choice,
            Some(2)
        );
    }

    #[test]
    fn all_unhealthy_yields_no_choice() {
        let (cache, ctx) = setup(vec![state("a", 1.0, 0.1, 0.0)], &[false]);
        assert_eq!(Policy::StateAware.select(&cache, &ctx).choice, None);
        assert_eq!(Policy::StateAwareV2.select(&cache, &ctx).choice, None);
        assert_eq!(
            Policy::CostPredictV3
                .select_predictive(&cache, &ctx, 40)
                .choice,
            None
        );
        assert_eq!(ctx.pending_dispatches().counts(), vec![0]);
        assert_eq!(Policy::RoundRobin.select(&cache, &ctx).choice, None);
        assert_eq!(Policy::Random.select(&cache, &ctx).choice, None);
    }

    #[test]
    fn round_robin_cycles_and_skips_unhealthy() {
        let (cache, ctx) = setup(
            vec![
                state("a", 1.0, 0.0, 0.0),
                state("b", 1.0, 0.0, 0.0),
                state("c", 1.0, 0.0, 0.0),
            ],
            &[true, false, true], // b is down
        );
        // Should alternate between the two healthy nodes 0 and 2, never 1.
        let picks: Vec<usize> = (0..6)
            .map(|_| Policy::RoundRobin.select(&cache, &ctx).choice.unwrap())
            .collect();
        assert!(picks.iter().all(|&i| i == 0 || i == 2));
        assert!(picks.contains(&0) && picks.contains(&2));
    }

    #[test]
    fn stateless_policies_do_not_attach_snapshot() {
        // Confirms the measurement asymmetry: stateless policies return no
        // snapshot (they never loaded one inside the timed region).
        let (cache, ctx) = setup(vec![state("a", 1.0, 0.0, 0.0)], &[true]);
        assert!(Policy::RoundRobin.select(&cache, &ctx).snapshot.is_none());
        assert!(Policy::Random.select(&cache, &ctx).snapshot.is_none());
        assert!(Policy::WeightedRoundRobin
            .select(&cache, &ctx)
            .snapshot
            .is_none());
        assert!(Policy::StateAware.select(&cache, &ctx).snapshot.is_some());
        assert!(Policy::LeastConnections
            .select(&cache, &ctx)
            .snapshot
            .is_some());
    }

    #[test]
    fn smooth_wrr_respects_weights() {
        // weights 3:1 over two healthy nodes -> node 0 chosen 3x per 4.
        let (cache, ctx) = setup(
            vec![state("a", 3.0, 0.0, 0.0), state("b", 1.0, 0.0, 0.0)],
            &[true, true],
        );
        let mut count = [0usize; 2];
        for _ in 0..8 {
            let i = Policy::WeightedRoundRobin
                .select(&cache, &ctx)
                .choice
                .unwrap();
            count[i] += 1;
        }
        assert_eq!(count[0], 6);
        assert_eq!(count[1], 2);
    }

    #[test]
    fn stale_replica_never_wins_against_live() {
        // index 0 is fresh with a WORSE (higher) u; index 1 is stale with a
        // tempting low u. The stale one must be excluded, not chosen.
        let states = vec![
            state_aged("fresh", 1.0, 0.9, 0.0, Duration::from_millis(100)),
            state_aged("stale", 1.0, 0.1, 0.0, Duration::from_secs(10)),
        ];
        let (cache, ctx) = setup_ex(
            states,
            &[true, true],
            Duration::from_secs(1),
            Duration::from_millis(350),
        );
        let d = Policy::StateAware.select(&cache, &ctx);
        assert_eq!(
            d.choice,
            Some(0),
            "must pick the fresh replica despite its higher u"
        );
        assert_eq!(d.stale_mode, StaleMode::SomeStaleExcluded);
    }

    #[test]
    fn all_stale_same_generation_compares_last_known() {
        // Both stale, ages within one poll interval of each other (both ~10s,
        // 50ms apart) -> compare last-known, pick lower u (index 1).
        let states = vec![
            state_aged("a", 1.0, 0.9, 0.0, Duration::from_millis(10_000)),
            state_aged("b", 1.0, 0.1, 0.0, Duration::from_millis(10_050)),
        ];
        let (cache, ctx) = setup_ex(
            states,
            &[true, true],
            Duration::from_secs(1),
            Duration::from_millis(350),
        );
        let d = Policy::StateAware.select(&cache, &ctx);
        assert_eq!(
            d.choice,
            Some(1),
            "same-generation stale: compare, lowest u wins"
        );
        assert_eq!(d.stale_mode, StaleMode::AllStaleCompare);
    }

    #[test]
    fn all_stale_divergent_ages_fall_back_to_rr() {
        // Both stale but ages differ by ~8s >> poll_interval -> not comparable,
        // round-robin over eligible (index 0 first).
        let states = vec![
            state_aged("a", 1.0, 0.9, 0.0, Duration::from_millis(2_000)),
            state_aged("b", 1.0, 0.1, 0.0, Duration::from_millis(10_000)),
        ];
        let (cache, ctx) = setup_ex(
            states,
            &[true, true],
            Duration::from_secs(1),
            Duration::from_millis(350),
        );
        let d = Policy::StateAware.select(&cache, &ctx);
        assert_eq!(d.stale_mode, StaleMode::AllStaleRr);
        // RR starts at index 0; the low-u stale replica does NOT get to win.
        assert_eq!(d.choice, Some(0));
    }

    #[test]
    fn cost_model_prefers_lower_cost_when_b_and_r_oppose() {
        // Same (Pi 4B) coefficients for both: A = 47*1 + 0.070*800 = 103.0; B = 47*3 + 0.070*150 = 151.5.
        // A wins (c0 is the same for both, so it does not change the order).
        let a = cost_state("A", 1.0, 800, 0.0, 47.0, 0.070);
        let b = cost_state("B", 3.0, 150, 0.0, 47.0, 0.070);
        assert!((Policy::CostModel.score(&a).unwrap() - 103.0).abs() < 1e-9);
        assert!((Policy::CostModel.score(&b).unwrap() - 151.5).abs() < 1e-9);
        let (cache, ctx) = setup(vec![a, b], &[true, true]);
        assert_eq!(Policy::CostModel.select(&cache, &ctx).choice, Some(0));
    }

    #[test]
    fn cost_model_applies_per_node_coefficients_not_global() {
        // Same load, different per-token cost: node0 = 82, node1 = 52, so node 1 wins.
        // With global coefficients they would tie and node 0 would win the tie-break.
        let n0 = cost_state("hi_c2", 1.0, 500, 0.0, 47.0, 0.070);
        let n1 = cost_state("lo_c2", 1.0, 500, 0.0, 47.0, 0.010);
        assert!((Policy::CostModel.score(&n0).unwrap() - 82.0).abs() < 1e-9);
        assert!((Policy::CostModel.score(&n1).unwrap() - 52.0).abs() < 1e-9);
        let (cache, ctx) = setup(vec![n0, n1], &[true, true]);
        assert_eq!(Policy::CostModel.select(&cache, &ctx).choice, Some(1));
    }

    #[test]
    fn cost_model_excludes_stale_replica() {
        // Fresh node with a HIGHER cost vs a stale node with a tempting low cost.
        // The stale one must be excluded (uniform stale rule), not chosen.
        let fresh = cost_state_aged(
            "fresh",
            5.0,
            900,
            0.0,
            47.0,
            0.070,
            Duration::from_millis(100),
        );
        let stale = cost_state_aged("stale", 1.0, 10, 0.0, 47.0, 0.070, Duration::from_secs(10));
        let (cache, ctx) = setup_ex(
            vec![fresh, stale],
            &[true, true],
            Duration::from_secs(1),
            Duration::from_millis(350),
        );
        let d = Policy::CostModel.select(&cache, &ctx);
        assert_eq!(
            d.choice,
            Some(0),
            "must pick the fresh replica despite its higher cost"
        );
        assert_eq!(d.stale_mode, StaleMode::SomeStaleExcluded);
    }

    #[test]
    fn v3_long_request_is_cheaper_on_pi5_than_loaded_pi4() {
        let pi5 = cost_state("pi5", 3.0, 1_000, 87.5638, 0.0, 0.013507);
        let pi4 = cost_state("pi4", 3.0, 1_000, 53.0, 47.0, 0.070);
        let (cache, ctx) = setup(vec![pi5, pi4], &[true, true]);
        let d = Policy::CostPredictV3.select_predictive(&cache, &ctx, 640);
        let logs = d.v3_candidate_logs.unwrap();

        assert_eq!(d.choice, Some(0));
        assert!(logs[0].predicted_tpot < logs[1].predicted_tpot);
        assert!((logs[0].predicted_tpot - 109.715_28).abs() < 1e-6);
        assert!((logs[1].predicted_tpot - 355.8).abs() < 1e-6);
    }

    #[test]
    fn v3_short_request_can_choose_either_node_from_projected_cost() {
        // Idle Pi 4B: 53 + 47*(0+1) + .070*40 = 102.8.
        // Loaded Pi 5: 87.5638 + .013507*(2000+40) = 115.11908 -> Pi 4B.
        let loaded_pi5 = cost_state("pi5", 4.0, 2_000, 87.5638, 0.0, 0.013507);
        let idle_pi4 = cost_state("pi4", 0.0, 0, 53.0, 47.0, 0.070);
        let (cache, ctx) = setup(vec![loaded_pi5, idle_pi4], &[true, true]);
        assert_eq!(
            Policy::CostPredictV3
                .select_predictive(&cache, &ctx, 40)
                .choice,
            Some(1)
        );

        // At lower resident load the Pi 5 score falls below 102.8, so it wins.
        let lighter_pi5 = cost_state("pi5", 4.0, 400, 87.5638, 0.0, 0.013507);
        let idle_pi4 = cost_state("pi4", 0.0, 0, 53.0, 47.0, 0.070);
        let (cache, ctx) = setup(vec![lighter_pi5, idle_pi4], &[true, true]);
        assert_eq!(
            Policy::CostPredictV3
                .select_predictive(&cache, &ctx, 40)
                .choice,
            Some(0)
        );
    }

    #[test]
    fn v3_clamps_projected_resident_tokens_to_context_budget() {
        let mut pi4 = cost_state("pi4", 1.0, 4_000, 53.0, 47.0, 0.070);
        pi4.context_budget = 4096.0;
        let log = v3_prediction(&pi4, 500);
        assert_eq!(log.r_proj_unclamped, 4_500);
        assert_eq!(log.r_proj, 4_096);
        assert!(log.r_proj_clamped);
        assert!((log.predicted_tpot - (53.0 + 47.0 * 2.0 + 0.070 * 4096.0)).abs() < 1e-9);
    }

    #[test]
    fn congestion_matches_v3_when_nothing_is_deferred() {
        let s = cost_state("pi4", 1.0, 800, 53.0, 47.0, 0.070);
        assert_eq!(s.requests_deferred, 0.0);
        let v3 = v3_prediction(&s, 100);
        let cong = congestion_prediction(&s, 100);
        assert!((v3.predicted_tpot - cong.predicted_tpot).abs() < 1e-9);
    }

    #[test]
    fn congestion_penalty_is_weight_times_deferred_squared() {
        let mut s = cost_state("pi4", 1.0, 800, 53.0, 47.0, 0.070);
        s.requests_deferred = 2.0;
        let base = v3_prediction(&s, 100).predicted_tpot;
        let cong = congestion_prediction(&s, 100).predicted_tpot;
        assert!((cong - base - CONGESTION_WEIGHT_MS * 4.0).abs() < 1e-9);
    }

    #[test]
    fn congestion_prefers_the_less_backed_up_node_even_if_nominally_slower() {
        // "fast" node projects a lower base cost but has a real queue backing
        // up behind it; "slow" node is nominally worse per-token but has
        // nothing deferred. This is exactly the starvation scenario the
        // simulator's length_aware_congestion was built to avoid.
        let mut fast_but_backed_up = cost_state("fast", 1.0, 200, 20.0, 47.0, 0.070);
        fast_but_backed_up.requests_deferred = 3.0;
        let slow_but_clear = cost_state("slow", 1.0, 200, 40.0, 47.0, 0.070);

        let a = congestion_prediction(&fast_but_backed_up, 100).predicted_tpot;
        let b = congestion_prediction(&slow_but_clear, 100).predicted_tpot;
        assert!(b < a, "clear node should now score better than the backed-up one");

        // Sanity: v3 (no congestion awareness) would have picked the other way.
        let a_v3 = v3_prediction(&fast_but_backed_up, 100).predicted_tpot;
        let b_v3 = v3_prediction(&slow_but_clear, 100).predicted_tpot;
        assert!(a_v3 < b_v3, "v3 should still prefer the nominally-faster node");
    }

    #[test]
    fn length_aware_congestion_is_parsed_and_reads_state() {
        assert_eq!(
            Policy::parse("length_aware_congestion"),
            Ok(Policy::LengthAwareCongestion)
        );
        assert_eq!(Policy::LengthAwareCongestion.as_str(), "length_aware_congestion");
        assert!(Policy::LengthAwareCongestion.reads_state());
    }

    #[test]
    fn v3_uses_pi5_zero_c1_and_pi4_c1_47() {
        let pi5_b0 = cost_state("pi5", 0.0, 500, 87.5638, 0.0, 0.013507);
        let pi5_b3 = cost_state("pi5", 3.0, 500, 87.5638, 0.0, 0.013507);
        assert!(
            (v3_prediction(&pi5_b0, 40).predicted_tpot - v3_prediction(&pi5_b3, 40).predicted_tpot)
                .abs()
                < 1e-12
        );

        let pi4_b0 = cost_state("pi4", 0.0, 500, 53.0, 47.0, 0.070);
        let pi4_b3 = cost_state("pi4", 3.0, 500, 53.0, 47.0, 0.070);
        assert!(
            (v3_prediction(&pi4_b3, 40).predicted_tpot
                - v3_prediction(&pi4_b0, 40).predicted_tpot
                - 3.0 * 47.0)
                .abs()
                < 1e-9
        );
    }

    #[test]
    fn v3_ignores_weight_and_excludes_stale_replica() {
        let mut fresh = cost_state_aged(
            "fresh",
            3.0,
            1_000,
            53.0,
            47.0,
            0.070,
            Duration::from_millis(100),
        );
        fresh.weight = 0.001;
        let mut stale = cost_state_aged(
            "stale-cheap",
            0.0,
            0,
            1.0,
            0.0,
            0.001,
            Duration::from_secs(10),
        );
        stale.weight = 1_000.0;
        let (cache, ctx) = setup_ex(
            vec![fresh, stale],
            &[true, true],
            Duration::from_secs(1),
            Duration::from_millis(350),
        );
        let d = Policy::CostPredictV3.select_predictive(&cache, &ctx, 40);
        assert_eq!(d.choice, Some(0));
        assert_eq!(d.stale_mode, StaleMode::SomeStaleExcluded);
    }

    #[test]
    fn v3_selection_has_no_side_effect_on_existing_state_policies() {
        let states = vec![
            cost_state("a", 1.0, 800, 53.0, 47.0, 0.070),
            cost_state("b", 3.0, 150, 53.0, 47.0, 0.070),
        ];
        for policy in [
            Policy::LeastConnections,
            Policy::WeightedLeastConnections,
            Policy::StateAware,
            Policy::CostModel,
            Policy::StateAwareV2,
        ] {
            let (cache, ctx) = setup(states.clone(), &[true, true]);
            let before = policy.select(&cache, &ctx).choice;
            let v3 = Policy::CostPredictV3.select_predictive(&cache, &ctx, 640);
            drop(v3);
            let after = policy.select(&cache, &ctx).choice;
            assert_eq!(before, after, "{policy:?} changed after a v3 decision");
        }
    }

    #[test]
    fn stale_mode_uniform_across_state_policies() {
        // The same stale exclusion applies to lc / wlc / state_aware.
        let make = || {
            vec![
                state_aged("fresh", 1.0, 0.9, 9.0, Duration::from_millis(100)),
                state_aged("stale", 1.0, 0.1, 0.0, Duration::from_secs(10)),
            ]
        };
        for policy in [
            Policy::LeastConnections,
            Policy::WeightedLeastConnections,
            Policy::StateAware,
            Policy::CostPredictV3,
        ] {
            let (cache, ctx) = setup_ex(
                make(),
                &[true, true],
                Duration::from_secs(1),
                Duration::from_millis(350),
            );
            let d = if policy == Policy::CostPredictV3 {
                policy.select_predictive(&cache, &ctx, 40)
            } else {
                policy.select(&cache, &ctx)
            };
            assert_eq!(
                d.choice,
                Some(0),
                "{policy:?} must exclude the stale replica"
            );
            assert_eq!(d.stale_mode, StaleMode::SomeStaleExcluded, "{policy:?}");
        }
    }

    #[test]
    fn tracked_policies_parse_and_round_trip() {
        for name in ["state_aware_tracked", "wlc_tracked"] {
            let p = Policy::parse(name).unwrap();
            assert_eq!(p.as_str(), name);
            assert!(p.is_tracked());
            assert!(!p.reads_state());
            assert_eq!(p.score(&state("a", 1.0, 0.5, 1.0)), None);
        }
        assert!(!Policy::StateAware.is_tracked());
    }

    #[test]
    fn state_aware_tracked_uses_weighted_tracked_context() {
        // Polled u says "a" is empty; the tracked view must win regardless.
        let (_cache, ctx) = setup(
            vec![state("a", 3.0, 0.0, 0.0), state("b", 1.0, 0.0, 0.0)],
            &[true, true],
        );
        let ctx = ctx.with_context_budgets(vec![4096.0, 4096.0]);
        let tracker = ctx.tracker();
        let (_a1, _) = tracker.track(0, 2000); // a: 2000/4096/3 = 0.163
        let (_b1, _) = tracker.track(1, 640); // b: 640/4096/1 = 0.156
        let d = Policy::StateAwareTracked.select_tracked(&ctx, 40);
        assert_eq!(d.choice, Some(1));
        assert_eq!(d.stale_mode, StaleMode::Tracked);
        assert!(d.snapshot.is_none());
        let view = d.tracked_view.as_ref().unwrap();
        assert_eq!(view[0].tokens, 2000);
        assert_eq!(view[1].tokens, 640);
        // The new request is tracked on the chosen replica inside selection.
        assert_eq!(tracker.loads()[1].tokens, 680);
        assert!(d.tracking.is_some());
    }

    #[test]
    fn wlc_tracked_uses_weighted_outstanding_requests() {
        let (_cache, ctx) = setup(
            vec![state("a", 3.0, 0.0, 9.0), state("b", 1.0, 0.0, 0.0)],
            &[true, true],
        );
        let tracker = ctx.tracker();
        let held: Vec<_> = (0..3).map(|_| tracker.track(0, 10).0).collect();
        // a: 3/3 = 1.0, b: 0/1 = 0 -> b
        let d = Policy::WlcTracked.select_tracked(&ctx, 10);
        assert_eq!(d.choice, Some(1));
        // Dropping the decision releases b's entry; dropping `held` releases a's.
        drop(d);
        drop(held);
        // All released: tie at 0 -> lowest index.
        let d = Policy::WlcTracked.select_tracked(&ctx, 10);
        assert_eq!(d.choice, Some(0));
    }

    #[test]
    fn tracked_selection_respects_health_only() {
        let (_cache, ctx) = setup(
            vec![
                state_aged("a", 1.0, 0.0, 0.0, Duration::from_secs(60)),
                state("b", 1.0, 0.0, 0.0),
            ],
            &[false, true],
        );
        let d = Policy::WlcTracked.select_tracked(&ctx, 1);
        assert_eq!(d.choice, Some(1), "ineligible replica skipped");

        let (_cache, ctx) = setup(vec![state("a", 1.0, 0.0, 0.0)], &[false]);
        let d = Policy::StateAwareTracked.select_tracked(&ctx, 1);
        assert_eq!(d.choice, None);
        assert_eq!(d.stale_mode, StaleMode::NoEligible);
        assert!(d.tracking.is_none());
    }
}
