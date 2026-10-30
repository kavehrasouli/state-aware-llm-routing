//! Static replica identity (from config) and the live per-replica state the
//! poller refreshes each cycle.

use crate::config::ReplicaConfig;
use std::time::Instant;

/// Immutable per-replica config, resolved once at startup.
#[derive(Clone)]
pub struct Replica {
    pub name: String,
    pub base_url: String,
    pub slots_url: String,
    pub metrics_url: String,
    pub weight: f64,
    /// c — the per-node context budget (denominator of u = ΣR / c).
    pub context_budget: f64,
    /// Fitted cost coefficients in ms (c0 intercept, c1 per request, c2 per token);
    /// read only by the cost-model policies, 0.0 when absent.
    pub c0: f64,
    pub c1: f64,
    pub c2: f64,
    /// Experimental shape coefficient (ms/token, largest-slot-projected). Only
    /// `shape_aware` reads it. 0.0 (absent) = no shape adjustment.
    pub c3: f64,
}

impl Replica {
    pub fn from_config(c: &ReplicaConfig) -> Self {
        let base = c.url.trim_end_matches('/').to_string();
        Replica {
            name: c.name.clone(),
            slots_url: format!("{base}/slots"),
            metrics_url: format!("{base}/metrics"),
            base_url: base,
            weight: c.weight,
            context_budget: c.context_budget as f64,
            c0: c.c0.unwrap_or(0.0),
            c1: c.c1.unwrap_or(0.0),
            c2: c.c2.unwrap_or(0.0),
            c3: c.c3.unwrap_or(0.0),
        }
    }
}

// Some fields are kept only so each decision's inputs can be audited in the log.
#[allow(dead_code)]
/// Live state for one replica, produced by the poller and read by selection.
/// `b_metrics` (requests_processing from `/metrics`) and `b_slots_active` (busy slots
/// from `/slots`) agreed exactly in 99.84% of decisions; only `state_aware_v2` uses the latter.
#[derive(Clone)]
pub struct ReplicaState {
    pub name: String,
    pub weight: f64,
    /// u = resident_tokens / context_budget.
    pub u: f64,
    pub resident_tokens: u64,
    /// Resident tokens in the single largest busy slot (the "shape" signal).
    /// Only `shape_aware` reads it.
    pub max_slot_tokens: u64,
    pub context_budget: f64,
    /// requests_processing — feeds least_connections / weighted_least_connections.
    pub b_metrics: f64,
    /// Count of `/slots` entries with is_processing=true (used by state_aware_v2).
    pub b_slots_active: u64,
    pub n_slots: u64,
    pub requests_deferred: f64,
    /// Per-node cost coefficients (ms), copied from config each poll so fitted
    /// cost policies can read them without a replica-index side table.
    pub c0: f64,
    pub c1: f64,
    pub c2: f64,
    pub c3: f64,
    /// true when `/slots` was unavailable and u came from kv_cache_usage_ratio.
    pub degraded_metrics_only: bool,
    /// When /metrics answered; kept unchanged after a failed poll, so the state ages into stale.
    pub updated_at: Instant,
}

impl ReplicaState {
    /// Placeholder used to seed the cache before the first poll completes:
    /// unhealthy, so it is excluded from selection until warmed.
    pub fn placeholder(r: &Replica) -> Self {
        ReplicaState {
            name: r.name.clone(),
            weight: r.weight,
            u: 0.0,
            resident_tokens: 0,
            max_slot_tokens: 0,
            context_budget: r.context_budget,
            b_metrics: 0.0,
            b_slots_active: 0,
            n_slots: 0,
            requests_deferred: 0.0,
            c0: r.c0,
            c1: r.c1,
            c2: r.c2,
            c3: r.c3,
            degraded_metrics_only: false,
            updated_at: Instant::now(),
        }
    }
}
