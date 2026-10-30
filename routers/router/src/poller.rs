//! Background poller: every `poll_interval` it fetches `/slots` and `/metrics` from all replicas
//! concurrently and stores one cluster snapshot. A failed poll marks the state stale and never
//! evicts; a successful `/metrics` poll restores an evicted replica.

use crate::cache::StateCache;
use crate::parse::{parse_metric, parse_slots};
use crate::reachability::Reachability;
use crate::record::RecordWriter;
use crate::replica::{Replica, ReplicaState};
use crate::reservation::PendingDispatches;
use reqwest::Client;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub struct PollerCfg {
    pub poll_interval: Duration,
    pub poll_timeout: Duration,
    /// For emitting recovery events into the JSONL (see record.rs).
    pub writer: Arc<RecordWriter>,
    pub run_id: Arc<str>,
    pub policy: &'static str,
    pub pending_dispatches: Arc<PendingDispatches>,
}

pub async fn run(
    replicas: Arc<Vec<Replica>>,
    cache: Arc<StateCache>,
    reach: Arc<Reachability>,
    client: Client,
    cfg: PollerCfg,
) {
    let mut ticker = tokio::time::interval(cfg.poll_interval);
    // If a cycle overruns, don't fire a burst to catch up — just delay.
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    loop {
        ticker.tick().await;
        poll_cycle(
            &replicas,
            &cache,
            &reach,
            &client,
            cfg.poll_timeout,
            &cfg.writer,
            &cfg.run_id,
            cfg.policy,
            &cfg.pending_dispatches,
        )
        .await;
    }
}

/// One poll cycle: refresh the cache and apply reachability recovery + logging.
/// Extracted (and pub(crate)) so tests can drive a single cycle without the loop.
#[allow(clippy::too_many_arguments)]
pub async fn poll_cycle(
    replicas: &[Replica],
    cache: &StateCache,
    reach: &Reachability,
    client: &Client,
    poll_timeout: Duration,
    writer: &RecordWriter,
    run_id: &str,
    policy: &str,
    pending_dispatches: &PendingDispatches,
) {
    let prev = cache.load_full();
    let results = poll_all(replicas, client, poll_timeout, &prev).await;

    let mut states = Vec::with_capacity(results.len());
    let mut successful_polls = Vec::new();
    for (i, (state, metrics_ok, poll_started_at)) in results.into_iter().enumerate() {
        if metrics_ok {
            successful_polls.push((i, poll_started_at));
            // Successful poll: recover eligibility (only meaningful if it had
            // been dispatch-evicted). Log + emit an event on the transition.
            if reach.on_poll_success(i) {
                eprintln!(
                    "[poller] RECOVERED replica '{}' -> back in selection (successful /metrics poll)",
                    state.name
                );
                writer.write_event(run_id, policy, &state.name, "recovered", 0);
            }
        }
        // Transition-only degraded logging (no per-poll spam).
        if !prev[i].degraded_metrics_only && state.degraded_metrics_only {
            eprintln!(
                "[poller] WARN replica '{}' has NO /slots -> DEGRADED: u from kv_cache_usage_ratio, \
                 b_slots_active unavailable. Check llama-server flags (--no-slots?).",
                state.name
            );
        }
        states.push(state);
    }

    cache.store(states);
    // Publish first, then retire reservations whose dispatch could be reflected
    // by this successful poll. Guard Drop remains an idempotent fallback.
    for (i, poll_started_at) in successful_polls {
        pending_dispatches.release_observed_by_poll(i, poll_started_at);
    }
}

/// Poll every replica concurrently. Returns state, metrics success, and the
/// fetch-start timestamp used by pending-dispatch expiry.
pub async fn poll_all(
    replicas: &[Replica],
    client: &Client,
    poll_timeout: Duration,
    prev: &[ReplicaState],
) -> Vec<(ReplicaState, bool, Instant)> {
    let futs = replicas.iter().enumerate().map(|(i, r)| {
        let prev_state = prev.get(i).cloned();
        poll_one(r, client, poll_timeout, prev_state)
    });
    futures::future::join_all(futs).await
}

async fn poll_one(
    r: &Replica,
    client: &Client,
    poll_timeout: Duration,
    prev: Option<ReplicaState>,
) -> (ReplicaState, bool, Instant) {
    let poll_started_at = Instant::now();
    // Fetch both endpoints concurrently, so a poll takes at most about poll_timeout.
    let (metrics_res, slots_res) = tokio::join!(
        fetch_text(client, &r.metrics_url, poll_timeout),
        fetch_text(client, &r.slots_url, poll_timeout),
    );

    // /metrics failed: keep the last state and its old timestamp (so it ages into stale); no recovery.
    let metrics_text = match metrics_res {
        Ok(t) => t,
        Err(_e) => return (carry_stale(r, prev), false, poll_started_at),
    };
    let b_metrics = parse_metric(&metrics_text, "llamacpp:requests_processing").unwrap_or(0.0);
    let deferred = parse_metric(&metrics_text, "llamacpp:requests_deferred").unwrap_or(0.0);

    // Post-fetch timestamp: ≈ when /metrics actually answered.
    let now = Instant::now();

    let state = match slots_res {
        Ok(slots_text) => match parse_slots(&slots_text) {
            Ok(sp) => {
                let u = if r.context_budget > 0.0 {
                    sp.resident_tokens as f64 / r.context_budget
                } else {
                    0.0
                };
                ReplicaState {
                    name: r.name.clone(),
                    weight: r.weight,
                    u,
                    resident_tokens: sp.resident_tokens,
                    max_slot_tokens: sp.max_slot_tokens,
                    context_budget: r.context_budget,
                    b_metrics,
                    b_slots_active: sp.busy_slots,
                    n_slots: sp.n_slots,
                    requests_deferred: deferred,
                    c0: r.c0,
                    c1: r.c1,
                    c2: r.c2,
                    c3: r.c3,
                    degraded_metrics_only: false,
                    updated_at: now,
                }
            }
            Err(_e) => degraded(r, b_metrics, deferred, &metrics_text, now),
        },
        // /slots failure never affects health — only feeds degraded u.
        Err(_e) => degraded(r, b_metrics, deferred, &metrics_text, now),
    };
    (state, true, poll_started_at)
}

/// Degraded mode: /slots unusable but /metrics OK. Derive u from
/// kv_cache_usage_ratio (already ΣKV/total_ctx). Node stays serving.
fn degraded(
    r: &Replica,
    b_metrics: f64,
    deferred: f64,
    metrics_text: &str,
    now: Instant,
) -> ReplicaState {
    let ratio = parse_metric(metrics_text, "llamacpp:kv_cache_usage_ratio").unwrap_or(0.0);
    ReplicaState {
        name: r.name.clone(),
        weight: r.weight,
        u: ratio,
        resident_tokens: (ratio * r.context_budget) as u64,
        // No per-slot breakdown available in degraded mode (kv_cache_usage_ratio
        // is a single aggregate) -- 0 means shape_aware sees no concentration
        // bonus here, never a false one.
        max_slot_tokens: 0,
        context_budget: r.context_budget,
        b_metrics,
        b_slots_active: 0,
        n_slots: 0,
        requests_deferred: deferred,
        c0: r.c0,
        c1: r.c1,
        c2: r.c2,
        c3: r.c3,
        degraded_metrics_only: true,
        updated_at: now,
    }
}

/// /metrics failed this cycle: keep the previous state VERBATIM (including its
/// old `updated_at`, so `now - updated_at` grows and `select` sees it as stale).
/// Eligibility is not touched here — only the data path evicts.
fn carry_stale(r: &Replica, prev: Option<ReplicaState>) -> ReplicaState {
    prev.unwrap_or_else(|| ReplicaState::placeholder(r))
}

async fn fetch_text(client: &Client, url: &str, timeout: Duration) -> Result<String, String> {
    let resp = client
        .get(url)
        .timeout(timeout)
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("HTTP {}", resp.status()));
    }
    resp.text().await.map_err(|e| e.to_string())
}
