//! The shared forwarding path: identical for all policies except the timed `select` call.
//! Timeouts are recorded as failures and never retried or re-routed.

use crate::cache::StateCache;
use crate::policy::{Policy, SelectCtx};
use crate::reachability::Reachability;
use crate::record::{Outcome, RecordWriter, RequestRecord, StateVectorEntry, V3RequestLog};
use crate::replica::{Replica, ReplicaState};
use crate::request_tokens::{estimate_request_tokens, RequestTokenEstimate};
use crate::stream::instrumented_stream;
use axum::{
    body::Body,
    extract::State,
    http::{
        header::{ACCEPT, CONTENT_TYPE},
        HeaderMap, HeaderName, HeaderValue, StatusCode, Uri,
    },
    response::{IntoResponse, Response},
};

/// Optional client-supplied exact prompt token count (pre-tokenized offline
/// with the served model). Feeds the router-tracked load; absent, the
/// router falls back to its hybrid estimate or to 0.
const PROMPT_TOKENS_HEADER: &str = "x-prompt-tokens";

fn header_prompt_tokens(headers: &HeaderMap) -> Option<u64> {
    headers
        .get(PROMPT_TOKENS_HEADER)?
        .to_str()
        .ok()?
        .trim()
        .parse()
        .ok()
}
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

struct V3PreparedBody {
    bytes: bytes::Bytes,
    estimate: RequestTokenEstimate,
    body_read_ns: u64,
    estimate_ns: u64,
}

async fn prepare_v3_body(body: Body) -> Result<V3PreparedBody, String> {
    let body_read_start = Instant::now();
    let bytes = axum::body::to_bytes(body, usize::MAX)
        .await
        .map_err(|e| format!("failed to read request body: {e}"))?;
    let body_read_ns = body_read_start.elapsed().as_nanos() as u64;

    let estimate_start = Instant::now();
    let estimate = estimate_request_tokens(&bytes)?;
    let estimate_ns = estimate_start.elapsed().as_nanos() as u64;
    Ok(V3PreparedBody {
        bytes,
        estimate,
        body_read_ns,
        estimate_ns,
    })
}

pub struct AppState {
    pub client: reqwest::Client,
    /// Identifies this router run; logged per record so (run_id, request_id) is unique across restarts.
    pub run_id: Arc<str>,
    pub replicas: Vec<Replica>,
    pub cache: Arc<StateCache>,
    pub policy: Policy,
    pub select_ctx: SelectCtx,
    /// Shared with SelectCtx (mask reads) and the poller (recovery); the data
    /// path writes evictions here.
    pub reachability: Arc<Reachability>,
    pub writer: Arc<RecordWriter>,
    pub next_id: AtomicU64,
    pub upstream_timeout: Duration,
    pub stream_idle_timeout: Duration,
    /// Cached-state staleness threshold, for per-decision logging.
    pub max_state_age: Duration,
    /// `--verbose`: print a per-request decision line to STDERR (never JSONL).
    pub verbose: bool,
}

/// `--verbose` helper: compact, greppable per-candidate score line. Uses the
/// canonical `Policy::score` so it can't drift from the real argmin. Runs
/// OUTSIDE the SQ2 timed region, and only when `--verbose` is set.
fn verbose_scores(
    policy: Policy,
    snap: &[ReplicaState],
    mask: u64,
    now: Instant,
    max_age: Duration,
) -> String {
    let mut parts = Vec::with_capacity(snap.len());
    for (i, s) in snap.iter().enumerate() {
        let tag = if !Reachability::bit(mask, i) {
            "inelig".to_string()
        } else {
            let stale = now.saturating_duration_since(s.updated_at) > max_age;
            match policy.score(s) {
                Some(sc) if stale => format!("stale({sc:.3})"),
                Some(sc) => format!("{sc:.3}"),
                None if stale => "stale".to_string(),
                None => "elig".to_string(),
            }
        };
        parts.push(format!("{}={}", s.name, tag));
    }
    parts.join(" ")
}

/// Stamp the router's run_id + request_id onto the response so client-side logs
/// can be joined to the router's per-request records by the SAME (run_id,
/// request_id) key. Present on EVERY path — success and failure — so
/// timed-out/failed requests are joinable too.
fn tag(mut resp: Response, run_id: &str, request_id: u64) -> Response {
    let h = resp.headers_mut();
    if let Ok(v) = HeaderValue::from_str(&request_id.to_string()) {
        h.insert(HeaderName::from_static("x-router-request-id"), v);
    }
    if let Ok(v) = HeaderValue::from_str(run_id) {
        h.insert(HeaderName::from_static("x-router-run-id"), v);
    }
    resp
}

pub async fn forward(
    State(app): State<Arc<AppState>>,
    headers: HeaderMap,
    uri: Uri,
    body: Body,
) -> Response {
    let arrival_i = Instant::now();
    let arrival_sys = SystemTime::now();
    let request_id = app.next_id.fetch_add(1, Ordering::Relaxed);

    let mut body = Some(body);

    // ===== Selection region (timed) =====
    // cost_predict_v3 also parses the request body here, because that is part of its overhead.
    let sel_start = Instant::now();
    let header_tokens = header_prompt_tokens(&headers);
    // state_aware_tracked needs a prompt size to decide: without the header it
    // pays the same body inspection as v3.
    let needs_body_estimate = matches!(
        app.policy,
        Policy::CostPredictV3 | Policy::ShapeAware | Policy::LengthAwareCongestion
    ) || (app.policy == Policy::StateAwareTracked && header_tokens.is_none());
    let mut v3_prepared = if needs_body_estimate {
        match prepare_v3_body(body.take().expect("request body is present")).await {
            Ok(prepared) => Some(prepared),
            Err(message) => {
                let select_ns = sel_start.elapsed().as_nanos() as u64;
                let mut record = RequestRecord::new(
                    app.writer.clone(),
                    app.run_id.clone(),
                    request_id,
                    app.policy.as_str(),
                    app.policy.reads_state(),
                    select_ns,
                    arrival_i,
                    arrival_sys,
                );
                record.set_staleness(
                    false,
                    "request_parse_error",
                    app.max_state_age.as_millis() as u64,
                );
                record.set_choice(None, None);
                record.set_status(StatusCode::BAD_REQUEST.as_u16());
                record.set_outcome(Outcome::UpstreamError);
                record.finalize();
                return tag(
                    (StatusCode::BAD_REQUEST, message).into_response(),
                    &app.run_id,
                    request_id,
                );
            }
        }
    } else {
        None
    };
    let (prompt_tokens, prompt_source) = match (header_tokens, &v3_prepared) {
        (Some(t), _) => (t, "header"),
        (None, Some(prepared)) => (prepared.estimate.estimated, prepared.estimate.method),
        (None, None) => (0, "none"),
    };
    let mut decision = if app.policy.is_tracked() {
        app.policy.select_tracked(&app.select_ctx, prompt_tokens)
    } else if let Some(prepared) = &v3_prepared {
        app.policy
            .select_predictive(&app.cache, &app.select_ctx, prepared.estimate.estimated)
    } else {
        app.policy.select(&app.cache, &app.select_ctx)
    };
    let select_ns = sel_start.elapsed().as_nanos() as u64;
    // ======================= END SELECTION REGION ===========================

    let mut record = RequestRecord::new(
        app.writer.clone(),
        app.run_id.clone(),
        request_id,
        app.policy.as_str(),
        app.policy.reads_state(),
        select_ns,
        arrival_i,
        arrival_sys,
    );
    record.set_reservation(decision.reservation.take());
    record.set_snapshot_consumed(decision.snapshot.is_some());
    // Router-tracked load: tracked policies created their entry inside
    // selection; every other policy is tracked here, outside the timed region,
    // so the tracked view is logged for all policies.
    let tracked_view = match (decision.tracking.take(), decision.choice) {
        (Some(guard), _) => {
            record.set_tracking(guard, prompt_tokens, prompt_source);
            decision.tracked_view.take()
        }
        (None, Some(i)) => {
            let (guard, view) = app.select_ctx.tracker().track(i, prompt_tokens);
            record.set_tracking(guard, prompt_tokens, prompt_source);
            Some(view)
        }
        (None, None) => decision.tracked_view.take(),
    };
    if let Some(prepared) = &v3_prepared {
        record.set_v3_request_log(V3RequestLog {
            request_tokens: prepared.estimate.estimated,
            request_tokens_true: prepared.estimate.true_count,
            request_tokens_exact: prepared.estimate.exact,
            request_token_method: prepared.estimate.method,
            request_body_read_ns: prepared.body_read_ns,
            request_token_estimate_ns: prepared.estimate_ns,
        });
    }

    // Log the state the decision saw. For state policies this is the exact
    // snapshot they consumed; for stateless policies we read one now — OUTSIDE
    // the timed region — purely for the record (it did not inform the choice).
    let snap = decision.snapshot.unwrap_or_else(|| app.cache.load_full());
    let now = Instant::now();
    let mask = app.reachability.load();
    let v2_candidate_logs = decision.v2_candidate_logs.as_deref();
    let v3_candidate_logs = decision.v3_candidate_logs.as_deref();
    let request_estimate = v3_prepared.as_ref().map(|prepared| prepared.estimate);
    let state_vector: Vec<StateVectorEntry> = snap
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let age = now.saturating_duration_since(s.updated_at);
            let v2 = v2_candidate_logs.and_then(|logs| logs.get(i));
            let v3 = v3_candidate_logs.and_then(|logs| logs.get(i));
            let tracked = tracked_view.as_ref().and_then(|view| view.get(i));
            StateVectorEntry {
                replica: s.name.clone(),
                u: s.u,
                b_metrics: s.b_metrics,
                b_slots_active: s.b_slots_active,
                requests_deferred: s.requests_deferred,
                local_pending_dispatches: v2.map(|v| v.local_pending_dispatches),
                b_eff: v2.map(|v| v.b_eff),
                weighted_load: v2.map(|v| v.weighted_load),
                stage1_exclusion_reason: v2.and_then(|v| v.stage1_exclusion_reason),
                v2_stage1_guard_fallback: v2.map(|v| v.stage1_guard_fallback),
                v2_shortlisted: v2.map(|v| v.shortlisted),
                request_tokens: v3.and(request_estimate.map(|estimate| estimate.estimated)),
                request_tokens_true: v3
                    .and_then(|_| request_estimate.and_then(|estimate| estimate.true_count)),
                request_tokens_exact: v3.and(request_estimate.map(|estimate| estimate.exact)),
                b_proj: v3.map(|v| v.b_proj),
                r_proj_unclamped: v3.map(|v| v.r_proj_unclamped),
                r_proj: v3.map(|v| v.r_proj),
                r_proj_clamped: v3.map(|v| v.r_proj_clamped),
                predicted_tpot: v3.map(|v| v.predicted_tpot),
                v3_chosen: v3.map(|v| v.chosen),
                tracked_requests: tracked.map(|t| t.requests),
                tracked_tokens: tracked.map(|t| t.tokens),
                eligible: Reachability::bit(mask, i),
                stale: age > app.max_state_age,
                degraded_metrics_only: s.degraded_metrics_only,
                age_ms: age.as_secs_f64() * 1e3,
            }
        })
        .collect();
    record.set_state_vector(state_vector);

    // Was the decision made on stale state (stale chosen replica, or all-stale fallback)?
    // Tracked policies do not read polled state, so this never applies to them.
    let chosen_stale = !app.policy.is_tracked()
        && decision
        .choice
        .map(|i| now.saturating_duration_since(snap[i].updated_at) > app.max_state_age)
        .unwrap_or(false);
    record.set_staleness(
        chosen_stale,
        decision.stale_mode.as_str(),
        app.max_state_age.as_millis() as u64,
    );

    // --verbose: stash a compact decision line (outcome appended at finalize).
    // Built here, outside the timed region; no effect on the JSONL record.
    if app.verbose {
        let chosen = decision
            .choice
            .map(|i| app.replicas[i].name.as_str())
            .unwrap_or("none");
        record.set_verbose_prefix(format!(
            "id={request_id} policy={} chosen={} scores=[{}] stale_mode={}",
            app.policy.as_str(),
            chosen,
            verbose_scores(app.policy, &snap, mask, now, app.max_state_age),
            decision.stale_mode.as_str(),
        ));
    }

    let idx = match decision.choice {
        Some(i) => i,
        None => {
            record.set_choice(None, None);
            record.set_status(StatusCode::SERVICE_UNAVAILABLE.as_u16());
            record.set_outcome(Outcome::NoHealthyReplica);
            record.finalize();
            return tag(
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    "no healthy replica available",
                )
                    .into_response(),
                &app.run_id,
                request_id,
            );
        }
    };
    let replica = &app.replicas[idx];
    record.set_choice(Some(idx), Some(replica.name.clone()));

    // Existing policies still forward opaque bytes and read them only here.
    // v3 already buffered the same bytes inside its timed inspection path.
    let body_bytes = match v3_prepared.take() {
        Some(prepared) => prepared.bytes,
        None => {
            match axum::body::to_bytes(body.take().expect("request body is present"), usize::MAX)
                .await
            {
                Ok(b) => b,
                Err(e) => {
                    record.set_status(StatusCode::BAD_REQUEST.as_u16());
                    record.set_outcome(Outcome::UpstreamError);
                    record.finalize();
                    return tag(
                        (
                            StatusCode::BAD_REQUEST,
                            format!("failed to read request body: {e}"),
                        )
                            .into_response(),
                        &app.run_id,
                        request_id,
                    );
                }
            }
        }
    };

    let url = format!("{}{}", replica.base_url, uri.path());
    record.mark_dispatch();

    let mut req = app
        .client
        .post(&url)
        .timeout(app.upstream_timeout) // per-request connect/TTFB deadline
        .body(body_bytes);
    if let Some(ct) = headers.get(CONTENT_TYPE) {
        req = req.header(CONTENT_TYPE, ct);
    }
    if let Some(acc) = headers.get(ACCEPT) {
        req = req.header(ACCEPT, acc);
    }

    let upstream = match req.send().await {
        Ok(r) => {
            // Connection established (regardless of HTTP status): reachable, so
            // reset the dispatch-failure streak. An HTTP 500 from a live engine
            // is NOT a node failure and must never evict.
            app.reachability.on_dispatch_success(idx);
            r
        }
        Err(e) => {
            // Terminal failure — no retry, no re-route (F9). Classify precisely:
            // ONLY a connection-level failure counts toward eviction.
            let (outcome, code) = if e.is_connect() {
                // connect refused / host unreachable / connect-timeout: node may
                // be down. Apply dispatch hysteresis; evict on the Nth in a row.
                if let Some(streak) = app.reachability.on_dispatch_failure(idx) {
                    eprintln!(
                        "[router] WARN EVICTED replica '{}' after {} consecutive connect failures",
                        replica.name, streak
                    );
                    app.writer.write_event(
                        &app.run_id,
                        app.policy.as_str(),
                        &replica.name,
                        "evicted",
                        streak,
                    );
                }
                (Outcome::DispatchConnectFailure, StatusCode::BAD_GATEWAY)
            } else if e.is_timeout() {
                // Connected, but no response headers within upstream_timeout
                // (e.g. a long prefill). Alive-but-slow -> NEVER evict.
                (Outcome::UpstreamTimeoutHeaders, StatusCode::GATEWAY_TIMEOUT)
            } else {
                // Other transport error: conservative, do not evict.
                (Outcome::UpstreamError, StatusCode::BAD_GATEWAY)
            };
            record.set_status(code.as_u16());
            record.set_outcome(outcome);
            record.finalize();
            return tag(
                (code, format!("upstream request to {url} failed: {e}")).into_response(),
                &app.run_id,
                request_id,
            );
        }
    };

    let status =
        StatusCode::from_u16(upstream.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let content_type = upstream.headers().get(CONTENT_TYPE).cloned();
    record.set_status(status.as_u16());

    // Stream back untouched, timing TTFT and finalizing the record on end/drop.
    let body_stream = instrumented_stream(upstream.bytes_stream(), record, app.stream_idle_timeout);
    let mut response = Response::builder()
        .status(status)
        .body(Body::from_stream(body_stream))
        .unwrap();
    if let Some(ct) = content_type {
        response.headers_mut().insert(CONTENT_TYPE, ct);
    }
    tag(response, &app.run_id, request_id)
}
