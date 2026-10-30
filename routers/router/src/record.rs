//! Per-request structured JSON-line record. One line per request, flushed
//! immediately so an overnight run survives a crash. A record is written EXACTLY
//! once — either explicitly on an early-return path (send failure, no healthy
//! replica) or when the instrumented response stream ends or is dropped.

use crate::reservation::ReservationGuard;
use crate::tracker::TrackGuard;
use serde::Serialize;
use serde_json::json;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    InProgress,
    Success,
    ClientDisconnect,
    /// Connection could not be established (refused / unreachable /
    /// connect-timeout). The ONLY outcome that counts toward eviction.
    DispatchConnectFailure,
    /// Connected fine, but no response headers within upstream_timeout (e.g. a
    /// long prefill). Alive-but-slow — never evicts.
    UpstreamTimeoutHeaders,
    UpstreamTimeoutStream,
    UpstreamError,
    NoHealthyReplica,
}

impl Outcome {
    fn as_str(&self) -> &'static str {
        match self {
            Outcome::InProgress => "in_progress",
            Outcome::Success => "success",
            Outcome::ClientDisconnect => "client_disconnect",
            Outcome::DispatchConnectFailure => "dispatch_connect_failure",
            Outcome::UpstreamTimeoutHeaders => "upstream_timeout_headers",
            Outcome::UpstreamTimeoutStream => "upstream_timeout_stream",
            Outcome::UpstreamError => "upstream_error",
            Outcome::NoHealthyReplica => "no_healthy_replica",
        }
    }
    fn is_terminal(&self) -> bool {
        !matches!(self, Outcome::InProgress)
    }
}

/// Sink for records: a file (append) or stdout. `Send` so Drop can write from a
/// runtime worker thread.
pub struct RecordWriter {
    out: Mutex<Box<dyn Write + Send>>,
}

impl RecordWriter {
    pub fn stdout() -> RecordWriter {
        RecordWriter {
            out: Mutex::new(Box::new(std::io::stdout())),
        }
    }

    pub fn file(path: &str) -> std::io::Result<RecordWriter> {
        let f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        Ok(RecordWriter {
            out: Mutex::new(Box::new(f)),
        })
    }

    fn write_line(&self, line: &str) {
        if let Ok(mut o) = self.out.lock() {
            let _ = writeln!(o, "{line}");
            let _ = o.flush();
        }
    }

    /// Emit a reachability (eviction/recovery) event as its own JSONL line so a
    /// mid-run eviction is greppable and the run excludable. `action` is
    /// "evicted" or "recovered"; `streak` is the consecutive dispatch-failure
    /// count at eviction (0 for recovery).
    pub fn write_event(
        &self,
        run_id: &str,
        policy: &str,
        replica: &str,
        action: &str,
        streak: u32,
    ) {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let v = json!({
            "type": "reachability_event",
            "run_id": run_id,
            "policy": policy,
            "ts_ns": ts,
            "replica": replica,
            "action": action,
            "consecutive_dispatch_failures": streak,
        });
        self.write_line(&v.to_string());
    }
}

#[derive(Serialize, Clone)]
pub struct StateVectorEntry {
    pub replica: String,
    pub u: f64,
    pub b_metrics: f64, // nominal concurrency (feeds policies)
    pub b_slots_active: u64,
    pub requests_deferred: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub local_pending_dispatches: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub b_eff: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weighted_load: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage1_exclusion_reason: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub v2_stage1_guard_fallback: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub v2_shortlisted: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_tokens_true: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_tokens_exact: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub b_proj: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r_proj_unclamped: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r_proj: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r_proj_clamped: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub predicted_tpot: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub v3_chosen: Option<bool>,
    /// Router-tracked load at decision time (before this request was added):
    /// dispatched-but-unfinished requests and their context tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tracked_requests: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tracked_tokens: Option<u64>,
    /// eligible for selection at decision time (from the reachability mask).
    pub eligible: bool,
    /// state age exceeded max_state_age_ms at decision time (excluded from argmin).
    pub stale: bool,
    pub degraded_metrics_only: bool,
    /// poll-to-decision freshness: how old this replica's cached state was when
    /// the decision was taken (SQ2 freshness signal).
    pub age_ms: f64,
}

#[derive(Clone, Debug)]
pub struct V3RequestLog {
    pub request_tokens: u64,
    pub request_tokens_true: Option<u64>,
    pub request_tokens_exact: bool,
    pub request_token_method: &'static str,
    pub request_body_read_ns: u64,
    pub request_token_estimate_ns: u64,
}

pub struct RequestRecord {
    writer: Arc<RecordWriter>,
    run_id: Arc<str>,
    request_id: u64,
    policy: &'static str,
    reads_state: bool,
    /// whether the timed selection actually loaded the u/B snapshot.
    state_snapshot_consumed: bool,
    select_ns: u64,
    chosen_index: Option<usize>,
    chosen_replica: Option<String>,
    http_status: Option<u16>,
    // timing anchors: Instant for accurate deltas, SystemTime for absolute ts.
    arrival_i: Instant,
    arrival_sys: SystemTime,
    dispatch_i: Option<Instant>,
    ttft_i: Option<Instant>,
    completion_i: Option<Instant>,
    outcome: Outcome,
    state_vector: Vec<StateVectorEntry>,
    /// the chosen replica's state was stale (age > max_state_age) at decision time.
    decision_on_stale_state: bool,
    /// how the decision handled staleness: live | some_stale_excluded |
    /// all_stale_compare | all_stale_rr | no_eligible | stateless.
    stale_mode: &'static str,
    max_state_age_ms: u64,
    /// `--verbose` only: pre-rendered decision detail, printed to STDERR at
    /// finalize with the final outcome appended. NEVER serialized to the JSONL.
    verbose_prefix: Option<String>,
    /// v2-only dispatch reservation. Terminal outcomes release it explicitly;
    /// Drop is the cancellation-safe fallback.
    reservation: Option<ReservationGuard>,
    /// v3-only request inspection metadata. Serialized as top-level fields only
    /// for v3, leaving the existing policies' JSON schema untouched.
    v3_request_log: Option<V3RequestLog>,
    /// Router-tracked load entry for this request. Released on the terminal
    /// outcome, like the v2 reservation; Drop is the fallback.
    tracking: Option<TrackGuard>,
    tracked_prompt_tokens: Option<u64>,
    tracked_prompt_source: Option<&'static str>,
    tracked_output_tokens: u64,
    /// Final usage block from the engine's stream, for validating the
    /// tracked counts after the run.
    usage: Option<(u64, u64)>,
    written: bool,
}

impl RequestRecord {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        writer: Arc<RecordWriter>,
        run_id: Arc<str>,
        request_id: u64,
        policy: &'static str,
        reads_state: bool,
        select_ns: u64,
        arrival_i: Instant,
        arrival_sys: SystemTime,
    ) -> RequestRecord {
        RequestRecord {
            writer,
            run_id,
            request_id,
            policy,
            reads_state,
            state_snapshot_consumed: false,
            select_ns,
            chosen_index: None,
            chosen_replica: None,
            http_status: None,
            arrival_i,
            arrival_sys,
            dispatch_i: None,
            ttft_i: None,
            completion_i: None,
            outcome: Outcome::InProgress,
            state_vector: Vec::new(),
            decision_on_stale_state: false,
            stale_mode: "stateless",
            max_state_age_ms: 0,
            verbose_prefix: None,
            reservation: None,
            v3_request_log: None,
            tracking: None,
            tracked_prompt_tokens: None,
            tracked_prompt_source: None,
            tracked_output_tokens: 0,
            usage: None,
            written: false,
        }
    }

    /// `--verbose` only. Stashes a pre-rendered decision line printed to STDERR
    /// at finalize; has NO effect on the serialized JSONL record.
    pub fn set_verbose_prefix(&mut self, prefix: String) {
        self.verbose_prefix = Some(prefix);
    }

    pub fn set_staleness(&mut self, on_stale: bool, mode: &'static str, max_state_age_ms: u64) {
        self.decision_on_stale_state = on_stale;
        self.stale_mode = mode;
        self.max_state_age_ms = max_state_age_ms;
    }

    pub fn set_snapshot_consumed(&mut self, v: bool) {
        self.state_snapshot_consumed = v;
    }
    pub fn set_state_vector(&mut self, sv: Vec<StateVectorEntry>) {
        self.state_vector = sv;
    }
    pub fn set_choice(&mut self, index: Option<usize>, replica: Option<String>) {
        self.chosen_index = index;
        self.chosen_replica = replica;
    }
    pub fn set_reservation(&mut self, reservation: Option<ReservationGuard>) {
        self.reservation = reservation;
    }
    pub fn set_v3_request_log(&mut self, log: V3RequestLog) {
        self.v3_request_log = Some(log);
    }
    pub fn set_tracking(&mut self, guard: TrackGuard, prompt_tokens: u64, source: &'static str) {
        self.tracking = Some(guard);
        self.tracked_prompt_tokens = Some(prompt_tokens);
        self.tracked_prompt_source = Some(source);
    }
    pub fn add_output_tokens(&mut self, n: u64) {
        if let Some(guard) = &mut self.tracking {
            guard.add_output_tokens(n);
            self.tracked_output_tokens = guard.output_tokens();
        }
    }
    pub fn set_usage(&mut self, prompt_tokens: u64, completion_tokens: u64) {
        self.usage = Some((prompt_tokens, completion_tokens));
    }
    pub fn set_status(&mut self, status: u16) {
        self.http_status = Some(status);
    }
    pub fn mark_dispatch(&mut self) {
        let now = Instant::now();
        self.dispatch_i = Some(now);
        if let Some(reservation) = &mut self.reservation {
            reservation.mark_dispatched(now);
        }
    }
    pub fn mark_ttft(&mut self) {
        if self.ttft_i.is_none() {
            self.ttft_i = Some(Instant::now());
        }
    }
    /// Set the terminal outcome and stamp completion (first terminal wins).
    pub fn set_outcome(&mut self, outcome: Outcome) {
        if self.outcome.is_terminal() {
            return;
        }
        self.outcome = outcome;
        if let Some(mut reservation) = self.reservation.take() {
            reservation.release();
        }
        if let Some(mut guard) = self.tracking.take() {
            self.tracked_output_tokens = guard.output_tokens();
            guard.release();
        }
        if self.completion_i.is_none() {
            self.completion_i = Some(Instant::now());
        }
    }

    fn to_unix_ns(&self, i: Instant) -> u128 {
        let base = self
            .arrival_sys
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        base + i.saturating_duration_since(self.arrival_i).as_nanos()
    }

    /// Serialize + write once. Idempotent.
    pub fn finalize(&mut self) {
        if self.written {
            return;
        }
        self.written = true;
        let completion = self.completion_i.unwrap_or_else(Instant::now);

        // ttft_ms is the ROUTER-INTERNAL time-to-first-token (dispatch -> first
        // content delta). It EXCLUDES the router<->client hop and is a diagnostic
        // cross-check only; client-side TTFT is authoritative for reported p50/p95.
        let ttft_ms = match (self.dispatch_i, self.ttft_i) {
            (Some(d), Some(t)) => Some(t.saturating_duration_since(d).as_secs_f64() * 1e3),
            _ => None,
        };
        let total_ms = completion
            .saturating_duration_since(self.arrival_i)
            .as_secs_f64()
            * 1e3;

        let mut rec = json!({
            "type": "request",
            // (run_id, request_id) is the globally-unique join key across restarts.
            "run_id": &*self.run_id,
            "request_id": self.request_id,
            "policy": self.policy,
            "outcome": self.outcome.as_str(),
            "http_status": self.http_status,
            "chosen_index": self.chosen_index,
            "chosen_replica": self.chosen_replica,
            // --- SQ2 overhead: selection only, one region incl. any snapshot load ---
            "select_ns": self.select_ns,
            "select_us": self.select_ns as f64 / 1000.0,
            "reads_state": self.reads_state,
            "state_snapshot_consumed": self.state_snapshot_consumed,
            // --- absolute timestamps (unix ns) ---
            "arrival_ts_ns": self.to_unix_ns(self.arrival_i),
            "dispatch_ts_ns": self.dispatch_i.map(|i| self.to_unix_ns(i)),
            "ttft_ts_ns": self.ttft_i.map(|i| self.to_unix_ns(i)),
            "completion_ts_ns": self.to_unix_ns(completion),
            // --- derived durations ---
            "ttft_ms_router_internal": ttft_ms,
            "total_ms": total_ms,
            // --- staleness of the decision (post-incident: rode stale state?) ---
            "decision_on_stale_state": self.decision_on_stale_state,
            "stale_mode": self.stale_mode,
            "max_state_age_ms": self.max_state_age_ms,
            // --- the state the decision was taken on ---
            "state_vector": self.state_vector,
        });
        if let Some(v3) = &self.v3_request_log {
            let object = rec
                .as_object_mut()
                .expect("request record root is always a JSON object");
            object.insert("request_tokens".into(), json!(v3.request_tokens));
            object.insert("request_tokens_true".into(), json!(v3.request_tokens_true));
            object.insert(
                "request_tokens_exact".into(),
                json!(v3.request_tokens_exact),
            );
            object.insert(
                "request_token_method".into(),
                json!(v3.request_token_method),
            );
            object.insert(
                "request_body_read_ns".into(),
                json!(v3.request_body_read_ns),
            );
            object.insert(
                "request_token_estimate_ns".into(),
                json!(v3.request_token_estimate_ns),
            );
        }
        if let Some(source) = self.tracked_prompt_source {
            let object = rec
                .as_object_mut()
                .expect("request record root is always a JSON object");
            object.insert("tracked_prompt_tokens".into(), json!(self.tracked_prompt_tokens));
            object.insert("tracked_prompt_source".into(), json!(source));
            object.insert("tracked_output_tokens".into(), json!(self.tracked_output_tokens));
            object.insert("usage_prompt_tokens".into(), json!(self.usage.map(|u| u.0)));
            object.insert("usage_completion_tokens".into(), json!(self.usage.map(|u| u.1)));
        }
        self.writer.write_line(&rec.to_string());

        // --verbose: STDERR only, AFTER the JSONL write and completely separate
        // from it. The outcome is appended here because it is only final now.
        if let Some(prefix) = &self.verbose_prefix {
            eprintln!("[req] {prefix} outcome={}", self.outcome.as_str());
        }
    }
}

impl Drop for RequestRecord {
    fn drop(&mut self) {
        // A record that reached Drop without a terminal outcome means the client
        // hung up mid-stream. Never let it vanish from the stats.
        if !self.written {
            if !self.outcome.is_terminal() {
                self.set_outcome(Outcome::ClientDisconnect);
            }
            self.finalize();
        }
    }
}
