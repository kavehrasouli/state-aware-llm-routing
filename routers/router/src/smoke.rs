//! Forwarding-path test without hardware: a mock llama-server streams scripted SSE events.
//! Checks byte-for-byte forwarding, TTFT at the first content token, and the written record.

use crate::cache::StateCache;
use crate::policy::{Policy, SelectCtx};
use crate::proxy::{forward, AppState};
use crate::reachability::Reachability;
use crate::record::RecordWriter;
use crate::replica::{Replica, ReplicaState};
use crate::reservation::PendingDispatches;

use axum::{
    body::Body,
    extract::State,
    http::{header::CONTENT_TYPE, HeaderMap, StatusCode, Uri},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use bytes::Bytes;
use futures::StreamExt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

/// Gap between the role-only delta and the first content delta. Large enough
/// that a TTFT stamped on the role delta (~0ms) is unmistakably distinguishable
/// from one correctly stamped on the content delta (~ROLE_TO_CONTENT_MS).
const ROLE_TO_CONTENT_MS: u64 = 200;

async fn mock_sse() -> Response {
    // Each yielded frame is flushed as a separate HTTP body chunk, so the
    // role delta reaches the router before the sleep, exactly like a real
    // streaming engine emitting a role token then decoding.
    let stream = async_stream::stream! {
        yield Ok::<_, std::convert::Infallible>(Bytes::from_static(
            b"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n",
        ));
        tokio::time::sleep(Duration::from_millis(ROLE_TO_CONTENT_MS)).await;
        yield Ok(Bytes::from_static(
            b"data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n",
        ));
        yield Ok(Bytes::from_static(
            b"data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n",
        ));
        yield Ok(Bytes::from_static(b"data: [DONE]\n\n"));
    };
    let mut resp = Response::new(Body::from_stream(stream));
    resp.headers_mut()
        .insert(CONTENT_TYPE, "text/event-stream".parse().unwrap());
    resp
}

/// A role delta, a content delta (so TTFT is captured), then a long stall with
/// no further bytes — models a replica that emits the first token then hangs.
async fn mock_sse_then_stall() -> Response {
    let stream = async_stream::stream! {
        yield Ok::<_, std::convert::Infallible>(Bytes::from_static(
            b"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n",
        ));
        yield Ok(Bytes::from_static(
            b"data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n",
        ));
        // Stall far longer than the router's stream_idle_timeout, sending nothing.
        tokio::time::sleep(Duration::from_secs(30)).await;
        yield Ok(Bytes::from_static(
            b"data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n",
        ));
    };
    let mut resp = Response::new(Body::from_stream(stream));
    resp.headers_mut()
        .insert(CONTENT_TYPE, "text/event-stream".parse().unwrap());
    resp
}

/// Sends response headers, then fails the body stream.
async fn mock_sse_body_error() -> Response {
    let stream = async_stream::stream! {
        yield Ok::<_, std::io::Error>(Bytes::from_static(
            b"data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n",
        ));
        yield Err(std::io::Error::other("scripted upstream body failure"));
    };
    let mut resp = Response::new(Body::from_stream(stream));
    resp.headers_mut()
        .insert(CONTENT_TYPE, "text/event-stream".parse().unwrap());
    resp
}

/// Connects, then never sends headers (a slow but live engine): the request times out but must not evict.
async fn mock_hang_before_headers() -> Response {
    tokio::time::sleep(Duration::from_secs(30)).await;
    Response::new(Body::empty())
}

/// A live engine returning HTTP 500 — must never count as a dispatch failure.
async fn mock_500() -> Response {
    (StatusCode::INTERNAL_SERVER_ERROR, "boom").into_response()
}

/// Minimal healthy `/metrics` (Prometheus) for the recovery test.
async fn mock_metrics() -> Response {
    (
        StatusCode::OK,
        "# HELP x\n# TYPE x gauge\nllamacpp:requests_processing 1\nllamacpp:requests_deferred 0\nllamacpp:kv_cache_usage_ratio 0.10\n",
    )
        .into_response()
}

/// Minimal healthy `/slots` (JSON) for the recovery test.
async fn mock_slots() -> Response {
    (
        StatusCode::OK,
        r#"[{"id":0,"n_ctx":4096,"is_processing":false,"n_prompt_tokens":0}]"#,
    )
        .into_response()
}

/// An ephemeral port with nothing listening -> connections are REFUSED
/// (`is_connect()` true), the exact connection-level failure that evicts.
async fn closed_upstream() -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener); // close it; the port is now refusing
    format!("http://{addr}")
}

/// Bind an ephemeral port and serve the given upstream router. The socket is
/// listening as soon as `bind` returns, so no readiness sleep is needed.
async fn spawn_upstream(app: axum::Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

fn unique_log_path() -> String {
    // Atomic counter + nanos: tests run in parallel and macOS SystemTime nanos
    // are too coarse to be unique on their own, so two tests could collide.
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir()
        .join(format!("router_smoke_{nanos}_{seq}.jsonl"))
        .to_str()
        .unwrap()
        .to_string()
}

fn request_parts() -> (HeaderMap, Uri, Body) {
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, "application/json".parse().unwrap());
    let uri: Uri = "/v1/chat/completions".parse().unwrap();
    let body =
        Body::from(r#"{"model":"x","stream":true,"messages":[{"role":"user","content":"hi"}]}"#);
    (headers, uri, body)
}

fn build_state(
    upstream: &str,
    log_path: &str,
    upstream_timeout: Duration,
    stream_idle_timeout: Duration,
) -> Arc<AppState> {
    build_state_policy(
        upstream,
        log_path,
        upstream_timeout,
        stream_idle_timeout,
        Policy::RoundRobin,
        3,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_state_policy(
    upstream: &str,
    log_path: &str,
    upstream_timeout: Duration,
    stream_idle_timeout: Duration,
    policy: Policy,
    evict_threshold: u32,
    verbose: bool,
) -> Arc<AppState> {
    let (c0, c1, c2, c3) = if policy == Policy::CostPredictV3 {
        (53.0, 47.0, 0.070, 0.0)
    } else if policy == Policy::ShapeAware {
        (53.0, 47.0, 0.070, -0.0607)
    } else {
        (0.0, 0.0, 0.0, 0.0)
    };
    let replica = Replica {
        name: "mock".into(),
        base_url: upstream.to_string(),
        slots_url: format!("{upstream}/slots"),
        metrics_url: format!("{upstream}/metrics"),
        weight: 1.0,
        context_budget: 4096.0,
        c0,
        c1,
        c2,
        c3,
    };

    // One replica in the cache so the logged state vector is populated.
    let mut s0 = ReplicaState::placeholder(&replica);
    s0.u = 0.25;
    s0.b_metrics = 1.0;
    s0.b_slots_active = 1;
    s0.n_slots = 4;
    s0.updated_at = Instant::now(); // fresh, so it is never treated as stale here
    let cache = Arc::new(StateCache::new(vec![s0]));

    let reach = Arc::new(Reachability::new(1, evict_threshold));
    reach.on_poll_success(0); // admit replica 0 (eligible for selection)

    let max_state_age = Duration::from_secs(5);
    Arc::new(AppState {
        client: reqwest::Client::builder()
            .connect_timeout(Duration::from_millis(500))
            .build()
            .unwrap(),
        run_id: Arc::from("test-run"),
        replicas: vec![replica],
        cache,
        policy,
        select_ctx: SelectCtx::new(
            vec![1.0],
            reach.clone(),
            1,
            max_state_age,
            Duration::from_millis(350),
        ),
        reachability: reach,
        writer: Arc::new(RecordWriter::file(log_path).unwrap()),
        next_id: AtomicU64::new(0),
        upstream_timeout,
        stream_idle_timeout,
        max_state_age,
        verbose,
    })
}

fn pending_count(app: &AppState) -> u64 {
    app.select_ctx.pending_dispatches().counts().iter().sum()
}

fn header(resp: &Response, name: &str) -> String {
    resp.headers()
        .get(name)
        .unwrap_or_else(|| panic!("response must carry {name}"))
        .to_str()
        .unwrap()
        .to_string()
}

fn contains_content(chunk: &[u8]) -> bool {
    let s = String::from_utf8_lossy(chunk);
    if let Some(pos) = s.find("\"content\":\"") {
        let after = &s[pos + "\"content\":\"".len()..];
        return after.chars().next().is_some_and(|c| c != '"');
    }
    false
}

fn read_last_record(path: &str) -> serde_json::Value {
    // The record is flushed when the router's stream reaches EOF; by the time we
    // finish consuming the response it should be present. Retry briefly to avoid
    // any scheduling race, then fail loudly.
    for _ in 0..40 {
        if let Ok(text) = std::fs::read_to_string(path) {
            if let Some(line) = text.lines().last() {
                if !line.trim().is_empty() {
                    return serde_json::from_str(line).expect("record is valid JSON");
                }
            }
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    panic!("no record was written to {path}");
}

fn read_lines(path: &str) -> Vec<serde_json::Value> {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("valid JSON line"))
        .collect()
}

/// Last request record (skipping any reachability_event lines).
fn last_request(path: &str) -> serde_json::Value {
    for _ in 0..40 {
        if let Some(r) = read_lines(path)
            .into_iter()
            .rfind(|v| v["type"] == "request")
        {
            return r;
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    panic!("no request record in {path}");
}

/// First reachability_event with the given action, if any (with brief retry).
fn find_event(path: &str, action: &str) -> serde_json::Value {
    for _ in 0..40 {
        if let Some(ev) = read_lines(path)
            .into_iter()
            .find(|v| v["type"] == "reachability_event" && v["action"] == action)
        {
            return ev;
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    panic!("no '{action}' reachability_event in {path}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sse_forwarding_ttft_and_record() {
    let upstream =
        spawn_upstream(axum::Router::new().route("/v1/chat/completions", post(mock_sse))).await;
    let log_path = unique_log_path();
    let app = build_state(
        &upstream,
        &log_path,
        Duration::from_secs(5),
        Duration::from_secs(5),
    );

    let (headers, uri, body) = request_parts();
    let t_start = Instant::now();
    let response = forward(State(app.clone()), headers, uri, body).await;
    assert_eq!(
        response.status(),
        200,
        "forwarding path should return upstream 200"
    );

    // The join key: the router stamps run_id + request_id on the response so
    // client logs can be joined to router records. Capture before consuming.
    let hdr_id = header(&response, "x-router-request-id");
    let hdr_run = header(&response, "x-router-run-id");

    // Consume the streamed response, measuring CLIENT-side TTFT (to first content
    // delta) and reassembling the full body. Consuming to EOF drives the router's
    // instrumented stream to completion, which finalizes the record.
    let mut stream = response.into_body().into_data_stream();
    let mut client_ttft: Option<Duration> = None;
    let mut assembled = Vec::new();
    let mut first_chunk_at: Option<Duration> = None;
    while let Some(item) = stream.next().await {
        let chunk = item.expect("chunk should not error");
        first_chunk_at.get_or_insert_with(|| t_start.elapsed());
        if client_ttft.is_none() && contains_content(&chunk) {
            client_ttft = Some(t_start.elapsed());
        }
        assembled.extend_from_slice(&chunk);
    }

    // 1. Bytes forwarded intact, unbuffered (all scripted frames present, in order).
    let text = String::from_utf8(assembled).unwrap();
    assert!(
        text.contains("\"role\":\"assistant\""),
        "role delta present"
    );
    assert!(
        text.contains("Hello") && text.contains(" world"),
        "content deltas present"
    );
    assert!(
        text.trim_end().ends_with("[DONE]"),
        "stream terminates with [DONE]"
    );

    // 2. First frame (role-only) arrived early; content only after the gap — proof
    //    the stream was not buffered into one blob.
    let first = first_chunk_at.expect("received at least one chunk");
    let client_ttft = client_ttft.expect("saw a content delta");
    assert!(
        first < Duration::from_millis(ROLE_TO_CONTENT_MS),
        "role frame should arrive before the content gap ({first:?})"
    );
    assert!(
        client_ttft >= Duration::from_millis(ROLE_TO_CONTENT_MS - 50),
        "client TTFT should reflect the content gap, got {client_ttft:?}"
    );

    // 3. The written record: correct choice/policy/outcome, and router-internal
    //    TTFT stamped at the CONTENT delta (not the earlier role delta).
    let rec = read_last_record(&log_path);
    assert_eq!(rec["outcome"], "success");
    assert_eq!(rec["chosen_replica"], "mock");
    assert_eq!(rec["policy"], "round_robin");
    assert_eq!(rec["http_status"], 200);
    assert_eq!(rec["reads_state"], false);
    assert_eq!(rec["state_snapshot_consumed"], false);
    assert!(
        rec.get("request_tokens").is_none(),
        "existing policies must not gain v3-only top-level fields"
    );
    assert!(
        rec["state_vector"][0].get("predicted_tpot").is_none(),
        "existing policies must not gain v3-only candidate fields"
    );

    let ttft = rec["ttft_ms_router_internal"]
        .as_f64()
        .expect("router TTFT recorded");
    let total = rec["total_ms"].as_f64().expect("total recorded");
    assert!(
        ttft >= (ROLE_TO_CONTENT_MS as f64) - 50.0,
        "router TTFT must be stamped at the content delta, not the role delta: {ttft}ms"
    );
    assert!(
        ttft <= total + 1.0,
        "TTFT ({ttft}) must not exceed total ({total})"
    );

    let sv = rec["state_vector"]
        .as_array()
        .expect("state vector present");
    assert_eq!(sv.len(), 1);
    assert_eq!(sv[0]["replica"], "mock");

    // The stamped headers equal the logged (run_id, request_id) — join key works.
    assert_eq!(
        hdr_id,
        rec["request_id"].to_string(),
        "header id must match record request_id"
    );
    assert_eq!(
        hdr_run,
        rec["run_id"].as_str().unwrap(),
        "header run_id must match record run_id"
    );

    let _ = std::fs::remove_file(&log_path);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn v3_inspects_inside_timer_and_logs_prediction_audit_data() {
    let upstream =
        spawn_upstream(axum::Router::new().route("/v1/chat/completions", post(mock_sse))).await;
    let log_path = unique_log_path();
    let app = build_state_policy(
        &upstream,
        &log_path,
        Duration::from_secs(5),
        Duration::from_secs(5),
        Policy::CostPredictV3,
        3,
        false,
    );

    let (headers, uri, body) = request_parts();
    let response = forward(State(app), headers, uri, body).await;
    assert_eq!(response.status(), StatusCode::OK);
    let mut stream = response.into_body().into_data_stream();
    while let Some(item) = stream.next().await {
        item.unwrap();
    }
    drop(stream);

    let rec = last_request(&log_path);
    assert_eq!(rec["policy"], "cost_predict_v3");
    assert_eq!(rec["reads_state"], true);
    assert_eq!(rec["state_snapshot_consumed"], true);
    // "hi" estimates to one content token, plus four message-template tokens
    // and three generation-template tokens.
    assert_eq!(rec["request_tokens"], 8);
    assert_eq!(rec["request_tokens_true"], serde_json::Value::Null);
    assert_eq!(rec["request_tokens_exact"], false);
    assert_eq!(rec["request_token_method"], "hybrid_word_char_chat_v1");
    let body_read_ns = rec["request_body_read_ns"].as_u64().unwrap();
    let estimate_ns = rec["request_token_estimate_ns"].as_u64().unwrap();
    let select_ns = rec["select_ns"].as_u64().unwrap();
    assert!(
        select_ns >= body_read_ns + estimate_ns,
        "v3 select_ns must include body buffering and token estimation"
    );

    let candidate = &rec["state_vector"][0];
    assert_eq!(candidate["request_tokens"], 8);
    assert_eq!(candidate["request_tokens_exact"], false);
    assert_eq!(candidate["b_proj"], 2.0);
    assert_eq!(candidate["r_proj_unclamped"], 8);
    assert_eq!(candidate["r_proj"], 8);
    assert_eq!(candidate["r_proj_clamped"], false);
    assert_eq!(candidate["v3_chosen"], true);
    assert!(
        (candidate["predicted_tpot"].as_f64().unwrap() - 147.56).abs() < 1e-9,
        "53 + 47*2 + .070*8"
    );

    let _ = std::fs::remove_file(&log_path);
}

/// Headers timeout (long prefill): a marked failure, but the replica must not be evicted.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn headers_timeout_is_failure_but_does_not_evict() {
    let upstream = spawn_upstream(
        axum::Router::new().route("/v1/chat/completions", post(mock_hang_before_headers)),
    )
    .await;
    let log_path = unique_log_path();
    // Short header deadline so the test resolves quickly; connect succeeds fast.
    let app = build_state(
        &upstream,
        &log_path,
        Duration::from_millis(300),
        Duration::from_secs(5),
    );

    let (headers, uri, body) = request_parts();
    let response = forward(State(app.clone()), headers, uri, body).await;
    assert_eq!(
        response.status(),
        504,
        "headers timeout must surface as 504"
    );

    // Failed requests must be joinable too — both id headers present on errors.
    let hdr_id = header(&response, "x-router-request-id");
    let hdr_run = header(&response, "x-router-run-id");

    let rec = read_last_record(&log_path);
    assert_eq!(
        hdr_id,
        rec["request_id"].to_string(),
        "header id must match record on failure path"
    );
    assert_eq!(
        hdr_run,
        rec["run_id"].as_str().unwrap(),
        "header run_id must match record on failure path"
    );
    assert_eq!(rec["outcome"], "upstream_timeout_headers");
    assert_eq!(rec["http_status"], 504);
    assert!(
        rec["ttft_ms_router_internal"].is_null(),
        "a request that never got a first byte must have null TTFT"
    );
    // The crux of the fix: a slow-but-alive node is NOT evicted.
    assert!(
        app.reachability.is_eligible(0),
        "headers timeout must NOT evict the replica"
    );
    assert_eq!(
        app.reachability.failures(0),
        0,
        "headers timeout is not a dispatch failure"
    );

    let _ = std::fs::remove_file(&log_path);
}

/// Connection refused (nothing listening) is a connection-level failure. After
/// `threshold` consecutive ones the replica is evicted; before that it is not.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn connect_refused_evicts_after_threshold() {
    let upstream = closed_upstream().await;
    let log_path = unique_log_path();
    let threshold: u32 = 3;
    let app = build_state_policy(
        &upstream,
        &log_path,
        Duration::from_secs(2),
        Duration::from_secs(2),
        Policy::RoundRobin,
        threshold,
        false,
    );
    assert!(
        app.reachability.is_eligible(0),
        "eligible before any failure"
    );

    for k in 1..=threshold {
        let (h, u, b) = request_parts();
        let response = forward(State(app.clone()), h, u, b).await;
        assert_eq!(response.status(), 502, "connect refused -> 502 (k={k})");
        let rec = last_request(&log_path);
        assert_eq!(rec["outcome"], "dispatch_connect_failure", "k={k}");
        if k < threshold {
            assert!(
                app.reachability.is_eligible(0),
                "still eligible before threshold (k={k})"
            );
        } else {
            assert!(
                !app.reachability.is_eligible(0),
                "evicted exactly at threshold"
            );
        }
    }

    let ev = find_event(&log_path, "evicted");
    assert_eq!(ev["replica"], "mock");
    assert_eq!(ev["consecutive_dispatch_failures"], threshold);

    let _ = std::fs::remove_file(&log_path);
}

/// An HTTP 500 from a LIVE engine reaches `send()` as Ok(resp) — it must not be
/// treated as a dispatch failure and must not evict.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn http_500_from_live_upstream_does_not_evict() {
    let upstream =
        spawn_upstream(axum::Router::new().route("/v1/chat/completions", post(mock_500))).await;
    let log_path = unique_log_path();
    let app = build_state(
        &upstream,
        &log_path,
        Duration::from_secs(2),
        Duration::from_secs(2),
    );

    let (headers, uri, body) = request_parts();
    let response = forward(State(app.clone()), headers, uri, body).await;
    assert_eq!(response.status(), 500);
    // Drain the body so the record finalizes.
    let mut stream = response.into_body().into_data_stream();
    while stream.next().await.is_some() {}
    drop(stream);

    let rec = last_request(&log_path);
    assert_eq!(rec["http_status"], 500);
    assert!(
        app.reachability.is_eligible(0),
        "HTTP 500 must not evict a live engine"
    );
    assert_eq!(
        app.reachability.failures(0),
        0,
        "HTTP 500 is a dispatch SUCCESS (connection established)"
    );

    let _ = std::fs::remove_file(&log_path);
}

/// A dispatch-evicted node receives no traffic, so it can only recover via the
/// poller: one successful /metrics poll re-admits it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn evicted_node_recovers_on_successful_poll() {
    let upstream = spawn_upstream(
        axum::Router::new()
            .route("/metrics", get(mock_metrics))
            .route("/slots", get(mock_slots)),
    )
    .await;
    let log_path = unique_log_path();
    let replica = Replica {
        name: "mock".into(),
        base_url: upstream.clone(),
        slots_url: format!("{upstream}/slots"),
        metrics_url: format!("{upstream}/metrics"),
        weight: 1.0,
        context_budget: 4096.0,
        c0: 0.0,
        c1: 0.0,
        c2: 0.0,
        c3: 0.0,
    };

    let reach = Reachability::new(1, 2);
    reach.on_poll_success(0); // admit
    reach.on_dispatch_failure(0);
    assert_eq!(reach.on_dispatch_failure(0), Some(2)); // evict
    assert!(!reach.is_eligible(0), "evicted precondition");

    let cache = StateCache::new(vec![ReplicaState::placeholder(&replica)]);
    let client = reqwest::Client::new();
    let writer = RecordWriter::file(&log_path).unwrap();

    crate::poller::poll_cycle(
        std::slice::from_ref(&replica),
        &cache,
        &reach,
        &client,
        Duration::from_millis(500),
        &writer,
        "test-run",
        "state_aware",
        &PendingDispatches::new(1),
    )
    .await;

    assert!(
        reach.is_eligible(0),
        "a successful /metrics poll must recover the evicted node"
    );
    let ev = find_event(&log_path, "recovered");
    assert_eq!(ev["replica"], "mock");

    let _ = std::fs::remove_file(&log_path);
}

/// Mid-stream idle timeout: a first token arrives, then the stream stalls; the request must count as a failure.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mid_stream_timeout_is_failure_not_success() {
    let upstream = spawn_upstream(
        axum::Router::new().route("/v1/chat/completions", post(mock_sse_then_stall)),
    )
    .await;
    let log_path = unique_log_path();
    // Send succeeds fast (headers arrive); the stall trips the idle timeout.
    let app = build_state(
        &upstream,
        &log_path,
        Duration::from_secs(5),
        Duration::from_millis(300),
    );

    let (headers, uri, body) = request_parts();
    let response = forward(State(app.clone()), headers, uri, body).await;
    assert_eq!(
        response.status(),
        200,
        "upstream returned 200 headers before stalling"
    );

    // Consume the stream: we should receive the content token, then a surfaced
    // error (NOT a clean EOF) when the idle timeout trips.
    let mut stream = response.into_body().into_data_stream();
    let mut got_content = false;
    let mut errored = false;
    while let Some(item) = stream.next().await {
        match item {
            Ok(chunk) => {
                if contains_content(&chunk) {
                    got_content = true;
                }
            }
            Err(_) => {
                errored = true;
                break;
            }
        }
    }
    // Dropping the stream drops the router's instrumented body -> finalizes the
    // record (outcome was already set before the error was yielded).
    drop(stream);

    assert!(
        got_content,
        "a first content token should have been received before the stall"
    );
    assert!(
        errored,
        "client must see an error, not a clean end-of-stream"
    );

    let rec = read_last_record(&log_path);

    // The crux: this is a FAILURE, not a success — despite a 200 status and a
    // valid TTFT. Any p50/p95 must filter on outcome=="success".
    assert_eq!(rec["outcome"], "upstream_timeout_stream");
    assert_ne!(rec["outcome"], "success");
    assert_eq!(
        rec["http_status"], 200,
        "status alone can't distinguish this from success — outcome must"
    );

    // Explicit TTFT decision: PRESENT (a real first token arrived), but the
    // request is flagged failed via outcome above. We keep the real measurement
    // rather than nulling it.
    let ttft = rec["ttft_ms_router_internal"].as_f64();
    assert!(
        ttft.is_some_and(|t| t >= 0.0),
        "mid-stream timeout must retain its real TTFT (got {ttft:?}), not discard it"
    );

    let _ = std::fs::remove_file(&log_path);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn v2_reservation_releases_on_success_and_logs_candidate_values() {
    let upstream =
        spawn_upstream(axum::Router::new().route("/v1/chat/completions", post(mock_sse))).await;
    let log_path = unique_log_path();
    let app = build_state_policy(
        &upstream,
        &log_path,
        Duration::from_secs(5),
        Duration::from_secs(5),
        Policy::StateAwareV2,
        3,
        false,
    );

    let (headers, uri, body) = request_parts();
    let response = forward(State(app.clone()), headers, uri, body).await;
    assert_eq!(
        pending_count(&app),
        1,
        "reservation remains until poll publication or request end"
    );
    let mut stream = response.into_body().into_data_stream();
    while let Some(item) = stream.next().await {
        item.unwrap();
    }
    drop(stream);
    assert_eq!(pending_count(&app), 0);

    let rec = last_request(&log_path);
    assert_eq!(rec["policy"], "state_aware_v2");
    assert_eq!(rec["reads_state"], true);
    let candidate = &rec["state_vector"][0];
    assert_eq!(candidate["local_pending_dispatches"], 0);
    assert_eq!(candidate["b_eff"], 1.0);
    assert_eq!(candidate["weighted_load"], 1.0);
    assert_eq!(candidate["v2_shortlisted"], true);
    assert!(candidate["stage1_exclusion_reason"].is_null());

    let _ = std::fs::remove_file(&log_path);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn v2_reservation_releases_on_request_body_failure() {
    let upstream =
        spawn_upstream(axum::Router::new().route("/v1/chat/completions", post(mock_sse))).await;
    let log_path = unique_log_path();
    let app = build_state_policy(
        &upstream,
        &log_path,
        Duration::from_secs(5),
        Duration::from_secs(5),
        Policy::StateAwareV2,
        3,
        false,
    );
    let (headers, uri, _) = request_parts();
    let body = Body::from_stream(futures::stream::once(async {
        Err::<Bytes, std::io::Error>(std::io::Error::other("scripted request body failure"))
    }));

    let response = forward(State(app.clone()), headers, uri, body).await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(pending_count(&app), 0);
    assert_eq!(last_request(&log_path)["outcome"], "upstream_error");

    let _ = std::fs::remove_file(&log_path);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn v2_reservation_releases_on_connect_failure_and_header_timeout() {
    let closed = closed_upstream().await;
    let connect_log = unique_log_path();
    let connect_app = build_state_policy(
        &closed,
        &connect_log,
        Duration::from_secs(2),
        Duration::from_secs(2),
        Policy::StateAwareV2,
        3,
        false,
    );
    let (headers, uri, body) = request_parts();
    let response = forward(State(connect_app.clone()), headers, uri, body).await;
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    assert_eq!(pending_count(&connect_app), 0);

    let hanging = spawn_upstream(
        axum::Router::new().route("/v1/chat/completions", post(mock_hang_before_headers)),
    )
    .await;
    let timeout_log = unique_log_path();
    let timeout_app = build_state_policy(
        &hanging,
        &timeout_log,
        Duration::from_millis(150),
        Duration::from_secs(2),
        Policy::StateAwareV2,
        3,
        false,
    );
    let (headers, uri, body) = request_parts();
    let response = forward(State(timeout_app.clone()), headers, uri, body).await;
    assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
    assert_eq!(pending_count(&timeout_app), 0);

    let _ = std::fs::remove_file(&connect_log);
    let _ = std::fs::remove_file(&timeout_log);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn v2_reservation_releases_on_upstream_body_failure_and_stream_timeout() {
    let failing = spawn_upstream(
        axum::Router::new().route("/v1/chat/completions", post(mock_sse_body_error)),
    )
    .await;
    let failure_log = unique_log_path();
    let failure_app = build_state_policy(
        &failing,
        &failure_log,
        Duration::from_secs(2),
        Duration::from_secs(2),
        Policy::StateAwareV2,
        3,
        false,
    );
    let (headers, uri, body) = request_parts();
    let response = forward(State(failure_app.clone()), headers, uri, body).await;
    let mut stream = response.into_body().into_data_stream();
    while stream.next().await.is_some() {}
    drop(stream);
    assert_eq!(pending_count(&failure_app), 0);
    assert_eq!(last_request(&failure_log)["outcome"], "upstream_error");

    let stalling = spawn_upstream(
        axum::Router::new().route("/v1/chat/completions", post(mock_sse_then_stall)),
    )
    .await;
    let timeout_log = unique_log_path();
    let timeout_app = build_state_policy(
        &stalling,
        &timeout_log,
        Duration::from_secs(2),
        Duration::from_millis(100),
        Policy::StateAwareV2,
        3,
        false,
    );
    let (headers, uri, body) = request_parts();
    let response = forward(State(timeout_app.clone()), headers, uri, body).await;
    let mut stream = response.into_body().into_data_stream();
    while stream.next().await.is_some() {}
    drop(stream);
    assert_eq!(pending_count(&timeout_app), 0);
    assert_eq!(
        last_request(&timeout_log)["outcome"],
        "upstream_timeout_stream"
    );

    let _ = std::fs::remove_file(&failure_log);
    let _ = std::fs::remove_file(&timeout_log);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn v2_reservation_releases_on_client_disconnect() {
    let upstream = spawn_upstream(
        axum::Router::new().route("/v1/chat/completions", post(mock_sse_then_stall)),
    )
    .await;
    let log_path = unique_log_path();
    let app = build_state_policy(
        &upstream,
        &log_path,
        Duration::from_secs(2),
        Duration::from_secs(2),
        Policy::StateAwareV2,
        3,
        false,
    );

    let (headers, uri, body) = request_parts();
    let response = forward(State(app.clone()), headers, uri, body).await;
    assert_eq!(pending_count(&app), 1);
    drop(response);
    tokio::task::yield_now().await;
    assert_eq!(pending_count(&app), 0);
    assert_eq!(last_request(&log_path)["outcome"], "client_disconnect");

    let _ = std::fs::remove_file(&log_path);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn v2_reservation_releases_when_forward_future_is_cancelled() {
    let upstream = spawn_upstream(
        axum::Router::new().route("/v1/chat/completions", post(mock_hang_before_headers)),
    )
    .await;
    let log_path = unique_log_path();
    let app = build_state_policy(
        &upstream,
        &log_path,
        Duration::from_secs(20),
        Duration::from_secs(2),
        Policy::StateAwareV2,
        3,
        false,
    );

    let (headers, uri, body) = request_parts();
    let task = tokio::spawn(forward(State(app.clone()), headers, uri, body));
    for _ in 0..100 {
        if pending_count(&app) == 1 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert_eq!(
        pending_count(&app),
        1,
        "request must reach selection before cancellation"
    );
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    tokio::task::yield_now().await;
    assert_eq!(pending_count(&app), 0);
    assert_eq!(last_request(&log_path)["outcome"], "client_disconnect");

    let _ = std::fs::remove_file(&log_path);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn successful_post_dispatch_poll_expires_v2_reservation() {
    let upstream = spawn_upstream(
        axum::Router::new()
            .route("/metrics", get(mock_metrics))
            .route("/slots", get(mock_slots)),
    )
    .await;
    let log_path = unique_log_path();
    let replica = Replica {
        name: "mock".into(),
        base_url: upstream.clone(),
        slots_url: format!("{upstream}/slots"),
        metrics_url: format!("{upstream}/metrics"),
        weight: 1.0,
        context_budget: 4096.0,
        c0: 0.0,
        c1: 0.0,
        c2: 0.0,
        c3: 0.0,
    };
    let reach = Reachability::new(1, 3);
    reach.on_poll_success(0);
    let cache = StateCache::new(vec![ReplicaState::placeholder(&replica)]);
    let writer = RecordWriter::file(&log_path).unwrap();
    let pending = Arc::new(PendingDispatches::new(1));
    let mut reservation = pending.select_and_reserve(|_| (Some(0), ())).0.unwrap();
    reservation.mark_dispatched(Instant::now());
    tokio::time::sleep(Duration::from_millis(1)).await;

    crate::poller::poll_cycle(
        std::slice::from_ref(&replica),
        &cache,
        &reach,
        &reqwest::Client::new(),
        Duration::from_millis(500),
        &writer,
        "test-run",
        "state_aware_v2",
        &pending,
    )
    .await;

    assert_eq!(pending.counts(), vec![0]);
    assert_eq!(pending.active_len(), 0);
    reservation.release();
    assert_eq!(
        pending.counts(),
        vec![0],
        "later RAII cleanup must not double-decrement"
    );

    let _ = std::fs::remove_file(&log_path);
}

/// Operator visibility must be STDERR-only: `--verbose` must not change the
/// JSONL record's schema and must not leak any diagnostic line into the log.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn verbose_does_not_alter_the_jsonl_log() {
    async fn run(verbose: bool) -> (Vec<String>, String) {
        let upstream =
            spawn_upstream(axum::Router::new().route("/v1/chat/completions", post(mock_sse))).await;
        let log_path = unique_log_path();
        let app = build_state_policy(
            &upstream,
            &log_path,
            Duration::from_secs(5),
            Duration::from_secs(5),
            Policy::StateAware, // a state policy so verbose has real scores to print
            3,
            verbose,
        );
        let (headers, uri, body) = request_parts();
        let response = forward(State(app), headers, uri, body).await;
        let mut stream = response.into_body().into_data_stream();
        while stream.next().await.is_some() {}
        drop(stream);

        let rec = last_request(&log_path);
        let mut keys: Vec<String> = rec.as_object().unwrap().keys().cloned().collect();
        keys.sort();
        let text = std::fs::read_to_string(&log_path).unwrap();
        let _ = std::fs::remove_file(&log_path);
        (keys, text)
    }

    let (keys_off, _) = run(false).await;
    let (keys_on, text_on) = run(true).await;

    // Same schema: verbose adds/removes no JSONL fields.
    assert_eq!(
        keys_off, keys_on,
        "verbose must not change the JSONL record schema"
    );

    // No diagnostic output leaked into the log file: every line is a JSON object
    // and none carries a verbose/heartbeat marker.
    for line in text_on.lines().filter(|l| !l.trim().is_empty()) {
        assert!(
            line.trim_start().starts_with('{'),
            "log line must be JSON, got: {line}"
        );
    }
    assert!(
        !text_on.contains("[req]") && !text_on.contains("[hb]") && !text_on.contains("scores="),
        "verbose/heartbeat output must never appear in the JSONL log"
    );
}

/// llama.cpp-style /v1/completions stream: three content events, an empty
/// one, then the usage block.
async fn mock_completions_usage() -> Response {
    let stream = async_stream::stream! {
        for t in ["one", " two", " three", ""] {
            yield Ok::<_, std::convert::Infallible>(Bytes::from(format!(
                "data: {{\"choices\":[{{\"text\":\"{t}\"}}]}}\n\n"
            )));
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        yield Ok(Bytes::from_static(
            b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":640,\"completion_tokens\":3}}\n\n",
        ));
        yield Ok(Bytes::from_static(b"data: [DONE]\n\n"));
    };
    let mut resp = Response::new(Body::from_stream(stream));
    resp.headers_mut()
        .insert(CONTENT_TYPE, "text/event-stream".parse().unwrap());
    resp
}

fn tracked_tokens(app: &AppState) -> (u64, u64) {
    let loads = app.select_ctx.tracker().loads();
    (
        loads.iter().map(|l| l.requests).sum(),
        loads.iter().map(|l| l.tokens).sum(),
    )
}

fn completion_parts(prompt_tokens: Option<u64>) -> (HeaderMap, Uri, Body) {
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, "application/json".parse().unwrap());
    if let Some(t) = prompt_tokens {
        headers.insert("x-prompt-tokens", t.to_string().parse().unwrap());
    }
    let uri: Uri = "/v1/completions".parse().unwrap();
    let body = Body::from(r#"{"model":"x","stream":true,"prompt":"a b c d"}"#);
    (headers, uri, body)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tracked_policy_counts_grow_and_release_and_log() {
    let upstream = spawn_upstream(
        axum::Router::new().route("/v1/completions", post(mock_completions_usage)),
    )
    .await;
    let log_path = unique_log_path();
    let app = build_state_policy(
        &upstream,
        &log_path,
        Duration::from_secs(5),
        Duration::from_secs(5),
        Policy::StateAwareTracked,
        3,
        false,
    );

    let (headers, uri, body) = completion_parts(Some(640));
    let response = forward(State(app.clone()), headers, uri, body).await;
    assert_eq!(response.status(), 200);
    assert_eq!(tracked_tokens(&app), (1, 640), "prompt tracked at dispatch");

    let mut stream = response.into_body().into_data_stream();
    let mut seen = 0;
    while let Some(item) = stream.next().await {
        let chunk = item.unwrap();
        if String::from_utf8_lossy(&chunk).contains("\"text\":\"one\"") {
            seen += 1;
            assert_eq!(tracked_tokens(&app), (1, 641), "output token tracked as streamed");
        }
    }
    drop(stream);
    assert_eq!(seen, 1);
    assert_eq!(tracked_tokens(&app), (0, 0), "released at stream end");

    let rec = last_request(&log_path);
    assert_eq!(rec["policy"], "state_aware_tracked");
    assert_eq!(rec["outcome"], "success");
    assert_eq!(rec["stale_mode"], "tracked");
    assert_eq!(rec["decision_on_stale_state"], false);
    assert_eq!(rec["state_snapshot_consumed"], false);
    assert_eq!(rec["tracked_prompt_tokens"], 640);
    assert_eq!(rec["tracked_prompt_source"], "header");
    assert_eq!(rec["tracked_output_tokens"], 3);
    assert_eq!(rec["usage_prompt_tokens"], 640);
    assert_eq!(rec["usage_completion_tokens"], 3);
    assert_eq!(rec["state_vector"][0]["tracked_requests"], 0);
    assert_eq!(rec["state_vector"][0]["tracked_tokens"], 0);
    // polled state is still logged next to the tracked view
    assert_eq!(rec["state_vector"][0]["u"], 0.25);

    let _ = std::fs::remove_file(&log_path);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tracked_entry_released_on_mid_stream_timeout() {
    let upstream = spawn_upstream(
        axum::Router::new().route("/v1/chat/completions", post(mock_sse_then_stall)),
    )
    .await;
    let log_path = unique_log_path();
    let app = build_state_policy(
        &upstream,
        &log_path,
        Duration::from_secs(5),
        Duration::from_millis(300),
        Policy::WlcTracked,
        3,
        false,
    );
    let (mut headers, uri, body) = request_parts();
    headers.insert("x-prompt-tokens", "12".parse().unwrap());
    let response = forward(State(app.clone()), headers, uri, body).await;
    let mut stream = response.into_body().into_data_stream();
    while let Some(item) = stream.next().await {
        if item.is_err() {
            break;
        }
    }
    drop(stream);
    assert_eq!(tracked_tokens(&app), (0, 0));
    let rec = last_request(&log_path);
    assert_eq!(rec["outcome"], "upstream_timeout_stream");
    assert_eq!(rec["tracked_output_tokens"], 1);
    assert!(rec["usage_completion_tokens"].is_null());

    let _ = std::fs::remove_file(&log_path);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tracked_entry_released_on_connect_failure() {
    let upstream = closed_upstream().await;
    let log_path = unique_log_path();
    let app = build_state_policy(
        &upstream,
        &log_path,
        Duration::from_secs(5),
        Duration::from_secs(5),
        Policy::StateAwareTracked,
        3,
        false,
    );
    let (headers, uri, body) = completion_parts(Some(100));
    let _ = forward(State(app.clone()), headers, uri, body).await;
    assert_eq!(tracked_tokens(&app), (0, 0));
    let rec = last_request(&log_path);
    assert_eq!(rec["outcome"], "dispatch_connect_failure");

    let _ = std::fs::remove_file(&log_path);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn polled_policy_also_logs_tracked_view_and_falls_back_without_header() {
    let upstream = spawn_upstream(
        axum::Router::new().route("/v1/completions", post(mock_completions_usage)),
    )
    .await;
    let log_path = unique_log_path();
    let app = build_state_policy(
        &upstream,
        &log_path,
        Duration::from_secs(5),
        Duration::from_secs(5),
        Policy::WeightedLeastConnections,
        3,
        false,
    );
    let (headers, uri, body) = completion_parts(None);
    let response = forward(State(app.clone()), headers, uri, body).await;
    assert_eq!(tracked_tokens(&app), (1, 0));
    let mut stream = response.into_body().into_data_stream();
    while let Some(item) = stream.next().await {
        item.unwrap();
    }
    drop(stream);
    assert_eq!(tracked_tokens(&app), (0, 0));
    let rec = last_request(&log_path);
    assert_eq!(rec["policy"], "weighted_least_connections");
    assert_eq!(rec["stale_mode"], "live");
    assert_eq!(rec["tracked_prompt_source"], "none");
    assert_eq!(rec["tracked_output_tokens"], 3);
    assert_eq!(rec["state_vector"][0]["tracked_requests"], 0);

    // state_aware_tracked without the header falls back to the body estimate.
    let app = build_state_policy(
        &upstream,
        &log_path,
        Duration::from_secs(5),
        Duration::from_secs(5),
        Policy::StateAwareTracked,
        3,
        false,
    );
    let (headers, uri, body) = completion_parts(None);
    let response = forward(State(app.clone()), headers, uri, body).await;
    assert_eq!(tracked_tokens(&app), (1, 4));
    drop(response);
    assert_eq!(tracked_tokens(&app), (0, 0), "dropped response releases");
    let rec = last_request(&log_path);
    assert_eq!(rec["tracked_prompt_source"], "hybrid_word_char_v1");

    let _ = std::fs::remove_file(&log_path);
}
