//! Router entrypoint: parse CLI + TOML config, warm the state cache with one
//! poll cycle, spawn the background poller, and serve the shared forwarding path
//! with the selected policy. `--diagnose` runs the signal-freshness check and
//! exits instead of serving.

mod cache;
mod config;
mod diagnose;
mod parse;
mod policy;
mod poller;
mod proxy;
mod reachability;
mod record;
mod replica;
mod request_tokens;
mod reservation;
mod stream;
mod tracker;

#[cfg(test)]
mod smoke;

use crate::cache::StateCache;
use crate::config::{Config, KvMode, ReplicaConfig};
use crate::policy::{Policy, SelectCtx};
use crate::poller::PollerCfg;
use crate::proxy::AppState;
use crate::reachability::Reachability;
use crate::record::RecordWriter;
use crate::replica::{Replica, ReplicaState};
use clap::Parser;
use std::net::SocketAddr;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use std::time::Duration;

#[derive(Parser, Debug)]
#[command(about = "Inference-state-aware LLM request router (nine switchable policy variants)")]
struct Cli {
    /// Path to the TOML config.
    #[arg(long, default_value = "router.toml")]
    config: String,

    /// Override the policy from config.
    #[arg(long)]
    policy: Option<String>,

    /// Override the per-request JSON log path from config.
    #[arg(long)]
    log: Option<String>,

    /// Run the signal-freshness diagnostic against one replica, then exit.
    #[arg(long)]
    diagnose: bool,

    /// Which replica to diagnose (name). Defaults to the first configured.
    #[arg(long)]
    diagnose_replica: Option<String>,

    /// Diagnostic duration in seconds.
    #[arg(long, default_value_t = 5)]
    diagnose_secs: u64,

    /// Operator visibility only (STDERR, never the JSONL log). Print one
    /// heartbeat line per replica every N seconds — eligibility, u, b_metrics,
    /// resident_tokens, state age. Off by default. Confirms the poller is alive.
    #[arg(long)]
    heartbeat_secs: Option<u64>,

    /// Operator visibility only (STDERR, never the JSONL log). Print a per-request
    /// decision line: chosen replica, per-candidate scores, stale_mode, outcome.
    /// Off by default.
    #[arg(long, default_value_t = false)]
    verbose: bool,
}

#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        eprintln!("[router] fatal: {e}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), String> {
    let cli = Cli::parse();
    let cfg = Config::load(&cli.config)?;

    let policy_str = cli.policy.as_deref().unwrap_or(&cfg.policy);
    let policy = Policy::parse(policy_str)?;

    let replicas: Vec<Replica> = cfg.replicas.iter().map(Replica::from_config).collect();
    let poll_interval = Duration::from_millis(cfg.poll_interval_ms);
    let poll_timeout = Duration::from_millis(cfg.poll_timeout_ms);

    // Separate clients for polling vs. forwarding so their timeouts/pools don't
    // interfere. Per-request timeouts are set at call sites, not globally.
    let poll_client = reqwest::Client::builder()
        .build()
        .map_err(|e| e.to_string())?;

    // --- diagnose mode: check signal freshness against one loaded replica ---
    if cli.diagnose {
        let target = match &cli.diagnose_replica {
            Some(name) => replicas
                .iter()
                .find(|r| &r.name == name)
                .ok_or_else(|| format!("no replica named '{name}'"))?,
            None => &replicas[0],
        };
        let code = diagnose::run(
            target,
            &poll_client,
            poll_interval,
            poll_timeout,
            Duration::from_secs(cli.diagnose_secs),
        )
        .await;
        std::process::exit(code);
    }

    // --- fitted-cost policies require trustworthy per-node coefficients: fail
    //     loudly rather than run on placeholders ---
    if matches!(
        policy,
        Policy::CostModel | Policy::CostPredictV3 | Policy::ShapeAware | Policy::LengthAwareCongestion
    ) {
        require_cost_model_coeffs(&cfg.replicas, policy)?;
        warn_cost_model_plausibility(&cfg.replicas, policy);
    }

    // --- startup sanity: config context_budget vs. observed /slots total n_ctx ---
    eprintln!("[router] kv_mode={}", cfg.kv_mode.as_str());
    cross_check_context_budget(&replicas, &poll_client, poll_timeout, cfg.kv_mode).await;

    // --- reachability (eligibility) + warm the cache with one real poll ---
    let reach = Arc::new(Reachability::new(
        replicas.len(),
        cfg.dispatch_eviction_threshold,
    ));
    let placeholders: Vec<ReplicaState> = replicas.iter().map(ReplicaState::placeholder).collect();
    let cache = Arc::new(StateCache::new(placeholders));
    {
        let prev = cache.load_full();
        let warm = poller::poll_all(&replicas, &poll_client, poll_timeout, &prev).await;
        let mut states = Vec::with_capacity(warm.len());
        for (i, (state, metrics_ok, _poll_started_at)) in warm.into_iter().enumerate() {
            if metrics_ok {
                reach.on_poll_success(i); // admit reachable nodes before serving
            } else {
                eprintln!(
                    "[router] WARN replica '{}' /metrics unreachable at startup — ineligible until a poll succeeds",
                    state.name
                );
            }
            states.push(state);
        }
        cache.store(states);
    }

    // --- logging sink + this run's id (the poller needs both for events) ---
    let writer = match cli.log.as_deref().or(cfg.log_path.as_deref()) {
        Some(path) => {
            Arc::new(RecordWriter::file(path).map_err(|e| format!("open log {path}: {e}"))?)
        }
        None => Arc::new(RecordWriter::stdout()),
    };
    // One id per router run (<unix_nanos>-<pid>); with the request counter, (run_id, request_id) is unique.
    let run_id: Arc<str> = Arc::from(
        format!(
            "{}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos(),
            std::process::id()
        )
        .as_str(),
    );

    // v2's pending-dispatch book is owned by SelectCtx and shared with the
    // poller for post-publication expiry. Other policies never touch it.
    let weights: Vec<f64> = replicas.iter().map(|r| r.weight).collect();
    let max_state_age = Duration::from_millis(cfg.max_state_age_ms);
    let select_ctx = SelectCtx::new(
        weights,
        reach.clone(),
        0xC0FFEE,
        max_state_age,
        poll_interval,
    )
    .with_context_budgets(replicas.iter().map(|r| r.context_budget).collect());
    let pending_dispatches = select_ctx.pending_dispatches();

    // --- spawn the background poller (recovers eligibility; never evicts) ---
    {
        let replicas = Arc::new(replicas.clone());
        let cache = cache.clone();
        let reach = reach.clone();
        let client = poll_client.clone();
        let pcfg = PollerCfg {
            poll_interval,
            poll_timeout,
            writer: writer.clone(),
            run_id: run_id.clone(),
            policy: policy.as_str(),
            pending_dispatches,
        };
        tokio::spawn(async move {
            poller::run(replicas, cache, reach, client, pcfg).await;
        });
    }

    // --- optional heartbeat (STDERR only; never the JSONL log) ---
    if let Some(secs) = cli.heartbeat_secs.filter(|&s| s > 0) {
        let cache = cache.clone();
        let reach = reach.clone();
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(Duration::from_secs(secs));
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                ticker.tick().await;
                let snap = cache.load_full();
                let mask = reach.load();
                let now = std::time::Instant::now();
                for (i, s) in snap.iter().enumerate() {
                    eprintln!(
                        "[hb] {} elig={} u={:.3} b={:.1} resident={} age_ms={}",
                        s.name,
                        Reachability::bit(mask, i) as u8,
                        s.u,
                        s.b_metrics,
                        s.resident_tokens,
                        now.saturating_duration_since(s.updated_at).as_millis()
                    );
                }
            }
        });
    }

    // --- build the app state and serve ---
    let app_state = Arc::new(AppState {
        client: reqwest::Client::builder()
            .connect_timeout(Duration::from_millis(cfg.connect_timeout_ms))
            .build()
            .map_err(|e| e.to_string())?,
        run_id: run_id.clone(),
        replicas,
        cache,
        policy,
        select_ctx,
        reachability: reach,
        writer,
        next_id: AtomicU64::new(0),
        upstream_timeout: Duration::from_millis(cfg.upstream_timeout_ms),
        stream_idle_timeout: Duration::from_millis(cfg.stream_idle_timeout_ms),
        max_state_age,
        verbose: cli.verbose,
    });

    let app = axum::Router::new()
        .route("/v1/completions", axum::routing::post(proxy::forward))
        .route("/v1/chat/completions", axum::routing::post(proxy::forward))
        .with_state(app_state);

    let addr: SocketAddr = cfg
        .listen
        .parse()
        .map_err(|e| format!("bad listen addr '{}': {e}", cfg.listen))?;
    println!(
        "[router] listening on {addr}  run_id={run_id}  policy={}  poll_interval={}ms  upstream_timeout={}ms",
        policy.as_str(),
        cfg.poll_interval_ms,
        cfg.upstream_timeout_ms
    );
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| e.to_string())?;
    axum::serve(listener, app)
        .await
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// Fitted-cost policies must never run on placeholder/absent coefficients.
/// Both require c1 and c2 on every replica; v3 also requires c0. An all-zero
/// coefficient set is rejected because it would make that replica always win.
/// The original cost_model retains its deliberate c0=0 omission mode.
fn require_cost_model_coeffs(replicas: &[ReplicaConfig], policy: Policy) -> Result<(), String> {
    let policy_name = policy.as_str();
    let require_c0 = matches!(
        policy,
        Policy::CostPredictV3 | Policy::ShapeAware | Policy::LengthAwareCongestion
    );
    for r in replicas {
        if require_c0 && r.c0.is_none() {
            return Err(format!(
                "policy '{policy_name}' requires fitted c0, c1, and c2 on EVERY replica, but '{}' is missing c0",
                r.name
            ));
        }
        let (c1, c2) = match (r.c1, r.c2) {
            (Some(c1), Some(c2)) => (c1, c2),
            _ => {
                return Err(format!(
                    "policy '{policy_name}' requires fitted c1 AND c2 on EVERY replica, but '{}' is missing \
                     them. {policy_name} will NOT run with absent/placeholder coefficients — supply real \
                     per-node fitted values (c1 ms/request, c2 ms/token; optional c0 ms intercept) in the \
                     TOML before selecting this policy.",
                    r.name
                ))
            }
        };
        let c0 = r.c0.unwrap_or(0.0);
        if c0 == 0.0 && c1 == 0.0 && c2 == 0.0 {
            return Err(format!(
                "replica '{}' has all-zero {policy_name} coefficients (c0=c1=c2=0). That is a degenerate config, \
                 not a choice — a zero-cost node always wins argmin. Supply real fitted coefficients (this is \
                 the placeholder guard: {policy_name} refuses to run until '{}' has trustworthy c0/c1/c2).",
                r.name, r.name
            ));
        }
    }
    Ok(())
}

/// Warns (does not fail) about implausible cost_model coefficients, e.g. near-zero fits or a missing c0.
fn warn_cost_model_plausibility(replicas: &[ReplicaConfig], policy: Policy) {
    let policy_name = policy.as_str();
    // (a) c0 omitted -> selection biased toward the SLOWER node (its larger fixed
    //     cost is undercounted). Allowed, but must be deliberate.
    let missing_c0: Vec<&str> = replicas
        .iter()
        .filter(|r| r.c0.is_none())
        .map(|r| r.name.as_str())
        .collect();
    if !missing_c0.is_empty() {
        eprintln!(
            "[router] WARN {policy_name}: no c0 intercept on [{}] (treated as 0.0). Omitting c0 biases selection \
             toward the SLOWER node — its larger fixed cost is undercounted. Allowed (for the c0-in/c0-out \
             sensitivity check), but set c0 to remove the bias if this is not intentional.",
            missing_c0.join(", ")
        );
    }

    // (b) A coefficient >10x smaller than the largest across replicas: a near-zero
    //     fit scores near-zero and would win EVERY argmin silently. Warn, never block.
    oom_warn(replicas, policy_name, "c1", |r| r.c1);
    oom_warn(replicas, policy_name, "c2", |r| r.c2);
}

/// Warn if any replica's coefficient is more than an order of magnitude below
/// the largest across replicas. `get` selects the coefficient.
fn oom_warn(
    replicas: &[ReplicaConfig],
    policy_name: &str,
    label: &str,
    get: impl Fn(&ReplicaConfig) -> Option<f64>,
) {
    let vals: Vec<(&str, f64)> = replicas
        .iter()
        .filter_map(|r| get(r).map(|v| (r.name.as_str(), v)))
        .collect();
    if vals.len() < 2 {
        return;
    }
    let max = vals.iter().map(|(_, v)| *v).fold(f64::MIN, f64::max);
    if max <= 0.0 {
        return;
    }
    let small: Vec<String> = vals
        .iter()
        .filter(|(_, v)| *v * 10.0 < max)
        .map(|(n, v)| format!("{n}={v}"))
        .collect();
    if !small.is_empty() {
        eprintln!(
            "[router] WARN {policy_name}: {label} on [{}] is >10x smaller than the largest {label} (={max}) across \
             replicas. An unstable fit that produces near-zero coefficients scores near-zero and would win EVERY \
             argmin silently. NOT blocked (real hardware heterogeneity can differ this much) — verify the fit.",
            small.join(", ")
        );
    }
}

/// Warns if a node's `/slots` context disagrees with `context_budget`
/// (partitioned: sum of n_ctx; unified: no single slot may exceed the budget).
async fn cross_check_context_budget(
    replicas: &[Replica],
    client: &reqwest::Client,
    timeout: Duration,
    kv_mode: KvMode,
) {
    for r in replicas {
        let observed = client.get(&r.slots_url).timeout(timeout).send().await.ok();
        let Some(resp) = observed else { continue };
        let Ok(text) = resp.text().await else {
            continue;
        };
        if let Ok(p) = parse::parse_slots(&text) {
            if let Some(msg) =
                context_budget_warning(kv_mode, p.total_n_ctx, p.max_n_ctx, r.context_budget)
            {
                eprintln!("[router] WARN replica '{}': {msg}", r.name);
            }
        }
    }
}

/// Pure decision behind `cross_check_context_budget`; `None` means consistent.
fn context_budget_warning(
    kv_mode: KvMode,
    total_n_ctx: u64,
    max_n_ctx: u64,
    budget: f64,
) -> Option<String> {
    match kv_mode {
        KvMode::Partitioned => (total_n_ctx > 0 && (total_n_ctx as f64 - budget).abs() > 0.5)
            .then(|| {
                format!(
                    "kv_mode=partitioned: config context_budget={budget} but /slots reports \
                     Σn_ctx={total_n_ctx}. u = ΣR / c will use the CONFIG value; fix the TOML \
                     if the engine changed."
                )
            }),
        KvMode::Unified => (max_n_ctx as f64 > budget + 0.5).then(|| {
            format!(
                "kv_mode=unified: a slot reports n_ctx={max_n_ctx}, above the shared \
                 context_budget={budget}. u = ΣR / c will use the CONFIG value; check the \
                 engine's --ctx-size and --kv-unified flags."
            )
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partitioned_budget_check_uses_slot_sum() {
        // 4 x 1024 partitions == 4096 budget: consistent.
        assert!(context_budget_warning(KvMode::Partitioned, 4096, 1024, 4096.0).is_none());
        // Engine restarted with a different --ctx-size: flagged.
        assert!(context_budget_warning(KvMode::Partitioned, 2048, 512, 4096.0).is_some());
        // No n_ctx reported at all: nothing to compare, not flagged.
        assert!(context_budget_warning(KvMode::Partitioned, 0, 0, 4096.0).is_none());
    }

    #[test]
    fn unified_budget_check_ignores_overlapping_sum() {
        // Four slots each reporting the shared 2048 buffer sum to 8192; that is
        // overlap, not capacity, and must NOT be flagged.
        assert!(context_budget_warning(KvMode::Unified, 8192, 2048, 2048.0).is_none());
        // Slots reporting less than the shared budget are also consistent.
        assert!(context_budget_warning(KvMode::Unified, 2048, 512, 2048.0).is_none());
        // A single slot claiming more than the shared budget is a real mismatch.
        assert!(context_budget_warning(KvMode::Unified, 16384, 4096, 2048.0).is_some());
    }

    #[test]
    fn kv_mode_defaults_to_partitioned_and_rejects_unknown_values() {
        let base = r#"
            listen = "0.0.0.0:3000"
            policy = "state_aware"
            poll_interval_ms = 350
            poll_timeout_ms = 300
            upstream_timeout_ms = 120000
            [[replicas]]
            name = "n"
            url = "http://127.0.0.1:8080"
            weight = 1.0
            context_budget = 4096
        "#;
        let cfg: Config = toml::from_str(base).expect("default config parses");
        assert_eq!(cfg.kv_mode, KvMode::Partitioned);
        let unified: Config = toml::from_str(&format!("kv_mode = \"unified\"\n{base}"))
            .expect("unified parses");
        assert_eq!(unified.kv_mode, KvMode::Unified);
        assert!(toml::from_str::<Config>(&format!("kv_mode = \"paged\"\n{base}")).is_err());
    }

    fn replica(c0: Option<f64>, c1: Option<f64>, c2: Option<f64>) -> ReplicaConfig {
        ReplicaConfig {
            name: "test-node".into(),
            url: "http://127.0.0.1:8080".into(),
            weight: 1.0,
            context_budget: 4096,
            c0,
            c1,
            c2,
            c3: None,
        }
    }

    #[test]
    fn v3_startup_requires_all_three_coefficients() {
        for cfg in [
            replica(None, Some(47.0), Some(0.070)),
            replica(Some(53.0), None, Some(0.070)),
            replica(Some(53.0), Some(47.0), None),
        ] {
            assert!(require_cost_model_coeffs(&[cfg], Policy::CostPredictV3).is_err());
        }
    }

    #[test]
    fn v3_accepts_requested_pi5_r_only_coefficients() {
        let cfg = replica(Some(87.5638), Some(0.0), Some(0.013507));
        assert!(require_cost_model_coeffs(&[cfg], Policy::CostPredictV3).is_ok());
    }

    #[test]
    fn v3_rejects_degenerate_all_zero_replica() {
        let cfg = replica(Some(0.0), Some(0.0), Some(0.0));
        assert!(require_cost_model_coeffs(&[cfg], Policy::CostPredictV3).is_err());
    }

    #[test]
    fn original_cost_model_still_allows_omitted_intercept() {
        let cfg = replica(None, Some(47.0), Some(0.070));
        assert!(require_cost_model_coeffs(&[cfg], Policy::CostModel).is_ok());
    }
}
