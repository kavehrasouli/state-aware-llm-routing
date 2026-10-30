use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub listen: String,
    pub policy: String,
    #[serde(default = "default_poll_interval_ms")]
    pub poll_interval_ms: u64,
    #[serde(default = "default_upstream_timeout_ms")]
    pub upstream_timeout_ms: u64,
    #[serde(default = "default_stream_idle_timeout_ms")]
    pub stream_idle_timeout_ms: u64,
    #[serde(default = "default_poll_timeout_ms")]
    pub poll_timeout_ms: u64,
    /// Consecutive CONNECTION-LEVEL dispatch failures (connect refused /
    /// unreachable / connect-timeout, on the request path) before a replica is
    /// evicted from selection. A /metrics poll failure does NOT count — that
    /// only marks state stale. Recovery is on a successful /metrics poll.
    #[serde(default = "default_dispatch_eviction_threshold")]
    pub dispatch_eviction_threshold: u32,
    /// Cached state older than this counts as stale and is excluded from comparison (default 1000 ms,
    /// above the worst successful-poll age and below the prefill stall).
    #[serde(default = "default_max_state_age_ms")]
    pub max_state_age_ms: u64,
    /// Connect deadline, separate from `upstream_timeout_ms`, so a connect timeout (evictable) differs
    /// from a slow prefill (not evictable). Must be < upstream_timeout_ms.
    #[serde(default = "default_connect_timeout_ms")]
    pub connect_timeout_ms: u64,
    #[serde(default)]
    pub log_path: Option<String>,
    /// How the engine allocates its KV cache; only the startup context-budget check uses it.
    /// Partitioned: per-slot budgets sum to the total. Unified: slots share one buffer.
    #[serde(default)]
    pub kv_mode: KvMode,
    pub replicas: Vec<ReplicaConfig>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KvMode {
    #[default]
    Partitioned,
    Unified,
}

impl KvMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            KvMode::Partitioned => "partitioned",
            KvMode::Unified => "unified",
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ReplicaConfig {
    pub name: String,
    pub url: String,
    pub weight: f64,
    /// c — per-node context budget (token denominator of u).
    pub context_budget: u64,
    /// Fitted-cost coefficients (predicted-latency ms). Per-node fitted values;
    /// only `cost_model` and experimental `cost_predict_v3` read them. c1 =
    /// ms/in-flight-request, c2 = ms/resident-token, c0 = ms fixed intercept.
    /// cost_model requires c1/c2; v3 requires all three.
    #[serde(default)]
    pub c0: Option<f64>,
    #[serde(default)]
    pub c1: Option<f64>,
    #[serde(default)]
    pub c2: Option<f64>,
    /// Experimental shape coefficient (ms per token in the largest busy slot); only `shape_aware` uses it.
    #[serde(default)]
    pub c3: Option<f64>,
}

fn default_poll_interval_ms() -> u64 {
    100
}
fn default_upstream_timeout_ms() -> u64 {
    30_000
}
fn default_stream_idle_timeout_ms() -> u64 {
    30_000
}
fn default_poll_timeout_ms() -> u64 {
    80
}
fn default_dispatch_eviction_threshold() -> u32 {
    3
}
fn default_max_state_age_ms() -> u64 {
    1000
}
fn default_connect_timeout_ms() -> u64 {
    1000
}

impl Config {
    pub fn load(path: &str) -> Result<Config, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read config {path}: {e}"))?;
        let cfg: Config =
            toml::from_str(&text).map_err(|e| format!("failed to parse config {path}: {e}"))?;
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<(), String> {
        if self.replicas.is_empty() {
            return Err("config has no replicas".into());
        }
        if self.replicas.len() > 64 {
            return Err("health bitmask supports at most 64 replicas".into());
        }
        if self.poll_timeout_ms >= self.poll_interval_ms {
            return Err(format!(
                "poll_timeout_ms ({}) must be < poll_interval_ms ({}) so a dead node cannot stall the poller",
                self.poll_timeout_ms, self.poll_interval_ms
            ));
        }
        for r in &self.replicas {
            if r.weight <= 0.0 || !r.weight.is_finite() {
                return Err(format!(
                    "replica {} has non-positive or non-finite weight {}",
                    r.name, r.weight
                ));
            }
            if r.context_budget == 0 {
                return Err(format!("replica {} has zero context_budget", r.name));
            }
            // Sanity-check any coefficients that ARE present, regardless of the
            // selected policy. c0 may be any finite value (an intercept); c1/c2
            // are costs and must be finite and non-negative (a negative cost
            // would prefer the more-loaded node — almost certainly a mis-entry).
            for (label, v) in [("c0", r.c0), ("c1", r.c1), ("c2", r.c2)] {
                if let Some(x) = v {
                    if !x.is_finite() {
                        return Err(format!("replica {} {label} is not finite ({x})", r.name));
                    }
                }
            }
            for (label, v) in [("c1", r.c1), ("c2", r.c2)] {
                if let Some(x) = v {
                    if x < 0.0 {
                        return Err(format!(
                            "replica {} {label}={x} is negative; cost_model coefficients must be >= 0",
                            r.name
                        ));
                    }
                }
            }
        }
        if self.dispatch_eviction_threshold < 1 {
            return Err(
                "dispatch_eviction_threshold must be >= 1 (1 = evict on first connect failure)"
                    .into(),
            );
        }
        if self.max_state_age_ms == 0 {
            return Err("max_state_age_ms must be > 0".into());
        }
        if self.connect_timeout_ms == 0 || self.connect_timeout_ms >= self.upstream_timeout_ms {
            return Err(format!(
                "connect_timeout_ms ({}) must be > 0 and < upstream_timeout_ms ({}) so a slow prefill \
                 (connected, slow headers) is distinguishable from a connect failure",
                self.connect_timeout_ms, self.upstream_timeout_ms
            ));
        }
        Ok(())
    }
}
