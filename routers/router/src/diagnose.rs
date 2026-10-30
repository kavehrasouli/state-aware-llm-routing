//! `--diagnose`: polls one loaded replica quickly and reports (PASS/FAIL) whether `/slots`
//! refreshes at least as often as the configured poll interval.

use crate::parse::parse_slots;
use crate::replica::Replica;
use reqwest::Client;
use std::time::{Duration, Instant};

pub async fn run(
    replica: &Replica,
    client: &Client,
    poll_interval: Duration,
    poll_timeout: Duration,
    duration: Duration,
) -> i32 {
    // Sample much faster than the poll interval so we can actually observe how
    // often the underlying value changes.
    let sample_every = Duration::from_millis(((poll_interval.as_millis() as u64) / 20).max(2));
    println!(
        "[diagnose] replica '{}' {}  poll_interval={}ms  sampling every {}ms for {}s",
        replica.name,
        replica.slots_url,
        poll_interval.as_millis(),
        sample_every.as_millis(),
        duration.as_secs()
    );

    let deadline = Instant::now() + duration;
    let mut last_sig: Option<String> = None;
    let mut last_change = Instant::now();
    let mut change_intervals_ms: Vec<f64> = Vec::new();
    let mut fetch_latencies_ms: Vec<f64> = Vec::new();
    let mut samples = 0u64;
    let mut ok = 0u64;
    let mut ever_busy = false;

    while Instant::now() < deadline {
        let t0 = Instant::now();
        let fetched = fetch(client, &replica.slots_url, poll_timeout).await;
        let lat = t0.elapsed().as_secs_f64() * 1e3;
        samples += 1;

        if let Ok(text) = fetched {
            ok += 1;
            fetch_latencies_ms.push(lat);
            if let Ok(p) = parse_slots(&text) {
                if p.busy_slots > 0 {
                    ever_busy = true;
                }
                // Signature = the things that move under load. A change means the
                // engine published fresh state.
                let sig = format!("{}:{}", p.resident_tokens, p.busy_slots);
                match &last_sig {
                    Some(prev) if *prev != sig => {
                        let now = Instant::now();
                        change_intervals_ms
                            .push(now.duration_since(last_change).as_secs_f64() * 1e3);
                        last_change = now;
                    }
                    None => last_change = Instant::now(),
                    _ => {}
                }
                last_sig = Some(sig);
            }
        }
        tokio::time::sleep(sample_every).await;
    }

    // ---- report ----
    println!(
        "[diagnose] samples={samples} ok={ok} observed_changes={}",
        change_intervals_ms.len()
    );
    if !fetch_latencies_ms.is_empty() {
        let mut f = fetch_latencies_ms.clone();
        f.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "[diagnose] /slots fetch latency ms: p50={:.1} p95={:.1} max={:.1}",
            pct(&f, 50.0),
            pct(&f, 95.0),
            f[f.len() - 1]
        );
    }

    let poll_ms = poll_interval.as_millis() as f64;

    if ok == 0 {
        println!("[diagnose] FAIL: no successful /slots fetch — endpoint unreachable or disabled.");
        return 1;
    }
    if !ever_busy || change_intervals_ms.is_empty() {
        println!(
            "[diagnose] INCONCLUSIVE: /slots value never changed (replica idle?). \
             Put the replica under concurrent load and re-run."
        );
        return 2;
    }

    let mut ci = change_intervals_ms.clone();
    ci.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (p50, p95, max) = (pct(&ci, 50.0), pct(&ci, 95.0), ci[ci.len() - 1]);
    println!(
        "[diagnose] observed /slots update interval ms: p50={p50:.1} p95={p95:.1} max={max:.1}  (poll_interval={poll_ms:.0}ms)"
    );

    // PASS iff the engine refreshes /slots at least as fast as we poll (p95 of
    // the update interval within the poll interval).
    if p95 <= poll_ms {
        println!("[diagnose] PASS: /slots refreshes at least as fast as the poll interval — state is fresh.");
        0
    } else {
        println!(
            "[diagnose] FAIL (WARN LOUDLY): /slots refreshes SLOWER than the poll interval \
             (p95 update {p95:.1}ms > {poll_ms:.0}ms). You are reading STALE state no matter how fast you poll. \
             Increase poll_interval_ms toward the observed update interval, or treat u/B as coarser than you assumed."
        );
        1
    }
}

fn pct(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

async fn fetch(client: &Client, url: &str, timeout: Duration) -> Result<String, String> {
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
