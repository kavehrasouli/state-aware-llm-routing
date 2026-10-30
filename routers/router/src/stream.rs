//! Response stream: forwards bytes unchanged, records TTFT at the first content token,
//! and fails a request that stalls longer than the idle timeout.

use crate::record::{Outcome, RequestRecord};
use bytes::Bytes;
use futures::{Stream, StreamExt};
use std::time::Duration;

pub fn instrumented_stream(
    upstream: impl Stream<Item = reqwest::Result<Bytes>> + Send + 'static,
    mut record: RequestRecord,
    idle_timeout: Duration,
) -> impl Stream<Item = Result<Bytes, std::io::Error>> {
    async_stream::stream! {
        futures::pin_mut!(upstream);
        let mut ttft_seen = false;
        let mut sse = SseCounter::default();

        loop {
            match tokio::time::timeout(idle_timeout, upstream.next()).await {
                Ok(Some(Ok(chunk))) => {
                    if !ttft_seen && chunk_has_content(&chunk) {
                        record.mark_ttft();
                        ttft_seen = true;
                    }
                    let fed = sse.feed(&chunk);
                    record.add_output_tokens(fed.content_events);
                    if let Some((prompt, completion)) = fed.usage {
                        record.set_usage(prompt, completion);
                    }
                    yield Ok(chunk);
                }
                Ok(Some(Err(e))) => {
                    record.set_outcome(Outcome::UpstreamError);
                    yield Err(std::io::Error::other(e.to_string()));
                    break;
                }
                Ok(None) => {
                    record.set_outcome(Outcome::Success);
                    break;
                }
                Err(_elapsed) => {
                    // No chunk within idle_timeout: terminal failure, no retry
                    // (F9 no-migration). Marked so it never vanishes from stats.
                    record.set_outcome(Outcome::UpstreamTimeoutStream);
                    yield Err(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        "upstream stream idle timeout",
                    ));
                    break;
                }
            }
        }
        // `record` drops here (or earlier if the client hung up) -> finalize().
        drop(record);
    }
}

/// What one fed chunk contained: content events (each normally one output
/// token; llama.cpp can merge tokens of incomplete multi-byte text, so this may
/// undercount) and the final usage block, if present.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct SseFeed {
    pub content_events: u64,
    pub usage: Option<(u64, u64)>,
}

/// Line-based SSE parser for the router-tracked output count. Keeps an
/// incomplete trailing line across chunks so an event split over two network
/// chunks is counted once.
#[derive(Default)]
pub struct SseCounter {
    carry: Vec<u8>,
}

impl SseCounter {
    pub fn feed(&mut self, chunk: &[u8]) -> SseFeed {
        self.carry.extend_from_slice(chunk);
        let mut out = SseFeed::default();
        while let Some(pos) = self.carry.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = self.carry.drain(..=pos).collect();
            let Ok(line) = std::str::from_utf8(&line) else {
                continue;
            };
            let Some(data) = line.trim().strip_prefix("data:") else {
                continue;
            };
            let data = data.trim();
            if data == "[DONE]" {
                continue;
            }
            let Ok(v) = serde_json::from_str::<serde_json::Value>(data) else {
                continue;
            };
            if let Some(choice) = v.get("choices").and_then(|c| c.get(0)) {
                let text = choice
                    .get("text")
                    .or_else(|| choice.get("delta").and_then(|d| d.get("content")))
                    .and_then(|t| t.as_str());
                if text.is_some_and(|t| !t.is_empty()) {
                    out.content_events += 1;
                }
            }
            if let Some(usage) = v.get("usage").filter(|u| u.is_object()) {
                let get = |k| usage.get(k).and_then(serde_json::Value::as_u64);
                if let (Some(p), Some(c)) = (get("prompt_tokens"), get("completion_tokens")) {
                    out.usage = Some((p, c));
                }
            }
        }
        out
    }
}

/// Heuristic first-token detector for SSE. Client-side TTFT is authoritative;
/// this only sets the router-internal cross-check. Skips role-only deltas
/// (`"content":""`) and matches both chat (`content`) and completions (`text`).
fn chunk_has_content(chunk: &[u8]) -> bool {
    let s = String::from_utf8_lossy(chunk);
    for marker in ["\"content\":\"", "\"text\":\""] {
        if let Some(pos) = s.find(marker) {
            let after = &s[pos + marker.len()..];
            if let Some(c) = after.chars().next() {
                if c != '"' {
                    return true; // non-empty string value
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_completion_and_chat_events_and_usage() {
        let mut c = SseCounter::default();
        let body = concat!(
            "data: {\"choices\":[{\"text\":\"Hel\"}]}\n\n",
            "data: {\"choices\":[{\"text\":\"\"}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":641,\"completion_tokens\":2}}\n\n",
            "data: [DONE]\n\n",
        );
        let fed = c.feed(body.as_bytes());
        assert_eq!(fed.content_events, 2);
        assert_eq!(fed.usage, Some((641, 2)));
    }

    #[test]
    fn event_split_across_chunks_counts_once() {
        let mut c = SseCounter::default();
        let a = c.feed(b"data: {\"choices\":[{\"te");
        assert_eq!(a.content_events, 0);
        let b = c.feed(b"xt\":\" word\"}]}\n\ndata: {\"choices\":[{\"text\":\"x\"}]}\n");
        assert_eq!(b.content_events, 2);
    }

    #[test]
    fn ignores_non_data_and_malformed_lines() {
        let mut c = SseCounter::default();
        let fed = c.feed(b": ping\nevent: x\ndata: not-json\n\n");
        assert_eq!(fed, SseFeed::default());
    }
}
