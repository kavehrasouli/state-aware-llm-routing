//! Fast prompt-token estimate for the predictive policy: max(words, ceil(chars / 4));
//! token-id arrays are counted exactly.

use serde_json::Value;

const CHAT_TOKENS_PER_MESSAGE: u64 = 4;
const CHAT_GENERATION_OVERHEAD: u64 = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RequestTokenEstimate {
    pub estimated: u64,
    pub true_count: Option<u64>,
    pub exact: bool,
    pub method: &'static str,
}

pub fn estimate_request_tokens(body: &[u8]) -> Result<RequestTokenEstimate, String> {
    let request: Value =
        serde_json::from_slice(body).map_err(|e| format!("request body is not valid JSON: {e}"))?;

    if let Some(prompt) = request.get("prompt") {
        if let Some(count) = token_id_count(prompt) {
            return Ok(RequestTokenEstimate {
                estimated: count,
                true_count: Some(count),
                exact: true,
                method: "token_id_array_exact",
            });
        }
        return Ok(RequestTokenEstimate {
            estimated: content_tokens(prompt),
            true_count: None,
            exact: false,
            method: "hybrid_word_char_v1",
        });
    }

    if let Some(messages) = request.get("messages").and_then(Value::as_array) {
        let content: u64 = messages
            .iter()
            .map(|message| message.get("content").map(content_tokens).unwrap_or(0))
            .sum();
        return Ok(RequestTokenEstimate {
            estimated: content
                .saturating_add(CHAT_TOKENS_PER_MESSAGE * messages.len() as u64)
                .saturating_add(CHAT_GENERATION_OVERHEAD),
            true_count: None,
            exact: false,
            method: "hybrid_word_char_chat_v1",
        });
    }

    Err("request body has neither 'prompt' nor 'messages'".into())
}

fn text_tokens(text: &str) -> u64 {
    let words = text.split_whitespace().count() as u64;
    let chars = text.chars().count() as u64;
    words.max(chars.div_ceil(4))
}

fn content_tokens(value: &Value) -> u64 {
    match value {
        Value::String(text) => text_tokens(text),
        Value::Array(parts) => parts
            .iter()
            .map(|part| match part {
                Value::String(text) => text_tokens(text),
                Value::Object(_) => part
                    .get("text")
                    .and_then(Value::as_str)
                    .map(text_tokens)
                    .unwrap_or(0),
                _ => 0,
            })
            .sum(),
        _ => 0,
    }
}

fn token_id_count(value: &Value) -> Option<u64> {
    let items = value.as_array()?;
    if items.iter().all(Value::is_i64) {
        return Some(items.len() as u64);
    }
    if items.iter().all(|item| {
        item.as_array()
            .is_some_and(|tokens| tokens.iter().all(Value::is_i64))
    }) {
        return Some(
            items
                .iter()
                .map(|item| item.as_array().map_or(0, Vec::len) as u64)
                .sum(),
        );
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completion_string_uses_hybrid_estimate() {
        let body = br#"{"prompt":"one two three four"}"#;
        let estimate = estimate_request_tokens(body).unwrap();
        assert_eq!(estimate.estimated, 5);
        assert!(!estimate.exact);
        assert_eq!(estimate.true_count, None);
        assert_eq!(estimate.method, "hybrid_word_char_v1");
    }

    #[test]
    fn token_id_arrays_are_exact() {
        let single = estimate_request_tokens(br#"{"prompt":[10,11,12]}"#).unwrap();
        assert_eq!(single.estimated, 3);
        assert_eq!(single.true_count, Some(3));
        assert!(single.exact);

        let batched = estimate_request_tokens(br#"{"prompt":[[1,2],[3,4,5]]}"#).unwrap();
        assert_eq!(batched.estimated, 5);
        assert_eq!(batched.true_count, Some(5));
        assert!(batched.exact);
    }

    #[test]
    fn chat_counts_text_parts_and_template_allowance() {
        let body = br#"{
            "messages": [
                {"role":"system","content":"be concise"},
                {"role":"user","content":[{"type":"text","text":"one two three"}]}
            ]
        }"#;
        let estimate = estimate_request_tokens(body).unwrap();
        assert_eq!(estimate.estimated, 3 + 4 + 2 * 4 + 3);
        assert_eq!(estimate.method, "hybrid_word_char_chat_v1");
        assert!(!estimate.exact);
    }

    #[test]
    fn malformed_or_unsupported_requests_fail_loudly() {
        assert!(estimate_request_tokens(b"not-json").is_err());
        assert!(estimate_request_tokens(br#"{"model":"x"}"#).is_err());
    }
}
