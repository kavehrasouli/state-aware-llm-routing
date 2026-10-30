//! Parsers for the two llama.cpp endpoints. Kept pure and unit-testable.

use serde_json::Value;

pub struct SlotsParsed {
    /// ΣR — resident tokens summed over busy slots.
    pub resident_tokens: u64,
    /// max single-slot resident tokens among busy slots -- the "shape" signal
    /// (distinguishes one dominant slot from an even spread at equal ΣR).
    /// 0 when no slots are busy.
    pub max_slot_tokens: u64,
    /// count of slots with is_processing = true (effective concurrency).
    pub busy_slots: u64,
    pub n_slots: u64,
    /// Σ n_ctx over all slots — used only to cross-check config `context_budget`.
    pub total_n_ctx: u64,
    /// Largest single-slot n_ctx — the cross-check value under unified KV,
    /// where slots share one buffer and Σ n_ctx overstates capacity.
    pub max_n_ctx: u64,
}

/// Parse the `/slots` JSON array (llama.cpp fields `is_processing`, `n_ctx`, `n_prompt_tokens`).
pub fn parse_slots(text: &str) -> Result<SlotsParsed, String> {
    let v: Value = serde_json::from_str(text).map_err(|e| format!("/slots not JSON: {e}"))?;
    let arr = v
        .as_array()
        .ok_or_else(|| "/slots did not return a JSON array".to_string())?;

    let mut resident = 0u64;
    let mut max_slot = 0u64;
    let mut busy = 0u64;
    let mut total_ctx = 0u64;
    let mut max_ctx = 0u64;

    for slot in arr {
        let processing = slot
            .get("is_processing")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let slot_ctx = slot.get("n_ctx").and_then(Value::as_u64).unwrap_or(0);
        total_ctx += slot_ctx;
        max_ctx = max_ctx.max(slot_ctx);
        if processing {
            busy += 1;
            // n_prompt_tokens = prompt+generated tokens resident in this slot's KV.
            // n_past is a defensive fallback if the field name differs by build.
            let toks = slot
                .get("n_prompt_tokens")
                .and_then(Value::as_u64)
                .or_else(|| slot.get("n_past").and_then(Value::as_u64))
                .unwrap_or(0);
            resident += toks;
            max_slot = max_slot.max(toks);
        }
    }

    Ok(SlotsParsed {
        resident_tokens: resident,
        max_slot_tokens: max_slot,
        busy_slots: busy,
        n_slots: arr.len() as u64,
        total_n_ctx: total_ctx,
        max_n_ctx: max_ctx,
    })
}

/// Reads one scalar from Prometheus `/metrics` text, skipping comment lines and matching the exact key.
pub fn parse_metric(text: &str, name: &str) -> Option<f64> {
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.split_whitespace();
        let key = it.next()?;
        let key_name = key.split('{').next().unwrap_or(key);
        if key_name == name {
            if let Some(val) = it.next() {
                if let Ok(f) = val.parse::<f64>() {
                    return Some(f);
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slots_sums_busy_only() {
        let json = r#"[
            {"id":0,"n_ctx":1024,"is_processing":true,"n_prompt_tokens":200},
            {"id":1,"n_ctx":1024,"is_processing":false,"n_prompt_tokens":999},
            {"id":2,"n_ctx":1024,"is_processing":true,"n_prompt_tokens":50},
            {"id":3,"n_ctx":1024,"is_processing":false,"n_prompt_tokens":0}
        ]"#;
        let p = parse_slots(json).unwrap();
        assert_eq!(p.resident_tokens, 250); // only busy slots
        assert_eq!(p.max_slot_tokens, 200); // larger of the two busy slots (200, 50)
        assert_eq!(p.busy_slots, 2);
        assert_eq!(p.n_slots, 4);
        assert_eq!(p.total_n_ctx, 4096);
        assert_eq!(p.max_n_ctx, 1024);
    }

    #[test]
    fn metric_skips_help_type_and_labels() {
        let text = "\
# HELP llamacpp:requests_processing Number of requests processing.
# TYPE llamacpp:requests_processing gauge
llamacpp:requests_processing 3
llamacpp:requests_deferred 1
llamacpp:kv_cache_usage_ratio 0.42
";
        assert_eq!(
            parse_metric(text, "llamacpp:requests_processing"),
            Some(3.0)
        );
        assert_eq!(parse_metric(text, "llamacpp:requests_deferred"), Some(1.0));
        assert_eq!(
            parse_metric(text, "llamacpp:kv_cache_usage_ratio"),
            Some(0.42)
        );
        assert_eq!(parse_metric(text, "llamacpp:nonexistent"), None);
    }
}
