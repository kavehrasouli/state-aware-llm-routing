//! Per-replica eligibility, separate from state freshness.
//! A slow `/metrics` poll only marks the state stale (a long prefill can block it while the node
//! still serves). Eviction happens only after `evict_threshold` consecutive connection failures
//! on the data path; recovery only after a successful poll. Eligibility is one atomic bitmask.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

pub struct Reachability {
    /// bit i set = replica i is eligible for selection.
    eligible: AtomicU64,
    /// consecutive connection-level dispatch failures, per replica.
    dispatch_failures: Vec<AtomicU32>,
    /// consecutive dispatch failures required to evict.
    evict_threshold: u32,
}

impl Reachability {
    /// Start with everyone INELIGIBLE: the warm poll admits reachable nodes
    /// (via `on_poll_success`) before serving, so nothing routes on an all-zero
    /// mask in practice.
    pub fn new(n: usize, evict_threshold: u32) -> Self {
        let mut dispatch_failures = Vec::with_capacity(n);
        for _ in 0..n {
            dispatch_failures.push(AtomicU32::new(0));
        }
        Reachability {
            eligible: AtomicU64::new(0),
            dispatch_failures,
            evict_threshold,
        }
    }

    /// One atomic read — the consistent eligibility view a single decision uses.
    pub fn load(&self) -> u64 {
        self.eligible.load(Ordering::Acquire)
    }

    #[inline]
    pub fn bit(mask: u64, i: usize) -> bool {
        mask & (1u64 << i) != 0
    }

    /// POLLER, on a successful /metrics poll: reset the failure streak and mark
    /// eligible. Returns true iff this was a transition ineligible->eligible (a
    /// recovery or first admit), so the caller logs it exactly once.
    pub fn on_poll_success(&self, i: usize) -> bool {
        self.dispatch_failures[i].store(0, Ordering::Release);
        let prev = self.eligible.fetch_or(1u64 << i, Ordering::AcqRel);
        prev & (1u64 << i) == 0
    }

    /// DATA PATH, on a successful dispatch (connection established): reset the
    /// failure streak. Eligibility is unchanged (we only dispatch to eligible
    /// nodes, so the bit is already set).
    pub fn on_dispatch_success(&self, i: usize) {
        self.dispatch_failures[i].store(0, Ordering::Release);
    }

    /// DATA PATH, on a CONNECTION-LEVEL dispatch failure. Increments the streak;
    /// once it reaches `evict_threshold`, clears the eligibility bit. Returns
    /// Some(streak) iff THIS call caused the eviction transition (bit was set,
    /// now cleared) so the caller emits exactly one eviction event.
    pub fn on_dispatch_failure(&self, i: usize) -> Option<u32> {
        let streak = self.dispatch_failures[i].fetch_add(1, Ordering::AcqRel) + 1;
        if streak >= self.evict_threshold {
            let prev = self.eligible.fetch_and(!(1u64 << i), Ordering::AcqRel);
            if prev & (1u64 << i) != 0 {
                return Some(streak);
            }
        }
        None
    }

    #[cfg(test)]
    pub fn is_eligible(&self, i: usize) -> bool {
        Reachability::bit(self.load(), i)
    }

    #[cfg(test)]
    pub fn failures(&self, i: usize) -> u32 {
        self.dispatch_failures[i].load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::Reachability;

    #[test]
    fn evicts_only_after_threshold_consecutive_dispatch_failures() {
        let r = Reachability::new(1, 3);
        assert!(r.on_poll_success(0)); // admit (transition true)
        assert!(r.is_eligible(0));

        assert_eq!(r.on_dispatch_failure(0), None); // 1
        assert!(r.is_eligible(0));
        assert_eq!(r.on_dispatch_failure(0), None); // 2
        assert!(r.is_eligible(0));
        assert_eq!(r.on_dispatch_failure(0), Some(3)); // 3 -> evict
        assert!(!r.is_eligible(0));
    }

    #[test]
    fn dispatch_success_resets_the_streak() {
        let r = Reachability::new(1, 3);
        r.on_poll_success(0);
        r.on_dispatch_failure(0);
        r.on_dispatch_failure(0);
        r.on_dispatch_success(0); // reset
        assert_eq!(r.failures(0), 0);
        // Two more failures must NOT evict (streak restarted).
        assert_eq!(r.on_dispatch_failure(0), None);
        assert_eq!(r.on_dispatch_failure(0), None);
        assert!(r.is_eligible(0));
    }

    #[test]
    fn poll_success_recovers_an_evicted_node() {
        let r = Reachability::new(1, 2);
        r.on_poll_success(0);
        r.on_dispatch_failure(0);
        assert_eq!(r.on_dispatch_failure(0), Some(2)); // evicted
        assert!(!r.is_eligible(0));

        assert!(r.on_poll_success(0)); // recovery transition
        assert!(r.is_eligible(0));
        assert_eq!(r.failures(0), 0);
    }

    #[test]
    fn eviction_transition_is_reported_once() {
        let r = Reachability::new(1, 1);
        r.on_poll_success(0);
        assert_eq!(r.on_dispatch_failure(0), Some(1)); // first failure evicts (threshold 1)
        assert_eq!(r.on_dispatch_failure(0), None); // already evicted -> no duplicate event
    }
}
