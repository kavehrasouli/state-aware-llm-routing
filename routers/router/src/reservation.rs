//! Router-local reservations for dispatches not yet visible in the polled state (state_aware_v2 only).
//! Selection and increment share one mutex, so concurrent decisions see each other.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

struct Entry {
    replica: usize,
    dispatch_started_at: Option<Instant>,
}

struct Inner {
    counts: Vec<u64>,
    entries: HashMap<u64, Entry>,
    next_id: u64,
}

/// Shared pending-dispatch state. The mutex is deliberately scoped to v2's
/// selection/reservation transaction and brief poll/release updates.
pub struct PendingDispatches {
    inner: Mutex<Inner>,
}

impl PendingDispatches {
    pub fn new(n: usize) -> Self {
        Self {
            inner: Mutex::new(Inner {
                counts: vec![0; n],
                entries: HashMap::new(),
                next_id: 0,
            }),
        }
    }

    /// Run v2 selection against one mutually consistent pending-count view and
    /// reserve the chosen replica before releasing the lock.
    pub fn select_and_reserve<T>(
        self: &Arc<Self>,
        select: impl FnOnce(&[u64]) -> (Option<usize>, T),
    ) -> (Option<ReservationGuard>, T) {
        let mut inner = self.inner.lock().unwrap();
        let (choice, detail) = select(&inner.counts);
        let guard = choice.map(|replica| {
            assert!(
                replica < inner.counts.len(),
                "reservation replica index out of bounds"
            );
            let id = inner.next_id;
            inner.next_id = inner.next_id.wrapping_add(1);
            inner.counts[replica] += 1;
            inner.entries.insert(
                id,
                Entry {
                    replica,
                    dispatch_started_at: None,
                },
            );
            ReservationGuard {
                owner: self.clone(),
                id: Some(id),
            }
        });
        (guard, detail)
    }

    fn arm(&self, id: u64, dispatch_started_at: Instant) {
        if let Some(entry) = self.inner.lock().unwrap().entries.get_mut(&id) {
            entry.dispatch_started_at = Some(dispatch_started_at);
        }
    }

    fn release(&self, id: u64) -> bool {
        let mut inner = self.inner.lock().unwrap();
        let Some(entry) = inner.entries.remove(&id) else {
            return false;
        };
        debug_assert!(inner.counts[entry.replica] > 0);
        inner.counts[entry.replica] = inner.counts[entry.replica].saturating_sub(1);
        true
    }

    /// Expire reservations that a newly published successful poll could have
    /// observed. A poll that began before dispatch does not qualify.
    pub fn release_observed_by_poll(&self, replica: usize, poll_started_at: Instant) -> usize {
        let mut inner = self.inner.lock().unwrap();
        let ids: Vec<u64> = inner
            .entries
            .iter()
            .filter_map(|(&id, entry)| {
                (entry.replica == replica
                    && entry
                        .dispatch_started_at
                        .is_some_and(|dispatch_started_at| dispatch_started_at < poll_started_at))
                .then_some(id)
            })
            .collect();

        for id in &ids {
            let entry = inner
                .entries
                .remove(id)
                .expect("reservation disappeared while lock held");
            debug_assert!(inner.counts[entry.replica] > 0);
            inner.counts[entry.replica] = inner.counts[entry.replica].saturating_sub(1);
        }
        ids.len()
    }

    #[cfg(test)]
    pub(crate) fn counts(&self) -> Vec<u64> {
        self.inner.lock().unwrap().counts.clone()
    }

    #[cfg(test)]
    pub(crate) fn active_len(&self) -> usize {
        self.inner.lock().unwrap().entries.len()
    }
}

/// One idempotent RAII reservation. Poll expiry may remove its entry first;
/// later explicit release or Drop then becomes a harmless no-op.
pub struct ReservationGuard {
    owner: Arc<PendingDispatches>,
    id: Option<u64>,
}

impl ReservationGuard {
    /// Arm the reservation at the actual upstream dispatch boundary. Until
    /// armed, no poll can expire it.
    pub fn mark_dispatched(&mut self, dispatch_started_at: Instant) {
        if let Some(id) = self.id {
            self.owner.arm(id, dispatch_started_at);
        }
    }

    pub fn release(&mut self) {
        if let Some(id) = self.id.take() {
            self.owner.release(id);
        }
    }
}

impl Drop for ReservationGuard {
    fn drop(&mut self) {
        self.release();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn reserve(book: &Arc<PendingDispatches>, replica: usize) -> ReservationGuard {
        book.select_and_reserve(|_| (Some(replica), ())).0.unwrap()
    }

    #[test]
    fn drop_and_explicit_release_are_idempotent() {
        let book = Arc::new(PendingDispatches::new(1));
        let mut guard = reserve(&book, 0);
        assert_eq!(book.counts(), vec![1]);
        guard.release();
        guard.release();
        assert_eq!(book.counts(), vec![0]);
        assert_eq!(book.active_len(), 0);
        drop(guard);
        assert_eq!(book.counts(), vec![0]);
    }

    #[test]
    fn only_poll_started_after_dispatch_expires_reservation() {
        let book = Arc::new(PendingDispatches::new(1));
        let before_dispatch = Instant::now();
        let mut guard = reserve(&book, 0);

        assert_eq!(book.release_observed_by_poll(0, before_dispatch), 0);
        assert_eq!(
            book.counts(),
            vec![1],
            "unarmed reservation must survive polls"
        );

        let dispatch = Instant::now();
        guard.mark_dispatched(dispatch);
        assert_eq!(book.release_observed_by_poll(0, before_dispatch), 0);
        assert_eq!(book.counts(), vec![1], "pre-dispatch poll must not release");

        let after_dispatch = dispatch + Duration::from_millis(1);
        assert_eq!(book.release_observed_by_poll(0, after_dispatch), 1);
        assert_eq!(book.counts(), vec![0]);
        assert_eq!(book.active_len(), 0);

        guard.release();
        assert_eq!(
            book.counts(),
            vec![0],
            "guard release after poll expiry is a no-op"
        );
    }

    #[test]
    fn poll_expiry_is_replica_local() {
        let book = Arc::new(PendingDispatches::new(2));
        let mut a = reserve(&book, 0);
        let mut b = reserve(&book, 1);
        let dispatch = Instant::now();
        a.mark_dispatched(dispatch);
        b.mark_dispatched(dispatch);

        assert_eq!(
            book.release_observed_by_poll(0, dispatch + Duration::from_millis(1)),
            1
        );
        assert_eq!(book.counts(), vec![0, 1]);
    }
}
