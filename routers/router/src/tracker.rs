//! Router-tracked load: per replica, the requests dispatched and not yet finished and their
//! context (prompt tokens plus streamed output tokens). All policies maintain it; only the
//! `*_tracked` policies route on it. Updated under one mutex with selection.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TrackedLoad {
    pub requests: u64,
    pub tokens: u64,
}

struct Entry {
    replica: usize,
    tokens: u64,
}

struct Inner {
    loads: Vec<TrackedLoad>,
    entries: HashMap<u64, Entry>,
    next_id: u64,
}

pub struct LoadTracker {
    inner: Mutex<Inner>,
}

impl LoadTracker {
    pub fn new(n: usize) -> Self {
        Self {
            inner: Mutex::new(Inner {
                loads: vec![TrackedLoad::default(); n],
                entries: HashMap::new(),
                next_id: 0,
            }),
        }
    }

    /// Run `select` against one consistent view and add the request to the
    /// chosen replica before releasing the lock. Returns the view the
    /// selection saw (before this request was added).
    pub fn select_and_track(
        self: &Arc<Self>,
        prompt_tokens: u64,
        select: impl FnOnce(&[TrackedLoad]) -> Option<usize>,
    ) -> (Option<TrackGuard>, Vec<TrackedLoad>) {
        let mut inner = self.inner.lock().unwrap();
        let view = inner.loads.clone();
        let guard = select(&inner.loads).map(|replica| {
            assert!(
                replica < inner.loads.len(),
                "tracked replica index out of bounds"
            );
            let id = inner.next_id;
            inner.next_id = inner.next_id.wrapping_add(1);
            inner.loads[replica].requests += 1;
            inner.loads[replica].tokens += prompt_tokens;
            inner.entries.insert(
                id,
                Entry {
                    replica,
                    tokens: prompt_tokens,
                },
            );
            TrackGuard {
                owner: self.clone(),
                id: Some(id),
                output_tokens: 0,
            }
        });
        (guard, view)
    }

    /// Track a request on a replica another policy already chose.
    pub fn track(
        self: &Arc<Self>,
        replica: usize,
        prompt_tokens: u64,
    ) -> (TrackGuard, Vec<TrackedLoad>) {
        let (guard, view) = self.select_and_track(prompt_tokens, |_| Some(replica));
        (guard.expect("fixed choice always tracks"), view)
    }

    fn add_tokens(&self, id: u64, n: u64) {
        let mut inner = self.inner.lock().unwrap();
        let Some(entry) = inner.entries.get_mut(&id) else {
            return;
        };
        entry.tokens += n;
        let replica = entry.replica;
        inner.loads[replica].tokens += n;
    }

    fn release(&self, id: u64) {
        let mut inner = self.inner.lock().unwrap();
        let Some(entry) = inner.entries.remove(&id) else {
            return;
        };
        let load = &mut inner.loads[entry.replica];
        debug_assert!(load.requests > 0 && load.tokens >= entry.tokens);
        load.requests = load.requests.saturating_sub(1);
        load.tokens = load.tokens.saturating_sub(entry.tokens);
    }

    #[cfg(test)]
    pub(crate) fn loads(&self) -> Vec<TrackedLoad> {
        self.inner.lock().unwrap().loads.clone()
    }
}

/// One tracked request. Released on the request's terminal outcome; Drop is
/// the cancellation-safe fallback. Release is idempotent.
pub struct TrackGuard {
    owner: Arc<LoadTracker>,
    id: Option<u64>,
    output_tokens: u64,
}

impl TrackGuard {
    pub fn add_output_tokens(&mut self, n: u64) {
        if n == 0 {
            return;
        }
        if let Some(id) = self.id {
            self.output_tokens += n;
            self.owner.add_tokens(id, n);
        }
    }

    pub fn output_tokens(&self) -> u64 {
        self.output_tokens
    }

    pub fn release(&mut self) {
        if let Some(id) = self.id.take() {
            self.owner.release(id);
        }
    }
}

impl Drop for TrackGuard {
    fn drop(&mut self) {
        self.release();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn track_grow_and_release() {
        let t = Arc::new(LoadTracker::new(2));
        let (mut a, view) = t.track(0, 640);
        assert_eq!(view, vec![TrackedLoad::default(); 2]);
        assert_eq!(t.loads()[0], TrackedLoad { requests: 1, tokens: 640 });

        a.add_output_tokens(1);
        a.add_output_tokens(49);
        assert_eq!(t.loads()[0], TrackedLoad { requests: 1, tokens: 690 });
        assert_eq!(a.output_tokens(), 50);

        let (_b, view) = t.track(0, 40);
        assert_eq!(view[0], TrackedLoad { requests: 1, tokens: 690 });
        assert_eq!(t.loads()[0], TrackedLoad { requests: 2, tokens: 730 });

        a.release();
        a.release();
        assert_eq!(t.loads()[0], TrackedLoad { requests: 1, tokens: 40 });
        a.add_output_tokens(5);
        assert_eq!(t.loads()[0], TrackedLoad { requests: 1, tokens: 40 });
    }

    #[test]
    fn drop_releases() {
        let t = Arc::new(LoadTracker::new(1));
        {
            let (mut g, _) = t.track(0, 100);
            g.add_output_tokens(3);
        }
        assert_eq!(t.loads()[0], TrackedLoad::default());
    }

    #[test]
    fn select_sees_view_and_none_tracks_nothing() {
        let t = Arc::new(LoadTracker::new(2));
        let (_g, _) = t.track(1, 10);
        let (guard, view) = t.select_and_track(5, |loads| {
            assert_eq!(loads[1].tokens, 10);
            None
        });
        assert!(guard.is_none());
        assert_eq!(view[1].tokens, 10);
        assert_eq!(t.loads(), vec![TrackedLoad::default(), TrackedLoad { requests: 1, tokens: 10 }]);
    }
}
