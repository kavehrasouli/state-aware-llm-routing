use crate::replica::ReplicaState;
use arc_swap::ArcSwap;
use std::sync::Arc;

pub type Snapshot = Vec<ReplicaState>;

pub struct StateCache {
    inner: ArcSwap<Snapshot>,
    n: usize,
}

impl StateCache {
    pub fn new(initial: Snapshot) -> Self {
        let n = initial.len();
        StateCache {
            inner: ArcSwap::from_pointee(initial),
            n,
        }
    }

    /// Wait-free read of a consistent, single-cycle snapshot.
    pub fn load_full(&self) -> Arc<Snapshot> {
        self.inner.load_full()
    }

    /// Atomically publish a fresh whole-cluster snapshot.
    pub fn store(&self, s: Snapshot) {
        self.inner.store(Arc::new(s));
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.n
    }
}
