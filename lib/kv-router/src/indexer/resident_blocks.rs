// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-rank resident block counts, published off the request path.
//!
//! Event threads own the per-rank block maps, so reading them is a queued round
//! trip. A poller makes that trip and publishes the result here, where worker
//! selection can read it without waiting on the indexer.

use std::sync::Arc;
use std::time::Duration;

use arc_swap::{ArcSwapOption, Guard};
use rustc_hash::FxHashMap;

use crate::protocols::WorkerWithDpRank;

/// How often routing hosts republish [`ResidentBlockCounts`].
#[cfg_attr(not(feature = "standalone-selection"), allow(dead_code))]
pub(crate) const RESIDENT_BLOCK_COUNTS_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Blocks the indexer tracked for each worker rank when polled.
#[derive(Debug)]
pub(crate) struct ResidentBlockCounts {
    counts: FxHashMap<WorkerWithDpRank, u64>,
}

impl ResidentBlockCounts {
    /// Blocks tracked for `worker`; zero when the indexer tracks none for it.
    pub(crate) fn get(&self, worker: WorkerWithDpRank) -> u64 {
        self.counts.get(&worker).copied().unwrap_or(0)
    }
}

impl FromIterator<(WorkerWithDpRank, usize)> for ResidentBlockCounts {
    fn from_iter<I: IntoIterator<Item = (WorkerWithDpRank, usize)>>(iter: I) -> Self {
        let mut counts = FxHashMap::default();
        for (worker, blocks) in iter {
            *counts.entry(worker).or_default() += blocks as u64;
        }
        Self { counts }
    }
}

/// The latest published [`ResidentBlockCounts`], shared by the poller and its readers.
#[derive(Clone, Default)]
pub(crate) struct ResidentBlockCountsHandle {
    latest: Arc<ArcSwapOption<ResidentBlockCounts>>,
}

impl ResidentBlockCountsHandle {
    /// The latest counts, or `None` before the first poll, while the indexer is backlogged, or
    /// after the poller stops. Hold the guard only for one selection.
    pub(crate) fn load(&self) -> Guard<Option<Arc<ResidentBlockCounts>>> {
        self.latest.load()
    }

    #[cfg_attr(not(feature = "standalone-indexer"), allow(dead_code))]
    pub(crate) fn publish(&self, counts: ResidentBlockCounts) {
        self.latest.store(Some(Arc::new(counts)));
    }

    /// Withdraw the counts so readers see `None` rather than stale counts.
    #[cfg_attr(not(feature = "standalone-indexer"), allow(dead_code))]
    pub(crate) fn clear(&self) {
        self.latest.store(None);
    }
}
