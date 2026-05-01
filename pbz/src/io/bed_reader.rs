//! BED file reader. Streams bgzipped BED into an in-memory interval map.
//!
//! `BedIntervalSource` is the canonical interval producer. `BedPerBaseReader`
//! adapts it to the per-base `ValueReader` contract used by the v0.1 ingest
//! pipeline; the v0.2 interval-layout writer will consume `BedIntervalSource`
//! directly.

use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BedFlavor {
    Mask,
    BedGraph,
}

impl BedFlavor {
    pub(crate) fn expected_cols(self) -> usize {
        match self {
            BedFlavor::Mask => 3,
            BedFlavor::BedGraph => 4,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum IntervalValue {
    Mask,
    Numeric(f64),
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct IntervalRecord {
    pub start: u64,
    pub end: u64,
    pub value: IntervalValue,
}

/// Owns the result of streaming a single BED file: a flavor and a per-contig
/// vector of sorted, non-overlapping intervals.
pub(crate) struct BedIntervalSource {
    pub(crate) path: PathBuf,
    pub(crate) flavor: BedFlavor,
    pub(crate) contigs: Vec<(String, u64)>,
    pub(crate) by_contig: HashMap<String, Vec<IntervalRecord>>,
}
