//! BAM/CRAM alignment reader: `backend` decodes records, `mate`/`walk` turn
//! them into depth/composition increments, and `reader`/`import` wire that
//! into the `ValueReader` pipeline as `BamReader`/`from_bam`.

// Submodules below are `pub` only so `tests/bam_*.rs` can exercise their
// internals directly (D-2); they are not part of the curated public API and
// carry `#[doc(hidden)]` so they don't show up in generated docs.
#[doc(hidden)]
pub mod backend;
mod import;
#[doc(hidden)]
pub mod mate;
mod reader;
#[doc(hidden)]
pub mod record;
#[doc(hidden)]
pub mod walk;

pub use import::from_bam;
pub use reader::BamReader;
pub use record::{AlignedRead, CigarKind};
pub use walk::ImportMode;

/// Which per-record fields a decode pass has to fill.
///
/// Depth mode with `min_bq == 0` never consults a base quality and never
/// looks at a base letter, so both backends can leave `AlignedRead::seq` and
/// `AlignedRead::qual` empty. On CRAM that is worth more than the skipped
/// copy: htslib is told through `CRAM_OPT_REQUIRED_FIELDS` not to decode the
/// SEQ and QUAL data series at all.
///
/// `pub` rather than `pub(crate)` because `Backend::open` takes it and is
/// itself `pub` in a `pub` module, which makes a `pub(crate)` parameter type
/// trip `private_interfaces` under `-D warnings`. `#[doc(hidden)]` keeps it
/// out of the generated docs, same as the submodules above.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordFields {
    /// Decode everything, including SEQ and QUAL.
    Full,
    /// Decode only what the depth walk reads; SEQ and QUAL stay empty.
    DepthOnly,
}

impl RecordFields {
    /// The depth walk reads a quality only to compare it against `min_bq`,
    /// and reads a base letter only in composition mode.
    pub fn for_pass(mode: ImportMode, filter: &DepthFilter) -> Self {
        if mode == ImportMode::Depth && filter.min_bq == 0 {
            RecordFields::DepthOnly
        } else {
            RecordFields::Full
        }
    }
}

#[cfg(test)]
mod record_fields_tests {
    use super::*;

    fn filter(min_bq: u8) -> DepthFilter {
        DepthFilter {
            min_bq,
            ..DepthFilter::default()
        }
    }

    /// All four corners of the `(mode, min_bq)` grid. Only depth with an
    /// ungated base quality may drop SEQ and QUAL; composition reads a base
    /// letter at every position, whatever `min_bq` says.
    #[test]
    fn for_pass_drops_seq_and_qual_only_for_ungated_depth() {
        assert_eq!(
            RecordFields::for_pass(ImportMode::Depth, &filter(0)),
            RecordFields::DepthOnly
        );
        assert_eq!(
            RecordFields::for_pass(ImportMode::Depth, &filter(20)),
            RecordFields::Full
        );
        assert_eq!(
            RecordFields::for_pass(ImportMode::Composition, &filter(0)),
            RecordFields::Full
        );
        assert_eq!(
            RecordFields::for_pass(ImportMode::Composition, &filter(20)),
            RecordFields::Full
        );
    }
}

/// How mate-overlap dedup gates on the `PROPER_PAIR` flag (0x2).
///
/// `ProperOnly` (the default) matches mosdepth: only pairs both flagged
/// proper-pair get their shared span collapsed to one count, so a discordant
/// (improper) overlapping pair double-counts, same as mosdepth's default
/// non-fast-mode behavior. `All` matches riker/samtools-mpileup-style
/// unconditional dedup: any same-contig, mapped, non-secondary/
/// non-supplementary mate pair whose spans overlap gets collapsed,
/// proper-pair or not. `None` disables overlap dedup entirely, so every
/// overlapping pair double-counts its shared span.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OverlapMode {
    /// No dedup: every overlapping mate pair double-counts its shared span.
    None,
    /// Dedup only pairs both flagged `PROPER_PAIR` (mosdepth's default).
    #[default]
    ProperOnly,
    /// Dedup every overlapping mate pair regardless of `PROPER_PAIR`
    /// (riker/samtools-mpileup-style unconditional dedup).
    All,
}

/// Per-record filtering and overlap-handling knobs for the depth/composition
/// walk. Defaults (`min_mapq = 0`, `exclude_flags = 1796`, `min_bq = 0`,
/// `overlap = ProperOnly`, `count_deletions = false`) are the
/// mosdepth-verified consensus (samtools/mosdepth/Picard agree), not a
/// riker mirror: riker's own default dedups every overlapping pair
/// unconditionally (`OverlapMode::All`), which mosdepth only does when both
/// mates are flagged `PROPER_PAIR`. See `bam-import-bench/workflow/scripts/
/// parity_mosdepth.py` for the verification writeup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DepthFilter {
    /// Minimum mapping quality; reads below this are excluded entirely.
    pub min_mapq: u8,
    /// SAM flag bits that exclude a record when any are set (default 1796 =
    /// UNMAP(4)|SECONDARY(256)|QCFAIL(512)|DUP(1024)).
    pub exclude_flags: u16,
    /// Minimum per-base quality; bases below this don't contribute to depth
    /// or a/c/g/t (they bump `n` in Composition mode instead).
    pub min_bq: u8,
    /// How mate-overlap dedup gates on the `PROPER_PAIR` flag; see [`OverlapMode`].
    pub overlap: OverlapMode,
    /// Whether CIGAR `D` (deletion) positions count toward depth. `false`
    /// (default) matches the samtools/mosdepth/Picard/riker consensus,
    /// which excludes deletions from depth entirely; `true` counts
    /// deletion spans toward depth, useful for callability masks. Either
    /// way, the composition `del` field always counts the deletion event;
    /// this only gates the depth increment.
    pub count_deletions: bool,
}

impl Default for DepthFilter {
    fn default() -> Self {
        Self {
            min_mapq: 0,
            exclude_flags: 1796,
            min_bq: 0,
            overlap: OverlapMode::default(),
            count_deletions: false,
        }
    }
}
