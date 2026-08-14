//! Normalized alignment record view shared by both BAM (noodles) and CRAM
//! (rust-htslib) backends, so downstream depth/composition code never has to
//! branch on which library decoded the record.

/// CIGAR operation kind, collapsed to the subset the importer cares about.
/// `H`/`P` (hard clip / pad) never advance the reference or contribute to
/// depth or composition, so both fold into `Other`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CigarKind {
    Match,
    Ins,
    Del,
    RefSkip,
    SoftClip,
    Other,
}

/// One alignment record, backend-agnostic. Callers recycle a single instance
/// across a `fetch` loop via [`AlignedRead::clear`] to avoid a per-record
/// allocation for `qname`/`cigar`/`seq`/`qual`.
#[derive(Debug, Default)]
pub struct AlignedRead {
    pub flags: u16,
    pub mapq: u8,
    /// 0-based alignment start on the queried contig.
    pub start: u64,
    /// 0-based mate alignment start, or `-1` when the mate is unmapped or
    /// the read is unpaired.
    pub mate_start: i64,
    pub mate_same_contig: bool,
    pub paired: bool,
    pub qname: Vec<u8>,
    pub cigar: Vec<(CigarKind, u32)>,
    pub seq: Vec<u8>,
    pub qual: Vec<u8>,
}

impl AlignedRead {
    /// Record the read name, normalizing SAM's "unavailable" placeholder to an
    /// empty name. noodles reports an absent BAM name as `None`, while htslib
    /// hands back the literal `*` it stores; both mean "this record has no
    /// name" and must reach the pairing logic identically, since an unnamed
    /// record can never be matched to a mate.
    pub fn set_qname(&mut self, name: &[u8]) {
        if name != b"*" {
            self.qname.extend_from_slice(name);
        }
    }

    /// Reset all buffers for reuse. Scalar fields are overwritten by the next
    /// decode regardless, so they're left as-is here.
    pub fn clear(&mut self) {
        self.qname.clear();
        self.cigar.clear();
        self.seq.clear();
        self.qual.clear();
    }
}
