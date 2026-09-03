//! Canonical genome types used across the `pbzarr` crate.
//!
//! - `ContigId` is a `u32`-newtype index into a [`Genome`]'s contig list.
//! - `Contig` is a `(name, length)` record.
//! - `Genome` owns an ordered list of contigs plus an O(1) name → id index.
//! - `Region` is a half-open `[start, end)` range over a `ContigId`.
//!
use crate::error::{PbzError, Result};
use crate::region_query::RegionQuery;
use hashbrown::HashMap;
use std::path::Path;

/// Index of a contig in a [`Genome`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ContigId(u32);

impl ContigId {
    pub fn as_u32(self) -> u32 {
        self.0
    }

    pub fn as_usize(self) -> usize {
        self.0 as usize
    }
}

impl std::fmt::Display for ContigId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "#{}", self.0)
    }
}

/// A contig (chromosome / sequence) with a name and length.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Contig {
    pub name: String,
    pub length: u64,
}

/// An ordered, named set of contigs.
///
/// Owns the canonical contig list for a reader, store, or other producer.
/// Provides O(1) lookup by name and by [`ContigId`], and serves as the
/// resolution context for turning a [`crate::RegionQuery`] into a concrete
/// [`Region`].
#[derive(Debug, Clone)]
pub struct Genome {
    contigs: Vec<Contig>,
    by_name: HashMap<String, ContigId>,
    name: Option<String>,
}

impl Genome {
    /// Build a `Genome` from an ordered list of contigs.
    ///
    /// Errors on duplicate names or empty names.
    pub fn new(contigs: Vec<Contig>) -> Result<Self> {
        if contigs.len() > u32::MAX as usize {
            return Err(PbzError::Metadata(format!(
                "too many contigs: {} (max {})",
                contigs.len(),
                u32::MAX
            )));
        }
        let mut by_name = HashMap::with_capacity(contigs.len());
        for (i, c) in contigs.iter().enumerate() {
            if c.name.is_empty() {
                return Err(PbzError::Metadata(format!("contig #{i} has empty name")));
            }
            if by_name.insert(c.name.clone(), ContigId(i as u32)).is_some() {
                return Err(PbzError::Metadata(format!(
                    "duplicate contig name: {:?}",
                    c.name
                )));
            }
        }
        Ok(Self {
            contigs,
            by_name,
            name: None,
        })
    }

    pub fn len(&self) -> usize {
        self.contigs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.contigs.is_empty()
    }

    /// All contigs in insertion order.
    pub fn contigs(&self) -> &[Contig] {
        &self.contigs
    }

    /// Look up a contig by id. Returns `None` if the id is out of range.
    pub fn get(&self, id: ContigId) -> Option<&Contig> {
        self.contigs.get(id.as_usize())
    }

    /// Look up a contig id by name.
    pub fn id(&self, name: &str) -> Option<ContigId> {
        self.by_name.get(name).copied()
    }

    /// Iterate `(ContigId, &Contig)` pairs in order.
    pub fn iter(&self) -> impl Iterator<Item = (ContigId, &Contig)> {
        self.contigs
            .iter()
            .enumerate()
            .map(|(i, c)| (ContigId(i as u32), c))
    }

    /// The prefix-sum flat-start index over this genome's contigs.
    ///
    /// Returns `k + 1` entries: `offsets[i]` is the flat start of contig `i`,
    /// `offsets[0] == 0`, and `offsets[k]` is `ΣL` (the total length). Written
    /// verbatim as the on-disk `offsets` array.
    pub fn offsets(&self) -> Vec<i64> {
        let mut out = Vec::with_capacity(self.contigs.len() + 1);
        let mut acc: i64 = 0;
        out.push(0);
        for c in &self.contigs {
            acc += c.length as i64;
            out.push(acc);
        }
        out
    }

    /// Attach an optional decorative name (e.g. `"hg38"`). Excluded from
    /// `checksum()`; never consulted for compatibility.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// The decorative genome name, if set.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// The canonical byte payload the genome checksum is computed over: one
    /// `"{name}\t{length}\n"` record per contig, contigs sorted by name in
    /// Unicode-codepoint (byte) order, with a trailing newline after the last
    /// record. This exact string is the cross-language contract; the Python
    /// implementation must reproduce it byte-for-byte.
    pub fn checksum_payload(&self) -> String {
        let mut ordered: Vec<&Contig> = self.contigs.iter().collect();
        ordered.sort_by(|a, b| a.name.cmp(&b.name));
        let mut buf = String::new();
        for c in ordered {
            buf.push_str(&c.name);
            buf.push('\t');
            buf.push_str(&c.length.to_string());
            buf.push('\n');
        }
        buf
    }

    /// A genome / contig-set identity: `"md5:"` + lowercase hex of the MD5 of
    /// `checksum_payload()`. The decorative name is excluded. This is the sole
    /// identity used for track mergeability (ADR 0002-adjacent; see spec).
    pub fn checksum(&self) -> String {
        let digest = md5::compute(self.checksum_payload().as_bytes());
        format!("md5:{digest:x}")
    }

    /// Build a `Genome` from a FASTA index (`.fai`): tab-separated, column 0 is
    /// the contig name and column 1 its length; remaining columns are ignored.
    ///
    /// A convenience for the hand-rolled / CLI case. Importers build the `Genome`
    /// from the source file's own header instead.
    pub fn from_fai(path: impl AsRef<Path>) -> Result<Self> {
        let text = std::fs::read_to_string(path.as_ref())
            .map_err(|e| PbzError::Store(format!("read fai: {e}")))?;
        let mut contigs = Vec::new();
        for (lineno, line) in text.lines().enumerate() {
            if line.is_empty() {
                continue;
            }
            let mut cols = line.split('\t');
            let name = cols.next().filter(|s| !s.is_empty()).ok_or_else(|| {
                PbzError::Metadata(format!("fai line {}: missing name", lineno + 1))
            })?;
            let length: u64 = cols
                .next()
                .ok_or_else(|| {
                    PbzError::Metadata(format!("fai line {}: missing length", lineno + 1))
                })?
                .parse()
                .map_err(|e| {
                    PbzError::Metadata(format!("fai line {}: bad length: {e}", lineno + 1))
                })?;
            contigs.push(Contig {
                name: name.to_owned(),
                length,
            });
        }
        Genome::new(contigs)
    }
    /// Resolve a [`RegionQuery`] against this genome.
    ///
    /// - The contig name is looked up; an unknown name returns
    ///   [`PbzError::ContigNotFound`].
    /// - `start` defaults to 0; `end` defaults to the contig's length.
    /// - `end` is clamped to the contig length (so `"chr1:0-99999999"` on a
    ///   1000-bp contig becomes `[0, 1000)`).
    /// - An empty range (after clamping) returns [`PbzError::InvalidRegion`].
    pub fn resolve(&self, query: &RegionQuery) -> Result<Region> {
        let id = self
            .id(&query.contig)
            .ok_or_else(|| PbzError::ContigNotFound {
                contig: query.contig.clone(),
                available: self.contigs.iter().map(|c| c.name.clone()).collect(),
            })?;
        let length = self.contigs[id.as_usize()].length;
        let start = query.start.unwrap_or(0);
        let end = query.end.unwrap_or(length).min(length);
        if start >= end {
            return Err(PbzError::InvalidRegion {
                message: format!(
                    "resolved range is empty on {} (length {}): start={}, end={}",
                    query.contig, length, start, end
                ),
            });
        }
        Ok(Region {
            contig: id,
            start,
            end,
        })
    }
}

/// A half-open `[start, end)` range on a [`ContigId`].
///
/// This is the canonical concrete region type that flows through reader
/// implementations. User-supplied strings are first parsed into a
/// [`crate::RegionQuery`], then resolved via [`Genome::resolve`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Region {
    pub contig: ContigId,
    pub start: u64,
    pub end: u64,
}

impl Region {
    pub fn len(&self) -> usize {
        (self.end - self.start) as usize
    }

    pub fn is_empty(&self) -> bool {
        self.end == self.start
    }
}

impl std::fmt::Display for Region {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}-{}", self.contig, self.start, self.end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offsets_is_prefix_sum() {
        let g = Genome::new(vec![
            Contig {
                name: "chr1".into(),
                length: 1000,
            },
            Contig {
                name: "chr2".into(),
                length: 500,
            },
            Contig {
                name: "chr3".into(),
                length: 250,
            },
        ])
        .unwrap();
        assert_eq!(g.offsets(), vec![0, 1000, 1500, 1750]);
    }

    #[test]
    fn checksum_payload_is_canonical() {
        // Deliberately unsorted input; payload must sort by name (codepoint order).
        let g = Genome::new(vec![
            Contig {
                name: "chr2".into(),
                length: 500,
            },
            Contig {
                name: "chr1".into(),
                length: 1000,
            },
        ])
        .unwrap();
        assert_eq!(g.checksum_payload(), "chr1\t1000\nchr2\t500\n");
    }

    #[test]
    fn checksum_excludes_name_and_is_md5_hex() {
        let a = Genome::new(vec![Contig {
            name: "chr1".into(),
            length: 10,
        }])
        .unwrap();
        let b = a.clone().with_name("hg38");
        // decorative name must not change the checksum
        assert_eq!(a.checksum(), b.checksum());
        let c = a.checksum();
        assert!(c.starts_with("md5:"));
        assert_eq!(c.len(), "md5:".len() + 32);
        assert!(
            c["md5:".len()..]
                .chars()
                .all(|ch| ch.is_ascii_hexdigit() && !ch.is_ascii_uppercase())
        );
    }

    #[test]
    fn from_fai_reads_name_and_length() {
        use std::io::Write;
        let dir = tempfile::TempDir::new().unwrap();
        let p = dir.path().join("ref.fa.fai");
        // name \t length \t offset \t linebases \t linewidth
        let mut f = std::fs::File::create(&p).unwrap();
        writeln!(f, "chr1\t1000\t6\t60\t61").unwrap();
        writeln!(f, "chr2\t500\t1024\t60\t61").unwrap();
        drop(f);

        let g = Genome::from_fai(&p).unwrap();
        assert_eq!(g.len(), 2);
        assert_eq!(
            g.contigs()[0],
            Contig {
                name: "chr1".into(),
                length: 1000
            }
        );
        assert_eq!(
            g.contigs()[1],
            Contig {
                name: "chr2".into(),
                length: 500
            }
        );
    }
}
