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
        Ok(Self { contigs, by_name })
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
