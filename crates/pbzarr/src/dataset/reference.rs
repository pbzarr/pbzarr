use crate::{PbzError, RegionQuery, Result};
use hashbrown::HashMap;
use std::{num::NonZeroU64, ops::Range, path::Path};

/// Index of a contig in a [`Reference`].
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

/// A contig (chromosome or sequence) with a name and length.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Contig {
    pub name: String,
    pub length: u64,
}

/// A half-open `[start, end)` range on one contig of a [`Reference`].
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

/// A base or contig-local fixed-width genomic axis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenomicAxis {
    Position,
    Bin(NonZeroU64),
}

/// The actual half-open interval represented by one genomic array row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinInterval {
    pub contig: String,
    pub start: u64,
    pub stop: u64,
}

/// Ordered contig list whose prefix sums fit signed 64-bit indexing.
#[derive(Debug, Clone)]
pub struct Reference {
    contigs: Vec<Contig>,
    by_name: HashMap<String, ContigId>,
    offsets: Vec<u64>,
}

impl Reference {
    /// Errors on an empty or duplicate name, more than `u32::MAX` contigs, or
    /// a total length past `i64::MAX`.
    pub fn new(contigs: Vec<Contig>) -> Result<Self> {
        if contigs.len() > u32::MAX as usize {
            return Err(PbzError::Metadata(format!(
                "too many contigs: {} (max {})",
                contigs.len(),
                u32::MAX
            )));
        }
        let mut by_name = HashMap::with_capacity(contigs.len());
        let mut offsets = Vec::with_capacity(contigs.len() + 1);
        offsets.push(0_u64);
        for (index, contig) in contigs.iter().enumerate() {
            if contig.name.is_empty() {
                return Err(PbzError::Metadata(format!(
                    "contig #{index} has an empty name"
                )));
            }
            if by_name
                .insert(contig.name.clone(), ContigId(index as u32))
                .is_some()
            {
                return Err(PbzError::Metadata(format!(
                    "duplicate contig name: {:?}",
                    contig.name
                )));
            }
            let end = offsets
                .last()
                .unwrap()
                .checked_add(contig.length)
                .filter(|&end| end <= i64::MAX as u64)
                .ok_or_else(|| {
                    PbzError::Metadata("reference extent exceeds signed 64-bit indexing".into())
                })?;
            offsets.push(end);
        }
        Ok(Self {
            contigs,
            by_name,
            offsets,
        })
    }

    /// Read a FASTA index (`.fai`): column 0 is the contig name and column 1
    /// its length. Other columns are ignored.
    pub fn from_fai(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|e| {
            std::io::Error::new(e.kind(), format!("read fai {}: {e}", path.display()))
        })?;
        let mut contigs = Vec::new();
        for (index, line) in text.lines().enumerate() {
            if line.is_empty() {
                continue;
            }
            let lineno = index + 1;
            let mut cols = line.split('\t');
            let name = cols
                .next()
                .filter(|s| !s.is_empty())
                .ok_or_else(|| PbzError::Metadata(format!("fai line {lineno}: missing name")))?;
            let length: u64 = cols
                .next()
                .ok_or_else(|| PbzError::Metadata(format!("fai line {lineno}: missing length")))?
                .parse()
                .map_err(|e| PbzError::Metadata(format!("fai line {lineno}: bad length: {e}")))?;
            contigs.push(Contig {
                name: name.to_owned(),
                length,
            });
        }
        Self::new(contigs)
    }

    pub fn contigs(&self) -> &[Contig] {
        &self.contigs
    }

    pub fn len(&self) -> usize {
        self.contigs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.contigs.is_empty()
    }

    pub fn get(&self, id: ContigId) -> Option<&Contig> {
        self.contigs.get(id.as_usize())
    }

    pub fn id(&self, name: &str) -> Option<ContigId> {
        self.by_name.get(name).copied()
    }

    pub fn iter(&self) -> impl Iterator<Item = (ContigId, &Contig)> {
        self.contigs
            .iter()
            .enumerate()
            .map(|(i, c)| (ContigId(i as u32), c))
    }

    /// Flat start of each contig on the position axis, then the total length.
    pub fn offsets(&self) -> &[u64] {
        &self.offsets
    }

    pub(crate) fn total_len(&self) -> u64 {
        *self.offsets.last().unwrap()
    }

    /// Parse a region string. A string that is a whole contig name is that
    /// contig, before any split on `':'`.
    pub fn parse_query(&self, text: &str) -> Result<RegionQuery> {
        RegionQuery::parse_with(text, |name| self.id(name).is_some())
    }

    /// Resolve `query` to a nonempty range inside one contig. A missing start
    /// is 0 and a missing end is the contig length. An end past the contig
    /// length is an error.
    pub fn region(&self, query: &RegionQuery) -> Result<Region> {
        let id = self
            .id(&query.contig)
            .ok_or_else(|| PbzError::ContigNotFound {
                contig: query.contig.clone(),
                available: self.contigs.iter().map(|c| c.name.clone()).collect(),
            })?;
        let length = self.contigs[id.as_usize()].length;
        let start = query.start.unwrap_or(0);
        let end = query.end.unwrap_or(length);
        if start >= end || end > length {
            return Err(PbzError::InvalidRegion {
                message: format!(
                    "{}:[{start},{end}) is empty or outside length {length}",
                    query.contig
                ),
            });
        }
        Ok(Region {
            contig: id,
            start,
            end,
        })
    }

    fn offsets_for(&self, axis: &GenomicAxis) -> Vec<u64> {
        match axis {
            GenomicAxis::Position => self.offsets.clone(),
            GenomicAxis::Bin(width) => {
                let width = width.get();
                let mut offsets = vec![0];
                for contig in &self.contigs {
                    // Each count is <= the already checked base count.
                    offsets.push(offsets.last().unwrap() + contig.length.div_ceil(width));
                }
                offsets
            }
        }
    }

    pub fn extent(&self, axis: &GenomicAxis) -> Result<u64> {
        Ok(*self.offsets_for(axis).last().unwrap())
    }

    pub fn rows(&self, axis: &GenomicAxis, query: &RegionQuery) -> Result<Range<u64>> {
        let Region { contig, start, end } = self.region(query)?;
        let base = self.offsets_for(axis)[contig.as_usize()];
        Ok(match axis {
            GenomicAxis::Position => base + start..base + end,
            GenomicAxis::Bin(width) => base + start / width.get()..base + end.div_ceil(width.get()),
        })
    }

    pub fn interval(&self, axis: &GenomicAxis, row: u64) -> Result<BinInterval> {
        let offsets = self.offsets_for(axis);
        if row >= *offsets.last().unwrap() {
            return Err(PbzError::InvalidRegion {
                message: format!("row {row} is outside genomic axis"),
            });
        }
        let index = offsets.partition_point(|&offset| offset <= row) - 1;
        let width = match axis {
            GenomicAxis::Position => 1,
            GenomicAxis::Bin(width) => width.get(),
        };
        let start = (row - offsets[index]) * width;
        Ok(BinInterval {
            contig: self.contigs[index].name.clone(),
            start,
            stop: start.saturating_add(width).min(self.contigs[index].length),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contig(name: &str, length: u64) -> Contig {
        Contig {
            name: name.into(),
            length,
        }
    }

    #[test]
    fn offsets_is_prefix_sum() {
        let reference = Reference::new(vec![
            contig("chr1", 1000),
            contig("chr2", 500),
            contig("chr3", 250),
        ])
        .unwrap();
        assert_eq!(reference.offsets(), [0, 1000, 1500, 1750]);
    }

    #[test]
    fn from_fai_reads_name_and_length() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("ref.fa.fai");
        std::fs::write(&path, "chr1\t1000\t6\t60\t61\nchr2\t500\t1024\t60\t61\n").unwrap();
        let reference = Reference::from_fai(&path).unwrap();
        assert_eq!(
            reference.contigs(),
            [contig("chr1", 1000), contig("chr2", 500)]
        );
    }
}
