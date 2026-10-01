use crate::{Contig, PbzError, RegionQuery, Result};
use std::{collections::BTreeMap, num::NonZeroU64, ops::Range};

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

/// Ordered reference geometry whose prefix sums fit signed 64-bit indexing.
#[derive(Debug, Clone)]
pub struct Reference {
    contigs: Vec<Contig>,
    by_name: BTreeMap<String, usize>,
    offsets: Vec<u64>,
}

impl Reference {
    pub fn new(contigs: Vec<Contig>) -> Result<Self> {
        let mut by_name = BTreeMap::new();
        let mut offsets = vec![0_u64];
        for (index, contig) in contigs.iter().enumerate() {
            if contig.name.is_empty() || by_name.insert(contig.name.clone(), index).is_some() {
                return Err(PbzError::Metadata(
                    "reference names must be nonempty and unique".into(),
                ));
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

    pub fn contigs(&self) -> &[Contig] {
        &self.contigs
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
        let index = *self
            .by_name
            .get(&query.contig)
            .ok_or_else(|| PbzError::ContigNotFound {
                contig: query.contig.clone(),
                available: self.contigs.iter().map(|c| c.name.clone()).collect(),
            })?;
        let length = self.contigs[index].length;
        let start = query.start.unwrap_or(0);
        let stop = query.end.unwrap_or(length);
        if start >= stop || stop > length {
            return Err(PbzError::InvalidRegion {
                message: format!(
                    "{}:[{start},{stop}) is empty or outside length {length}",
                    query.contig
                ),
            });
        }
        let base = self.offsets_for(axis)[index];
        Ok(match axis {
            GenomicAxis::Position => base + start..base + stop,
            GenomicAxis::Bin(width) => {
                base + start / width.get()..base + stop.div_ceil(width.get())
            }
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
