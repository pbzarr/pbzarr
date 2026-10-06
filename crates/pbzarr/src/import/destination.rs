//! Private destinations for the existing import engine.
use crate::io::{Dtype, Numeric};
use crate::{Genome, Result, Track};
use ndarray::ArrayD;
use std::{ops::Range, sync::Arc};
use zarrs::array::Array;
use zarrs::storage::ReadableWritableListableStorageTraits;

pub(crate) enum Destination<'a> {
    Legacy(&'a Track),
    /// A cached root-array handle. Only layout-independent Track methods are
    /// used; labels are held here because root coordinates are not track-local.
    Root {
        track: Box<Track>,
        labels: Option<Vec<String>>,
        /// First destination column on the track's column axis. Nonzero when
        /// the run appends to an existing column axis.
        column_offset: u64,
    },
}
impl Destination<'_> {
    fn track(&self) -> &Track {
        match self {
            Self::Legacy(track) => track,
            Self::Root { track, .. } => track,
        }
    }
    pub(crate) fn name(&self) -> &str {
        self.track().name()
    }
    pub(crate) fn genome(&self) -> &Arc<Genome> {
        self.track().genome()
    }
    pub(crate) fn dtype(&self) -> Dtype {
        self.track().dtype()
    }
    pub(crate) fn column_offset(&self) -> u64 {
        match self {
            Self::Legacy(_) => 0,
            Self::Root { column_offset, .. } => *column_offset,
        }
    }
    pub(crate) fn rank(&self) -> usize {
        self.track().rank()
    }
    pub(crate) fn total_len(&self) -> u64 {
        self.track().total_len()
    }
    pub(crate) fn columns_count(&self) -> Result<usize> {
        self.track().columns_count()
    }
    pub(crate) fn column_labels(&self) -> Result<Vec<String>> {
        match self {
            Self::Legacy(track) => track.column_labels(),
            Self::Root { labels, .. } => labels.clone().ok_or_else(|| {
                crate::PbzError::Metadata("rank-one destination has no column labels".into())
            }),
        }
    }
    pub(crate) fn values_array(
        &self,
    ) -> Result<Arc<Array<dyn ReadableWritableListableStorageTraits>>> {
        self.track().values_array()
    }
    pub(crate) fn write_unit_shape(&self) -> Result<Vec<usize>> {
        self.track().write_unit_shape()
    }
    pub(crate) fn write_flat<T: Numeric>(
        &self,
        start: u64,
        end: u64,
        values: ArrayD<T>,
    ) -> Result<()> {
        self.track().write_flat(start, end, values)
    }
    pub(crate) fn write_flat_columns<T: Numeric>(
        &self,
        position: Range<u64>,
        columns: Range<u64>,
        values: ArrayD<T>,
    ) -> Result<()> {
        self.track().write_flat_columns(position, columns, values)
    }
}
