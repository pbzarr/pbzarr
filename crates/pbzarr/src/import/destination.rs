//! Private destinations for the existing import engine.
use crate::io::{Dtype, Numeric};
use crate::track::Track;
use crate::{Genome, PbzError, Result};
use ndarray::ArrayD;
use std::{ops::Range, sync::Arc};
use zarrs::array::Array;
use zarrs::storage::ReadableWritableListableStorageTraits;

/// A cached root-array handle. Labels are held here because root
/// coordinates are not track-local.
pub(crate) struct Destination {
    pub(crate) track: Track,
    pub(crate) labels: Option<Vec<String>>,
    /// First destination column on the track's column axis. Nonzero when
    /// the run appends to an existing column axis.
    pub(crate) column_offset: u64,
}
impl Destination {
    fn track(&self) -> &Track {
        &self.track
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
        self.column_offset
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
        self.labels
            .clone()
            .ok_or_else(|| PbzError::Metadata("rank-one destination has no column labels".into()))
    }
    pub(crate) fn values_array(&self) -> Arc<Array<dyn ReadableWritableListableStorageTraits>> {
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
