//! The root-array handle that the import engine writes through.
use crate::dataset::Reference;
use crate::io::{Dtype, Numeric};
use crate::{PbzError, Result};
use ndarray::ArrayD;
use std::{ops::Range, sync::Arc};
use zarrs::array::{Array, ArraySubset};
use zarrs::storage::ReadableWritableListableStorageTraits;

/// One root array of a store, addressed by flat position. Labels are held
/// here because root coordinates are not array-local.
pub(crate) struct Destination {
    pub(crate) name: String,
    pub(crate) reference: Arc<Reference>,
    pub(crate) dtype: Dtype,
    pub(crate) values: Arc<Array<dyn ReadableWritableListableStorageTraits>>,
    pub(crate) labels: Option<Vec<String>>,
    /// First destination column on the column axis. Nonzero when the run
    /// appends to an existing column axis.
    pub(crate) column_offset: u64,
}

impl Destination {
    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    pub(crate) fn reference(&self) -> &Arc<Reference> {
        &self.reference
    }

    pub(crate) fn dtype(&self) -> Dtype {
        self.dtype
    }

    pub(crate) fn column_offset(&self) -> u64 {
        self.column_offset
    }

    /// 1 for a position-only array, 2 for an array with a column axis.
    pub(crate) fn rank(&self) -> usize {
        self.values.shape().len()
    }

    pub(crate) fn total_len(&self) -> u64 {
        self.reference.total_len()
    }

    /// 1 for a rank-1 array, else the column extent of the array.
    pub(crate) fn columns_count(&self) -> Result<usize> {
        if self.rank() == 1 {
            return Ok(1);
        }
        Ok(self.values.shape()[1] as usize)
    }

    pub(crate) fn column_labels(&self) -> Result<Vec<String>> {
        self.labels
            .clone()
            .ok_or_else(|| PbzError::Metadata("rank-one destination has no column labels".into()))
    }

    pub(crate) fn values_array(&self) -> Arc<Array<dyn ReadableWritableListableStorageTraits>> {
        Arc::clone(&self.values)
    }

    /// Shape of one independently writable outer unit: a regular chunk, or a
    /// shard when sharding is configured.
    pub(crate) fn write_unit_shape(&self) -> Result<Vec<usize>> {
        self.values
            .chunk_shape_usize(&vec![0; self.rank()])
            .map_err(|e| PbzError::Store(format!("write-unit shape for {}: {e}", self.name)))
    }

    fn check_dtype<T: Numeric>(&self) -> Result<()> {
        if T::DTYPE != self.dtype {
            return Err(PbzError::InvalidDtype {
                dtype: format!(
                    "track {:?} is {} but caller wrote {}",
                    self.name,
                    self.dtype,
                    T::DTYPE
                ),
            });
        }
        Ok(())
    }

    /// Write `data` into the half-open flat position range `[start, end)`,
    /// which may span contig boundaries.
    pub(crate) fn write_flat<T: Numeric>(
        &self,
        start: u64,
        end: u64,
        data: ArrayD<T>,
    ) -> Result<()> {
        self.check_dtype::<T>()?;
        let rank = self.rank();
        if data.ndim() != rank {
            return Err(PbzError::Metadata(format!(
                "rank mismatch for track {:?}: expected {rank} got {}",
                self.name,
                data.ndim(),
            )));
        }
        if rank == 2 {
            return self.write_flat_columns(start..end, 0..self.columns_count()? as u64, data);
        }
        if start > end || end > self.values.shape()[0] {
            return Err(PbzError::InvalidRegion {
                message: format!("flat range {start}..{end} is outside track {:?}", self.name),
            });
        }
        if data.shape()
            != [usize::try_from(end - start)
                .map_err(|_| PbzError::Metadata("flat write length exceeds usize".into()))?]
        {
            return Err(PbzError::Metadata(format!(
                "shape mismatch for track {:?}: expected [{}], got {:?}",
                self.name,
                end - start,
                data.shape(),
            )));
        }
        #[allow(clippy::single_range_in_vec_init)]
        let subset = ArraySubset::new_with_ranges(&[start..end]);
        self.values
            .store_array_subset(&subset, data)
            .map_err(|e| PbzError::Store(format!("write {}: {e}", self.name)))
    }

    /// Write one rectangular tile of a rank-2 array using flat position and
    /// column ranges. Import workers use full write units so they never race.
    pub(crate) fn write_flat_columns<T: Numeric>(
        &self,
        position: Range<u64>,
        columns: Range<u64>,
        data: ArrayD<T>,
    ) -> Result<()> {
        self.check_dtype::<T>()?;
        if self.rank() != 2 || data.ndim() != 2 {
            return Err(PbzError::Metadata(format!(
                "column tile for track {:?} requires rank-2 data",
                self.name
            )));
        }
        let shape = self.values.shape();
        if position.start > position.end
            || columns.start > columns.end
            || position.end > shape[0]
            || columns.end > shape[1]
        {
            return Err(PbzError::InvalidRegion {
                message: format!(
                    "column tile {:?} × {:?} is outside track {:?}",
                    position, columns, self.name
                ),
            });
        }
        let rows = usize::try_from(position.end - position.start)
            .map_err(|_| PbzError::Metadata("tile row count exceeds usize".into()))?;
        let width = usize::try_from(columns.end - columns.start)
            .map_err(|_| PbzError::Metadata("tile column count exceeds usize".into()))?;
        if data.shape() != [rows, width] {
            return Err(PbzError::Metadata(format!(
                "shape mismatch for track {:?}: expected [{rows}, {width}], got {:?}",
                self.name,
                data.shape(),
            )));
        }
        let subset = ArraySubset::new_with_ranges(&[position, columns]);
        self.values
            .store_array_subset(&subset, data)
            .map_err(|e| PbzError::Store(format!("write {}: {e}", self.name)))
    }
}
