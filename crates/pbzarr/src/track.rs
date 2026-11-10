//! The array handle that the import engine writes through.

use std::sync::Arc;

use ndarray::ArrayD;
use zarrs::array::{Array, ArraySubset};
use zarrs::storage::ReadableWritableListableStorageTraits;

use crate::Result;
use crate::error::PbzError;
use crate::genome::Genome;
use crate::io::{Dtype, Numeric};

/// One root array of a format-0.5 store, addressed by flat position.
pub(crate) struct Track {
    pub(crate) name: String,
    pub(crate) genome: Arc<Genome>,
    pub(crate) dtype: Dtype,
    pub(crate) rank: usize,
    pub(crate) values: Arc<Array<dyn ReadableWritableListableStorageTraits>>,
}

impl Track {
    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    pub(crate) fn genome(&self) -> &Arc<Genome> {
        &self.genome
    }

    pub(crate) fn dtype(&self) -> Dtype {
        self.dtype
    }

    /// 1 for scalar tracks, 2 for tracks with a column axis.
    pub(crate) fn rank(&self) -> usize {
        self.rank
    }

    /// Sum of the genome's contig lengths.
    pub(crate) fn total_len(&self) -> u64 {
        self.genome.contigs().iter().map(|c| c.length).sum()
    }

    /// 1 for rank-1 tracks, else the column extent of the array.
    pub(crate) fn columns_count(&self) -> Result<usize> {
        if self.rank == 1 {
            return Ok(1);
        }
        Ok(self.values.shape()[1] as usize)
    }

    /// Shape of one independently writable outer unit: a regular chunk, or a
    /// shard when sharding is configured.
    pub(crate) fn write_unit_shape(&self) -> Result<Vec<usize>> {
        self.values
            .chunk_shape_usize(&vec![0; self.rank])
            .map_err(|e| PbzError::Store(format!("write-unit shape for {}: {e}", self.name)))
    }

    pub(crate) fn values_array(&self) -> Arc<Array<dyn ReadableWritableListableStorageTraits>> {
        Arc::clone(&self.values)
    }

    /// Write `data` into the half-open flat position range `[start, end)`,
    /// which may span contig boundaries. The import pipeline uses this to write
    /// one physical chunk at a time regardless of where contigs begin.
    pub(crate) fn write_flat<T: Numeric>(
        &self,
        start: u64,
        end: u64,
        data: ArrayD<T>,
    ) -> Result<()> {
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
        if data.ndim() != self.rank {
            return Err(PbzError::Metadata(format!(
                "rank mismatch for track {:?}: expected {} got {}",
                self.name,
                self.rank,
                data.ndim(),
            )));
        }
        if self.rank == 2 {
            return self.write_flat_columns(start..end, 0..self.columns_count()? as u64, data);
        }
        let arr = &self.values;
        if start > end || end > arr.shape()[0] {
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
        arr.store_array_subset(&subset, data)
            .map_err(|e| PbzError::Store(format!("write {}: {e}", self.name)))
    }

    /// Write one rectangular tile of a rank-2 track using flat position and
    /// column ranges. Import workers use full write units so they never race.
    pub(crate) fn write_flat_columns<T: Numeric>(
        &self,
        position: std::ops::Range<u64>,
        columns: std::ops::Range<u64>,
        data: ArrayD<T>,
    ) -> Result<()> {
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
        if self.rank != 2 || data.ndim() != 2 {
            return Err(PbzError::Metadata(format!(
                "column tile for track {:?} requires rank-2 data",
                self.name
            )));
        }
        let array = &self.values;
        let shape = array.shape();
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
        array
            .store_array_subset(&subset, data)
            .map_err(|e| PbzError::Store(format!("write {}: {e}", self.name)))
    }
}
