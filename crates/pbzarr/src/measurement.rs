//! The read handle that `view` and `stat` use for one track.

use crate::dataset::{Dataset, GenomicAxis, Reference, Role, Support, VariableInfo};
use crate::io::{Dtype, Numeric};
use crate::{PbzError, Result};
use ndarray::ArrayD;
use std::ops::Range;

/// One readable track: a format-0.5 root data variable on the
/// base-resolution `position` axis, with its metadata resolved once at open.
pub struct Measurement<'a> {
    dataset: &'a Dataset,
    name: String,
    dims: Vec<String>,
    reference: &'a Reference,
    dtype: Dtype,
    columns: u64,
    chunk: usize,
    /// Decode unit as (positions, columns): the subchunk when sharded.
    inner: (u64, u64),
}

fn unsupported(info: &VariableInfo, reason: &str) -> PbzError {
    PbzError::UnsupportedEncoding(format!(
        "track {:?} with dimensions ({}): {reason}",
        info.name,
        info.dims.join(", ")
    ))
}

/// A float `_FillValue` of NaN declares the missing values these commands
/// already skip.
fn missing_is_nan(info: &VariableInfo) -> bool {
    matches!(
        info.metadata["data_type"].as_str(),
        Some("float32" | "float64")
    ) && crate::dataset::float_fill(info)
        .ok()
        .flatten()
        .is_some_and(f64::is_nan)
}

impl<'a> Measurement<'a> {
    /// Resolve one root data variable into a read handle, rejecting the shapes
    /// and encodings these commands cannot render without changing meaning.
    pub fn root(dataset: &'a Dataset, name: &str) -> Result<Self> {
        let info = dataset
            .variables()
            .get(name)
            .ok_or_else(|| PbzError::Metadata(format!("unknown track: {name}")))?;
        if info.role != Role::Measurement {
            return Err(PbzError::Metadata(format!(
                "{name:?} is a coordinate, not a track"
            )));
        }
        if let Support::Unsupported { reason } = &info.logical_support {
            return Err(PbzError::UnsupportedEncoding(format!("{name}: {reason}")));
        }
        match info.dims.first().map(String::as_str) {
            Some("position") => {}
            Some(dim) if GenomicAxis::from_dimension(dim)?.is_some() => {
                return Err(unsupported(info, "binned genomic axes are unsupported"));
            }
            _ => return Err(unsupported(info, "a leading position axis is required")),
        }
        if info.dims.len() > 2 {
            return Err(unsupported(
                info,
                "at most one axis beside position is supported",
            ));
        }
        if !info.metadata["attributes"]["_FillValue"].is_null() && !missing_is_nan(info) {
            return Err(unsupported(
                info,
                "declared missing values other than NaN are unsupported",
            ));
        }
        let stored = info.metadata["data_type"].as_str().unwrap_or("");
        let dtype = Dtype::from_str(stored).map_err(|_| PbzError::InvalidDtype {
            dtype: format!("{name} is {stored}, which these commands cannot render"),
        })?;
        if info.metadata["chunk_grid"]["name"] != "regular" {
            return Err(unsupported(info, "a regular chunk grid is required"));
        }
        let chunk = info.metadata["chunk_grid"]["configuration"]["chunk_shape"][0]
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .filter(|&n| n > 0)
            .ok_or_else(|| PbzError::Metadata(format!("{name}: invalid position chunk shape")))?;
        dataset.check_scope(name)?;
        let shape = dataset.inner_chunk_shape(name)?;
        let inner = (shape[0], shape.get(1).copied().unwrap_or(1));
        let reference = dataset.reference()?;
        Ok(Self {
            dataset,
            name: name.to_owned(),
            dims: info.dims.clone(),
            reference,
            dtype,
            columns: info.shape.get(1).copied().unwrap_or(1),
            chunk,
            inner,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn reference(&self) -> &'a Reference {
        self.reference
    }

    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Axis names in stored order, for messages about unsupported shapes.
    pub fn dimensions(&self) -> Vec<String> {
        self.dims.clone()
    }

    pub fn columns_count(&self) -> Result<usize> {
        usize::try_from(self.columns).map_err(|e| PbzError::Metadata(format!("{}: {e}", self.name)))
    }

    /// Position chunk extent, which `stat` uses to plan chunk-aligned batches.
    pub fn chunk_size(&self) -> Result<usize> {
        Ok(self.chunk)
    }

    /// Decode unit as (positions, columns), which `view` and `stat` plan reads around.
    pub fn inner_chunk_shape(&self) -> Result<(u64, u64)> {
        Ok(self.inner)
    }

    pub fn column_labels(&self) -> Result<Vec<String>> {
        let name = &self.name;
        let dim = self
            .dims
            .get(1)
            .ok_or_else(|| PbzError::Metadata(format!("track {name:?} has no column axis")))?;
        #[allow(clippy::single_range_in_vec_init)]
        let labels = self
            .dataset
            .read_raw::<String>(dim, &[0..self.columns])
            .map_err(|e| {
                PbzError::Metadata(format!("labels for {dim:?} of track {name:?}: {e}"))
            })?;
        Ok(labels.into_raw_vec_and_offset().0)
    }

    /// Read a half-open flat position range at full column width.
    pub fn read_flat<T: Numeric>(&self, position: Range<u64>) -> Result<ArrayD<T>> {
        if self.dims.len() == 1 {
            #[allow(clippy::single_range_in_vec_init)]
            self.dataset.read_checked(&self.name, &[position])
        } else {
            self.dataset
                .read_checked(&self.name, &[position, 0..self.columns])
        }
    }

    /// Read a flat position range for a column subrange of a rank-2 track.
    pub fn read_flat_columns<T: Numeric>(
        &self,
        position: Range<u64>,
        columns: Range<u64>,
    ) -> Result<ArrayD<T>> {
        self.dataset.read_checked(&self.name, &[position, columns])
    }
}
