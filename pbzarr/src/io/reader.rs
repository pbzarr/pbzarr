use crate::genome::{Genome, Region};
use crate::io::dtype::Numeric;
use crate::io::error::Result;
use ndarray::ArrayViewMut2;
pub trait ValueReader: Send + Sync {
    /// The numeric type this reader produces.
    type Item: Numeric;

    /// Contigs present in the source file, with lengths.
    fn contigs(&self) -> &Genome;

    /// Number of value columns per record. Determines the trailing
    /// axis of the buffer passed to `read_into`. Scalar tracks return 1.
    fn n_fields(&self) -> usize;

    /// Fill `dst` with values for `region`.
    ///
    /// `dst` has shape `(region.len(), self.n_fields())`. The caller
    /// pre-fills `dst` with the desired fill value; the reader only
    /// overwrites positions where the source file has data.
    fn read_into(&self, region: &Region, dst: ArrayViewMut2<'_, Self::Item>) -> Result<()>;

    /// Produce a worker-local handle for use on a single thread.
    /// Shared state (index, header) is reused via `Arc`; per-thread
    /// state (file handle, decode buffers) is freshly allocated.
    fn fork(&self) -> Result<Self>
    where
        Self: Sized;
}
