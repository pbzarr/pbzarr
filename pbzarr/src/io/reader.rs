use crate::genome::Genome;
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

    /// Fill `dst` with values for `[start, end)` on the contig named `contig_name`.
    ///
    /// The reader resolves `contig_name` in its own genome. `dst` has shape
    /// `((end - start) as usize, self.n_fields())`. The caller pre-fills `dst`
    /// with the desired fill value; the reader only overwrites positions where
    /// the source file has data.
    ///
    /// Names — not `ContigId`s — cross this boundary because `ContigId` is
    /// namespaced to its owning `Genome`. The caller's genome (e.g., a pbz
    /// store) and the reader's source file are independent.
    fn read_into(
        &self,
        contig_name: &str,
        start: u64,
        end: u64,
        dst: ArrayViewMut2<'_, Self::Item>,
    ) -> Result<()>;

    /// Produce a worker-local handle for use on a single thread.
    /// Shared state (index, header) is reused via `Arc`; per-thread
    /// state (file handle, decode buffers) is freshly allocated.
    fn fork(&self) -> Result<Self>
    where
        Self: Sized;
}
