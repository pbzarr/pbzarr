use crate::genome::Genome;
use crate::io::column::ColumnSinkMut;
use crate::io::dtype::{Dtype, Numeric};
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

    /// Cheap pre-check: could this reader have any data in `[start, end)` on
    /// `contig_name`? Defaults to `true` (always read). Sparse-source readers
    /// (e.g. a BAM/CRAM index) override this so the pipeline can skip a whole
    /// task's read and write when no reader reports coverage.
    fn may_have_data(&self, _contig_name: &str, _start: u64, _end: u64) -> bool {
        true
    }

    /// Produce a worker-local handle for use on a single thread.
    /// Shared state (index, header) is reused via `Arc`; per-thread
    /// state (file handle, decode buffers) is freshly allocated.
    fn fork(&self) -> Result<Self>
    where
        Self: Sized;
}

/// Multi-column analogue of [`ValueReader`]: one source fans out to several
/// tracks of possibly different dtypes in a single decode pass. Each element of
/// `columns()` is one target track's dtype, in track order; `read_into` fills
/// one [`ColumnSinkMut`] per column for the same `[start, end)` window.
pub trait MultiValueReader: Send + Sync {
    /// Contigs present in the source file, with lengths.
    fn contigs(&self) -> &Genome;

    /// Dtype of each target column/track, in order. `sinks` in `read_into`
    /// matches this length and order.
    fn columns(&self) -> &[Dtype];

    /// Fill each sink with values for `[start, end)` on the contig named
    /// `contig_name`. `sinks[i]` is already sliced to the window; the reader
    /// indexes it from 0. Positions the source does not cover are left as the
    /// caller's fill.
    fn read_into(
        &self,
        contig_name: &str,
        start: u64,
        end: u64,
        sinks: &mut [ColumnSinkMut<'_>],
    ) -> Result<()>;

    /// Cheap pre-check: could this reader have any data in `[start, end)` on
    /// `contig_name`? Defaults to `true` (always read). See
    /// `ValueReader::may_have_data`.
    fn may_have_data(&self, _contig_name: &str, _start: u64, _end: u64) -> bool {
        true
    }

    /// Worker-local handle, sharing index/header via `Arc` (see `ValueReader::fork`).
    fn fork(&self) -> Result<Self>
    where
        Self: Sized;
}
