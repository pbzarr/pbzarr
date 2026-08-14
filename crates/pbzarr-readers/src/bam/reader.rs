//! `MultiValueReader` over one BAM/CRAM alignment file: `read_into` walks
//! every overlapping record through `walk_record` into scratch
//! `Accumulators` (depth-only or full composition, per [`ImportMode`]), then
//! bulk-copies each field into its sink.
//!
//! `MultiValueReader` requires `Sync`, but a `Backend` and its decode scratch
//! (`Accumulators`, `MateBuffer`) are only ever driven by one worker thread at
//! a time — each fork gets its own. The `Mutex` here exists purely to satisfy
//! the trait bound, mirroring `BedMultiReader`'s bgzf reader.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use pbzarr::genome::Genome;
use pbzarr::io::{ColumnSinkMut, Dtype, MultiValueReader, ReaderError, Result};

use super::DepthFilter;
use super::backend::Backend;
use super::mate::MateBuffer;
use super::walk::{Accumulators, ImportMode, walk_record};

/// State shared, read-only, across every fork of one `BamReader`.
struct Shared {
    path: PathBuf,
    reference: Option<PathBuf>,
    genome: Genome,
    filter: DepthFilter,
    mode: ImportMode,
    dtypes: Vec<Dtype>,
}

/// Per-fork decode scratch: the open file handle plus the walk's working
/// buffers, reused window to window.
struct Scratch {
    backend: Backend,
    acc: Accumulators,
    mates: MateBuffer,
}

/// [`MultiValueReader`] over one BAM/CRAM alignment file, driving the depth
/// or composition walk (per [`ImportMode`]) into per-fork scratch buffers.
pub struct BamReader {
    shared: Arc<Shared>,
    scratch: Mutex<Scratch>,
}

impl BamReader {
    /// Open `path` (BAM or CRAM, by content magic, falling back to the
    /// extension) for `mode`/`filter`. `reference`
    /// is required for CRAM and ignored for BAM (see `Backend::open`).
    pub fn open(
        path: &Path,
        reference: Option<&Path>,
        mode: ImportMode,
        filter: DepthFilter,
    ) -> Result<Self> {
        let (backend, genome) = Backend::open(path, reference)?;
        let n_fields = mode.n_fields();
        Ok(Self {
            shared: Arc::new(Shared {
                path: path.to_path_buf(),
                reference: reference.map(Path::to_path_buf),
                genome,
                filter,
                mode,
                dtypes: vec![Dtype::I32; n_fields],
            }),
            scratch: Mutex::new(Scratch {
                backend,
                acc: Accumulators::new(n_fields, 0),
                mates: MateBuffer::default(),
            }),
        })
    }
}

impl MultiValueReader for BamReader {
    fn contigs(&self) -> &Genome {
        &self.shared.genome
    }

    fn columns(&self) -> &[Dtype] {
        &self.shared.dtypes
    }

    fn read_into(
        &self,
        contig_name: &str,
        start: u64,
        end: u64,
        sinks: &mut [ColumnSinkMut<'_>],
    ) -> Result<()> {
        if end <= start {
            return Ok(());
        }
        let window_len = (end - start) as usize;
        // Scratch is per-window and fully reset below, so a poisoned lock
        // (from a panic mid-window on another call) is safe to reuse as-is.
        let mut scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        let Scratch {
            backend,
            acc,
            mates,
        } = &mut *scratch;

        acc.reset(window_len);
        // Overlap claims never carry across independent walk windows.
        mates.clear();

        let mode = self.shared.mode;
        let filter = &self.shared.filter;
        backend.fetch(contig_name, start, end, &mut |read| {
            walk_record(read, start, end, filter, mode, acc, mates);
            Ok(())
        })?;

        // Fail loudly on a sink/field mismatch rather than silently truncating via `zip`.
        if sinks.len() != acc.fields.len() {
            return Err(ReaderError::Other(anyhow::anyhow!(
                "bam read_into: {} sinks but {} accumulator fields",
                sinks.len(),
                acc.fields.len()
            )));
        }
        for (sink, field) in sinks.iter_mut().zip(acc.fields.iter()) {
            sink.assign_i32(field)?;
        }
        Ok(())
    }

    fn may_have_data(&self, contig_name: &str, start: u64, end: u64) -> bool {
        // Same poisoned-lock recovery as `read_into`: this fn only returns a
        // bool, so there's no error path to propagate a poisoned mutex into.
        let mut scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        // A probe failure (e.g. a transient query error) shouldn't silently
        // skip a task; fall back to "assume data present" so the pipeline
        // still reads (and surfaces the real error there).
        scratch
            .backend
            .has_records(contig_name, start, end)
            .unwrap_or(true)
    }

    fn fork(&self) -> Result<Self> {
        let (backend, _genome) =
            Backend::open(&self.shared.path, self.shared.reference.as_deref())?;
        let n_fields = self.shared.mode.n_fields();
        Ok(Self {
            shared: Arc::clone(&self.shared),
            scratch: Mutex::new(Scratch {
                backend,
                acc: Accumulators::new(n_fields, 0),
                mates: MateBuffer::default(),
            }),
        })
    }
}
