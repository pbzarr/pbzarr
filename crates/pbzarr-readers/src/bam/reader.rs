//! Reader over one BAM/CRAM alignment file: `read_into` walks every
//! overlapping record through `walk_record` into scratch `Accumulators`
//! (depth-only or full composition, per [`ImportMode`]), then bulk-copies each
//! field into its sink.
//!
//! A `Backend` and its decode scratch (`Accumulators`, `MateBuffer`) are only
//! ever driven by one worker thread at a time, each fork getting its own. The
//! `Mutex` around them exists because `may_have_data` probes the index
//! through `&self` while `Backend::has_records` needs `&mut`; `read_into`
//! takes `&mut self` and goes through `get_mut` instead.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use ndarray::ArrayView1;

use pbzarr::genome::Genome;
use pbzarr::io::{
    Dtype, OutputField, OutputSchema, OutputSinkMut, ReaderError, Result, ValueReader,
};

use super::DepthFilter;
use super::backend::Backend;
use super::mate::MateBuffer;
use super::walk::{Accumulators, FIELDS, ImportMode, walk_record};

/// State shared, read-only, across every fork of one `BamReader`.
struct Shared {
    path: PathBuf,
    reference: Option<PathBuf>,
    genome: Genome,
    filter: DepthFilter,
    mode: ImportMode,
    schema: OutputSchema,
}

/// Per-fork decode scratch: the open file handle plus the walk's working
/// buffers, reused window to window.
struct Scratch {
    backend: Backend,
    acc: Accumulators,
    mates: MateBuffer,
}

/// Reader over one BAM/CRAM alignment file, driving the depth or composition
/// walk (per [`ImportMode`]) into per-fork scratch buffers.
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
                schema: schema_for(mode),
            }),
            scratch: Mutex::new(Scratch {
                backend,
                acc: Accumulators::new(n_fields, 0),
                mates: MateBuffer::default(),
            }),
        })
    }

    fn fork_handle(&self) -> Result<Self> {
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

    fn probe(&self, contig_name: &str, start: u64, end: u64) -> bool {
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
}

/// The leading `FIELDS` prefix this mode writes: `["depth"]` for `Depth`, all
/// of them for `Composition`.
fn schema_for(mode: ImportMode) -> OutputSchema {
    OutputSchema::new(
        FIELDS[..mode.n_fields()]
            .iter()
            .map(|name| OutputField {
                name: Some((*name).to_owned()),
                dtype: Dtype::I32,
            })
            .collect(),
    )
}

/// Walk every record overlapping `[start, end)` into `scratch.acc`, leaving
/// one filled `window_len` vector per field of the mode's schema.
fn accumulate(
    shared: &Shared,
    scratch: &mut Scratch,
    contig_name: &str,
    start: u64,
    end: u64,
) -> Result<()> {
    let window_len = (end - start) as usize;
    let Scratch {
        backend,
        acc,
        mates,
    } = scratch;

    acc.reset(window_len);
    // Overlap claims never carry across independent walk windows.
    mates.clear();

    let mode = shared.mode;
    let filter = &shared.filter;
    backend.fetch(contig_name, start, end, &mut |read| {
        walk_record(read, start, end, filter, mode, acc, mates);
        Ok(())
    })
}

impl ValueReader for BamReader {
    fn contigs(&self) -> &Genome {
        &self.shared.genome
    }

    fn output_schema(&self) -> &OutputSchema {
        &self.shared.schema
    }

    fn read_into(
        &mut self,
        contig: &str,
        start: u64,
        end: u64,
        outputs: &mut [OutputSinkMut<'_>],
    ) -> Result<()> {
        if end <= start {
            return Ok(());
        }
        let Self { shared, scratch } = self;
        // Scratch is per-window and fully reset by `accumulate`, so a poisoned
        // lock (from a panic mid-window on another call) is safe to reuse.
        let scratch = scratch.get_mut().unwrap_or_else(|e| e.into_inner());
        accumulate(shared, scratch, contig, start, end)?;

        if outputs.len() != scratch.acc.fields.len() {
            return Err(ReaderError::SchemaMismatch {
                message: format!(
                    "bam reader produces {} field(s) but the engine handed {} sink(s)",
                    scratch.acc.fields.len(),
                    outputs.len()
                ),
            });
        }
        for (sink, field) in outputs.iter_mut().zip(scratch.acc.fields.iter()) {
            let dst = sink.as_i32_mut()?;
            if dst.len() != field.len() {
                return Err(ReaderError::SchemaMismatch {
                    message: format!(
                        "bam reader filled {} position(s) but the sink holds {}",
                        field.len(),
                        dst.len()
                    ),
                });
            }
            dst.assign(&ArrayView1::from(field.as_slice()));
        }
        Ok(())
    }

    fn may_have_data(&self, contig: &str, start: u64, end: u64) -> bool {
        self.probe(contig, start, end)
    }

    fn fork(&self) -> Result<Self> {
        self.fork_handle()
    }
}
