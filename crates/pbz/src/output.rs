//! Output plumbing that `pbz view` and `pbz stat` share.

use std::fs::File;
use std::io::{self, BufWriter, Stdout, Write};
use std::num::NonZeroUsize;
use std::path::Path;

use anyhow::{Context as _, Result, bail};
use noodles_bgzf as bgzf;

/// Check `--precision` and size the global rayon pool. One pool serves zarrs
/// chunk decode, the command's own batches and BGZF compression.
pub(crate) fn prepare(precision: u8, threads: Option<NonZeroUsize>) -> Result<usize> {
    if !(1..=17).contains(&precision) {
        bail!("--precision must be between 1 and 17 significant digits");
    }
    if let Some(threads) = threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads.get())
            .build_global()
            .context("configure worker thread pool")?;
    }
    Ok(usize::from(precision))
}

/// Stdout, a BGZF file for a `.gz` path, or a plain file.
pub(crate) enum Output {
    Stdout(BufWriter<Stdout>),
    // Deflate dominates the export cost; compress blocks on a pool.
    Bgzf(bgzf::io::MultithreadedWriter<File>),
    File(BufWriter<File>),
}

impl Output {
    pub(crate) fn create(path: Option<&Path>) -> Result<Self> {
        let Some(path) = path else {
            return Ok(Self::Stdout(BufWriter::new(io::stdout())));
        };
        let file =
            File::create(path).with_context(|| format!("create output {}", path.display()))?;
        if path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("gz"))
        {
            Ok(Self::Bgzf(bgzf::io::MultithreadedWriter::new(file)))
        } else {
            Ok(Self::File(BufWriter::new(file)))
        }
    }

    /// Flush the buffers and write the BGZF end-of-file block.
    pub(crate) fn finish(self) -> io::Result<()> {
        match self {
            Self::Stdout(mut out) => out.flush(),
            Self::Bgzf(mut out) => out.finish().map(drop),
            Self::File(mut out) => out.flush(),
        }
    }
}

impl Write for Output {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            Self::Stdout(out) => out.write(buf),
            Self::Bgzf(out) => out.write(buf),
            Self::File(out) => out.write(buf),
        }
    }

    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        match self {
            Self::Stdout(out) => out.write_all(buf),
            Self::Bgzf(out) => out.write_all(buf),
            Self::File(out) => out.write_all(buf),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match self {
            Self::Stdout(out) => out.flush(),
            Self::Bgzf(out) => out.flush(),
            Self::File(out) => out.flush(),
        }
    }
}
