use crate::io::value_reader::{ValueChunk, ValueDtype, ValueReader};
use color_eyre::Result;
use color_eyre::eyre::{WrapErr, eyre};
use d4::ssio::D4TrackReader;
use ndarray::Array1;
use std::fs::File;
use std::path::{Path, PathBuf};

/// Naive single-sample D4 reader.
///
/// Mirrors clam's plain `D4Reader` pattern: opens a single track via
/// `D4TrackReader::from_reader` and reads dense per-base depths via the
/// streaming view API. No multisample / Bgzf support — that's deferred.
pub struct D4Reader {
    // Kept for diagnostic messages so chunk-read errors can name the offending input.
    path: PathBuf,
    contigs: Vec<(String, u64)>,
    inner: D4TrackReader<File>,
}

impl D4Reader {
    pub fn open<P: AsRef<Path>>(src: P) -> Result<Self> {
        let path = src.as_ref().to_path_buf();
        let file = File::open(&path)
            .wrap_err_with(|| format!("failed to open D4 file: {}", path.display()))?;
        let inner = D4TrackReader::from_reader(file, None).map_err(|e| {
            eyre!(
                "failed to create D4 track reader for {}: {e}",
                path.display()
            )
        })?;

        let contigs: Vec<(String, u64)> = inner
            .get_header()
            .chrom_list()
            .iter()
            .map(|c| (c.name.clone(), c.size as u64))
            .collect();

        Ok(Self {
            path,
            contigs,
            inner,
        })
    }
}

impl ValueReader for D4Reader {
    fn dtype(&self) -> ValueDtype {
        ValueDtype::U32
    }

    fn contigs(&self) -> &[(String, u64)] {
        &self.contigs
    }

    fn read_chunk(&mut self, contig: &str, start: u64, end: u64) -> Result<ValueChunk> {
        let s32 = u32::try_from(start)
            .map_err(|_| eyre!("D4 reads require start <= u32::MAX, got {start}"))?;
        let e32 =
            u32::try_from(end).map_err(|_| eyre!("D4 reads require end <= u32::MAX, got {end}"))?;
        let file = self.path.display();
        let mut view = self
            .inner
            .get_view(contig, s32, e32)
            .map_err(|e| eyre!("D4 view failed for {file}:{contig}:{start}-{end}: {e}"))?;
        let len = (end - start) as usize;
        let mut buf: Vec<u32> = Vec::with_capacity(len);
        for pos in s32..e32 {
            let (reported_pos, value) = view
                .next()
                .ok_or_else(|| {
                    eyre!("unexpected end of D4 view at {file}:{contig} position {pos}")
                })?
                .map_err(|e| eyre!("failed to read D4 value from {file}:{contig}:{pos}: {e}"))?;
            if reported_pos != pos {
                return Err(eyre!(
                    "D4 position mismatch at {file}:{contig}:{pos}: got {reported_pos}"
                ));
            }
            let depth: u32 = value.try_into().map_err(|_| {
                eyre!(
                    "D4 depth at {file}:{contig}:{pos} cannot be represented as u32 (got {value})"
                )
            })?;
            buf.push(depth);
        }
        Ok(ValueChunk::U32(Array1::from_vec(buf)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_missing_file_errors() {
        let result = D4Reader::open("/no/such/file.d4");
        assert!(result.is_err());
    }
}
