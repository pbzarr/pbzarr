use color_eyre::Result;
use ndarray::ArrayView1;
use std::io::Write;
use std::ops::Range;

pub struct BedWriter<W: Write> {
    inner: W,
    pending: Option<PendingRun>,
}

struct PendingRun {
    contig: String,
    start: u64,
    end: u64,
}

impl<W: Write> BedWriter<W> {
    pub fn new(inner: W) -> Self {
        Self {
            inner,
            pending: None,
        }
    }

    pub fn emit(
        &mut self,
        contig: &str,
        chunk_start: u64,
        pos_slice: Range<u64>,
        chunk: ArrayView1<bool>,
    ) -> Result<()> {
        let chunk_len = chunk.shape()[0] as u64;
        let local_start = pos_slice.start.saturating_sub(chunk_start);
        let local_end = (pos_slice.end - chunk_start).min(chunk_len);
        for local in local_start..local_end {
            let pos = chunk_start + local;
            let v = chunk[local as usize];
            if v {
                let extend = match &self.pending {
                    Some(p) => p.contig == contig && p.end == pos,
                    None => false,
                };
                if extend {
                    self.pending.as_mut().unwrap().end = pos + 1;
                } else {
                    self.flush_pending()?;
                    self.pending = Some(PendingRun {
                        contig: contig.to_string(),
                        start: pos,
                        end: pos + 1,
                    });
                }
            } else {
                self.flush_pending()?;
            }
        }
        Ok(())
    }

    pub fn finish(&mut self) -> Result<()> {
        self.flush_pending()?;
        self.inner.flush()?;
        Ok(())
    }

    fn flush_pending(&mut self) -> Result<()> {
        if let Some(p) = self.pending.take() {
            writeln!(self.inner, "{}\t{}\t{}", p.contig, p.start, p.end)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn emits_only_true_runs() {
        let mut buf = Vec::new();
        {
            let mut w = BedWriter::new(&mut buf);
            let chunk =
                Array1::<bool>::from_vec(vec![false, true, true, false, true, true, true, false]);
            w.emit("chr1", 0, 0..8, chunk.view()).unwrap();
            w.finish().unwrap();
        }
        let s = String::from_utf8(buf).unwrap();
        assert_eq!(s, "chr1\t1\t3\nchr1\t4\t7\n");
    }

    #[test]
    fn run_continues_across_chunks() {
        let mut buf = Vec::new();
        {
            let mut w = BedWriter::new(&mut buf);
            w.emit(
                "chr1",
                0,
                0..2,
                Array1::<bool>::from_vec(vec![true, true]).view(),
            )
            .unwrap();
            w.emit(
                "chr1",
                2,
                2..4,
                Array1::<bool>::from_vec(vec![true, false]).view(),
            )
            .unwrap();
            w.finish().unwrap();
        }
        let s = String::from_utf8(buf).unwrap();
        assert_eq!(s, "chr1\t0\t3\n");
    }

    #[test]
    fn flushes_on_contig_change() {
        let mut buf = Vec::new();
        {
            let mut w = BedWriter::new(&mut buf);
            w.emit(
                "chr1",
                0,
                0..2,
                Array1::<bool>::from_vec(vec![true, true]).view(),
            )
            .unwrap();
            w.emit(
                "chr2",
                0,
                0..2,
                Array1::<bool>::from_vec(vec![true, true]).view(),
            )
            .unwrap();
            w.finish().unwrap();
        }
        let s = String::from_utf8(buf).unwrap();
        assert_eq!(s, "chr1\t0\t2\nchr2\t0\t2\n");
    }
}
