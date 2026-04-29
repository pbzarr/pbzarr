use color_eyre::Result;
use ndarray::ArrayView1;
use std::fmt::Display;
use std::io::Write;
use std::ops::Range;

pub struct BedGraphWriter<W: Write, T: PartialEq + Copy + Display> {
    inner: W,
    fill: T,
    include_zero: bool,
    pending: Option<PendingRun<T>>,
}

struct PendingRun<T> {
    contig: String,
    start: u64,
    end: u64,
    value: T,
}

impl<W: Write, T: PartialEq + Copy + Display> BedGraphWriter<W, T> {
    pub fn new(inner: W, fill: T, include_zero: bool) -> Self {
        Self {
            inner,
            fill,
            include_zero,
            pending: None,
        }
    }

    pub fn finish(&mut self) -> Result<()> {
        self.flush_pending()?;
        self.inner.flush()?;
        Ok(())
    }

    fn flush_pending(&mut self) -> Result<()> {
        if let Some(p) = self.pending.take()
            && (self.include_zero || p.value != self.fill)
        {
            writeln!(
                self.inner,
                "{}\t{}\t{}\t{}",
                p.contig, p.start, p.end, p.value
            )?;
        }
        Ok(())
    }
}

macro_rules! emit_for_type {
    ($name:ident, $ty:ty) => {
        impl<W: Write> BedGraphWriter<W, $ty> {
            pub fn $name(
                &mut self,
                contig: &str,
                chunk_start: u64,
                pos_slice: Range<u64>,
                chunk: ArrayView1<$ty>,
            ) -> Result<()> {
                let chunk_len = chunk.shape()[0] as u64;
                let local_start = pos_slice.start.saturating_sub(chunk_start);
                let local_end = (pos_slice.end - chunk_start).min(chunk_len);
                for local in local_start..local_end {
                    let pos = chunk_start + local;
                    let v = chunk[local as usize];
                    let extend = match &self.pending {
                        Some(p) => p.contig == contig && p.value == v && p.end == pos,
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
                            value: v,
                        });
                    }
                }
                Ok(())
            }
        }
    };
}
emit_for_type!(emit_u8, u8);
emit_for_type!(emit_u16, u16);
emit_for_type!(emit_u32, u32);
emit_for_type!(emit_i8, i8);
emit_for_type!(emit_i16, i16);
emit_for_type!(emit_i32, i32);
emit_for_type!(emit_f32, f32);
emit_for_type!(emit_f64, f64);

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn collapses_equal_runs() {
        let mut buf = Vec::new();
        {
            let mut w = BedGraphWriter::new(&mut buf, 0u32, false);
            let chunk = Array1::<u32>::from_vec(vec![5, 5, 5, 7, 7, 0, 0, 5]);
            w.emit_u32("chr1", 100, 100..108, chunk.view()).unwrap();
            w.finish().unwrap();
        }
        let s = String::from_utf8(buf).unwrap();
        assert_eq!(
            s,
            "chr1\t100\t103\t5\nchr1\t103\t105\t7\nchr1\t107\t108\t5\n"
        );
    }

    #[test]
    fn includes_zero_when_requested() {
        let mut buf = Vec::new();
        {
            let mut w = BedGraphWriter::new(&mut buf, 0u32, true);
            let chunk = Array1::<u32>::from_vec(vec![0, 0, 5, 5]);
            w.emit_u32("chr1", 0, 0..4, chunk.view()).unwrap();
            w.finish().unwrap();
        }
        let s = String::from_utf8(buf).unwrap();
        assert_eq!(s, "chr1\t0\t2\t0\nchr1\t2\t4\t5\n");
    }

    #[test]
    fn flushes_on_contig_change() {
        let mut buf = Vec::new();
        {
            let mut w = BedGraphWriter::new(&mut buf, 0u32, false);
            let a = Array1::<u32>::from_vec(vec![5, 5]);
            w.emit_u32("chr1", 0, 0..2, a.view()).unwrap();
            let b = Array1::<u32>::from_vec(vec![5, 5]);
            w.emit_u32("chr2", 0, 0..2, b.view()).unwrap();
            w.finish().unwrap();
        }
        let s = String::from_utf8(buf).unwrap();
        assert_eq!(s, "chr1\t0\t2\t5\nchr2\t0\t2\t5\n");
    }

    #[test]
    fn run_continues_across_chunk_boundaries() {
        let mut buf = Vec::new();
        {
            let mut w = BedGraphWriter::new(&mut buf, 0u32, false);
            let a = Array1::<u32>::from_vec(vec![5, 5, 5]);
            w.emit_u32("chr1", 0, 0..3, a.view()).unwrap();
            let b = Array1::<u32>::from_vec(vec![5, 5]);
            w.emit_u32("chr1", 3, 3..5, b.view()).unwrap();
            w.finish().unwrap();
        }
        let s = String::from_utf8(buf).unwrap();
        assert_eq!(s, "chr1\t0\t5\t5\n");
    }
}
