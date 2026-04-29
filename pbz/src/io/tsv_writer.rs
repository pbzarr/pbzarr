use color_eyre::Result;
use ndarray::{ArrayView1, ArrayView2};
use std::io::Write;
use std::ops::Range;

pub struct TsvWriter<W: Write> {
    inner: W,
    header_written: bool,
}

impl<W: Write> TsvWriter<W> {
    pub fn new(inner: W) -> Self {
        Self {
            inner,
            header_written: false,
        }
    }

    pub fn header(&mut self, columns: &[impl AsRef<str>]) -> Result<()> {
        if self.header_written {
            return Ok(());
        }
        write!(self.inner, "contig\tpos")?;
        for c in columns {
            write!(self.inner, "\t{}", c.as_ref())?;
        }
        writeln!(self.inner)?;
        self.header_written = true;
        Ok(())
    }

    pub fn header_scalar(&mut self) -> Result<()> {
        if self.header_written {
            return Ok(());
        }
        writeln!(self.inner, "contig\tpos\tvalue")?;
        self.header_written = true;
        Ok(())
    }

    pub fn finish(&mut self) -> Result<()> {
        self.inner.flush()?;
        Ok(())
    }
}

macro_rules! emit_2d {
    ($name:ident, $ty:ty) => {
        impl<W: Write> TsvWriter<W> {
            pub fn $name(
                &mut self,
                contig: &str,
                chunk_start: u64,
                pos_slice: Range<u64>,
                chunk: ArrayView2<$ty>,
                col_indices: &[usize],
            ) -> Result<()> {
                let chunk_len = chunk.shape()[0] as u64;
                let local_start = pos_slice.start.saturating_sub(chunk_start);
                let local_end = (pos_slice.end - chunk_start).min(chunk_len);
                for local in local_start..local_end {
                    write!(self.inner, "{}\t{}", contig, chunk_start + local)?;
                    for &c in col_indices {
                        write!(self.inner, "\t{}", chunk[(local as usize, c)])?;
                    }
                    writeln!(self.inner)?;
                }
                Ok(())
            }
        }
    };
}
emit_2d!(emit_u8, u8);
emit_2d!(emit_u16, u16);
emit_2d!(emit_u32, u32);
emit_2d!(emit_i8, i8);
emit_2d!(emit_i16, i16);
emit_2d!(emit_i32, i32);
emit_2d!(emit_f32, f32);
emit_2d!(emit_f64, f64);

macro_rules! emit_1d {
    ($name:ident, $ty:ty) => {
        impl<W: Write> TsvWriter<W> {
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
                    writeln!(
                        self.inner,
                        "{}\t{}\t{}",
                        contig,
                        chunk_start + local,
                        chunk[local as usize]
                    )?;
                }
                Ok(())
            }
        }
    };
}
emit_1d!(emit_u8_scalar, u8);
emit_1d!(emit_u16_scalar, u16);
emit_1d!(emit_u32_scalar, u32);
emit_1d!(emit_i8_scalar, i8);
emit_1d!(emit_i16_scalar, i16);
emit_1d!(emit_i32_scalar, i32);
emit_1d!(emit_f32_scalar, f32);
emit_1d!(emit_f64_scalar, f64);
emit_1d!(emit_bool_scalar, bool);

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn writes_header_and_rows() {
        let mut buf = Vec::new();
        {
            let mut w = TsvWriter::new(&mut buf);
            let cols: Vec<&str> = vec!["s0", "s1"];
            w.header(&cols).unwrap();
            let chunk = Array2::<u32>::from_shape_vec((3, 2), vec![1, 2, 3, 4, 5, 6]).unwrap();
            w.emit_u32("chr1", 100, 100..103, chunk.view(), &[0, 1])
                .unwrap();
            w.finish().unwrap();
        }
        let s = String::from_utf8(buf).unwrap();
        assert_eq!(
            s,
            "contig\tpos\ts0\ts1\nchr1\t100\t1\t2\nchr1\t101\t3\t4\nchr1\t102\t5\t6\n"
        );
    }

    #[test]
    fn column_filter_subsets() {
        let mut buf = Vec::new();
        {
            let mut w = TsvWriter::new(&mut buf);
            let cols: Vec<&str> = vec!["s1"];
            w.header(&cols).unwrap();
            let chunk = Array2::<u32>::from_shape_vec((1, 2), vec![10, 20]).unwrap();
            w.emit_u32("chr1", 0, 0..1, chunk.view(), &[1]).unwrap();
            w.finish().unwrap();
        }
        let s = String::from_utf8(buf).unwrap();
        assert_eq!(s, "contig\tpos\ts1\nchr1\t0\t20\n");
    }
}
