//! The reader trait for the import engine.
//!
//! One trait covers every source shape: a reader exposes an ordered,
//! dtype-erased [`OutputSchema`] and fills one [`OutputSinkMut`] per schema
//! field in a single decode pass.

use crate::genome::Genome;
use crate::io::dtype::Dtype;
use crate::io::error::ReaderError;
use ndarray::ArrayViewMut1;

/// One field a reader produces.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputField {
    pub name: Option<String>,
    pub dtype: Dtype,
}

/// The ordered fields a reader produces. `read_into` receives exactly one
/// sink per field, in this order.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct OutputSchema {
    pub fields: Vec<OutputField>,
}

impl OutputSchema {
    pub fn new(fields: Vec<OutputField>) -> Self {
        Self { fields }
    }

    pub fn single(name: impl Into<String>, dtype: Dtype) -> Self {
        Self {
            fields: vec![OutputField {
                name: Some(name.into()),
                dtype,
            }],
        }
    }

    pub fn len(&self) -> usize {
        self.fields.len()
    }

    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    pub fn dtypes(&self) -> Vec<Dtype> {
        self.fields.iter().map(|f| f.dtype).collect()
    }
}

macro_rules! output_sink {
    ($( $variant:ident => $ty:ty, $as_mut:ident );* $(;)?) => {
        /// A borrowing, dtype-tagged mutable view a reader fills.
        pub enum OutputSinkMut<'a> {
            $( $variant(ArrayViewMut1<'a, $ty>), )*
        }

        impl<'a> OutputSinkMut<'a> {
            pub fn dtype(&self) -> Dtype {
                match self {
                    $( OutputSinkMut::$variant(_) => Dtype::$variant, )*
                }
            }

            pub fn len(&self) -> usize {
                match self {
                    $( OutputSinkMut::$variant(v) => v.len(), )*
                }
            }

            pub fn is_empty(&self) -> bool {
                self.len() == 0
            }

            $(
                /// `SchemaMismatch` when the sink holds a different dtype.
                pub fn $as_mut(&mut self) -> Result<&mut ArrayViewMut1<'a, $ty>, ReaderError> {
                    match self {
                        OutputSinkMut::$variant(v) => Ok(v),
                        other => Err(ReaderError::SchemaMismatch {
                            message: format!(
                                "requested {} sink but the engine handed {}",
                                Dtype::$variant,
                                other.dtype()
                            ),
                        }),
                    }
                }
            )*
        }
    };
}

output_sink! {
    U8 => u8, as_u8_mut;
    U16 => u16, as_u16_mut;
    U32 => u32, as_u32_mut;
    I8 => i8, as_i8_mut;
    I16 => i16, as_i16_mut;
    I32 => i32, as_i32_mut;
    F32 => f32, as_f32_mut;
    F64 => f64, as_f64_mut;
    Bool => bool, as_bool_mut;
}

/// Receives one window at a time from `ValueReader::read_windows`.
pub trait WindowSink {
    /// One sink per schema field for `[start, end)`, pre-filled with the
    /// target's fill value.
    fn sinks(&mut self, start: u64, end: u64) -> Result<Vec<OutputSinkMut<'_>>, ReaderError>;
    /// `[start, end)` is complete. `covered == false` means the source has
    /// no record overlapping it; `sinks` may not have been called for it.
    fn done(&mut self, start: u64, end: u64, covered: bool) -> Result<(), ReaderError>;
}

/// One source, decoded in a single pass into one sink per schema field.
///
/// `Send` but not `Sync`: each worker gets its own fork, so a reader holds
/// its file handles and decode buffers directly, with no internal `Mutex`.
pub trait ValueReader: Send + Sized {
    /// Contigs present in the source, with lengths.
    fn contigs(&self) -> &Genome;

    fn output_schema(&self) -> &OutputSchema;

    /// Fill `outputs` with values for `[start, end)` on `contig`. `outputs[i]`
    /// is `output_schema().fields[i]`, already sliced to the window and
    /// indexed from 0. The caller has pre-filled it with the target track's
    /// fill value: overwrite only the positions the source covers, never
    /// write gap filler.
    fn read_into(
        &mut self,
        contig: &str,
        start: u64,
        end: u64,
        outputs: &mut [OutputSinkMut<'_>],
    ) -> Result<(), ReaderError>;

    /// Cheap pre-check: could this reader have any data in `[start, end)`?
    /// A sparse source returns `false` so the engine can skip whole buffers.
    fn may_have_data(&self, _contig: &str, _start: u64, _end: u64) -> bool {
        true
    }

    /// A worker-local handle. Shared state (index, header) is reused through
    /// `Arc`; per-thread state (file handle, decode buffers) is reallocated.
    fn fork(&self) -> Result<Self, ReaderError>;

    /// Decode `[start, end)` as consecutive windows split at `cuts`
    /// (ascending, strictly inside the range), calling `sink` per window in
    /// order. The default probes and reads one window at a time; a reader
    /// that can open the range once and stream should override it.
    fn read_windows(
        &mut self,
        contig: &str,
        start: u64,
        end: u64,
        cuts: &[u64],
        sink: &mut dyn WindowSink,
    ) -> Result<(), ReaderError> {
        let mut lo = start;
        for &hi in cuts.iter().chain(std::iter::once(&end)) {
            if hi <= lo {
                continue;
            }
            let covered = self.may_have_data(contig, lo, hi);
            if covered {
                let mut outputs = sink.sinks(lo, hi)?;
                self.read_into(contig, lo, hi, &mut outputs)?;
            }
            sink.done(lo, hi, covered)?;
            lo = hi;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn typed_accessor_matches_and_mismatches() {
        let mut buf = Array1::<i32>::zeros(4);
        let mut sink = OutputSinkMut::I32(buf.view_mut());
        assert_eq!(sink.dtype(), Dtype::I32);
        assert_eq!(sink.len(), 4);
        sink.as_i32_mut().unwrap()[2] = 7;
        assert!(matches!(
            sink.as_f32_mut(),
            Err(ReaderError::SchemaMismatch { .. })
        ));
        assert_eq!(buf[2], 7);
    }

    /// The default `read_windows` visits every window in order, probes each
    /// one, and reads only the covered ones.
    #[test]
    fn default_read_windows_visits_each_window_once() {
        struct Log(Vec<(u64, u64, bool)>, Vec<i32>);
        impl WindowSink for Log {
            fn sinks(
                &mut self,
                start: u64,
                end: u64,
            ) -> Result<Vec<OutputSinkMut<'_>>, ReaderError> {
                self.1 = vec![0; (end - start) as usize];
                Ok(vec![OutputSinkMut::I32(ArrayViewMut1::from(
                    self.1.as_mut_slice(),
                ))])
            }
            fn done(&mut self, start: u64, end: u64, covered: bool) -> Result<(), ReaderError> {
                self.0.push((start, end, covered));
                Ok(())
            }
        }
        struct Half; // covers [0, 10) only
        impl ValueReader for Half {
            fn contigs(&self) -> &Genome {
                unimplemented!()
            }
            fn output_schema(&self) -> &OutputSchema {
                unimplemented!()
            }
            fn read_into(
                &mut self,
                _: &str,
                start: u64,
                end: u64,
                outputs: &mut [OutputSinkMut<'_>],
            ) -> Result<(), ReaderError> {
                assert!(
                    end <= 10,
                    "read_into called on an uncovered window {start}..{end}"
                );
                outputs[0].as_i32_mut()?.fill(1);
                Ok(())
            }
            fn may_have_data(&self, _: &str, start: u64, _end: u64) -> bool {
                start < 10
            }
            fn fork(&self) -> Result<Self, ReaderError> {
                Ok(Half)
            }
        }
        let mut log = Log(Vec::new(), Vec::new());
        Half.read_windows("c", 2, 18, &[5, 10, 15], &mut log)
            .unwrap();
        assert_eq!(
            log.0,
            vec![
                (2, 5, true),
                (5, 10, true),
                (10, 15, false),
                (15, 18, false)
            ]
        );
    }
}
