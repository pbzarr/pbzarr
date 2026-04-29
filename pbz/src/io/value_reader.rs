use color_eyre::Result;
use ndarray::Array1;

/// Source of per-base values for ingestion into a PBZ track.
///
/// PBZ tracks store per-base values of any kind (depth, methylation, masks,
/// etc.), so this trait is intentionally generic over what the values mean.
/// Implementations may hold mutable cursors (e.g. a D4 streaming view), so
/// `read_chunk` takes `&mut self` rather than `&self`.
pub trait ValueReader: Send {
    fn dtype(&self) -> ValueDtype;
    fn contigs(&self) -> &[(String, u64)];
    fn read_chunk(&mut self, contig: &str, start: u64, end: u64) -> Result<ValueChunk>;
}

#[derive(Debug)]
// Variants beyond U32 cover dtypes that future readers (BigWig, methylBED, BED
// masks) will produce. Keeping the full set in place now means new readers can
// land without churning the trait surface.
#[allow(dead_code)]
pub enum ValueChunk {
    U8(Array1<u8>),
    U16(Array1<u16>),
    U32(Array1<u32>),
    I8(Array1<i8>),
    I16(Array1<i16>),
    I32(Array1<i32>),
    F32(Array1<f32>),
    F64(Array1<f64>),
    Bool(Array1<bool>),
}

// Returning a string for dtype made typos at use sites silent. The enum mirrors
// `ValueChunk` 1:1 so reader implementations can declare their value kind without
// stringly-typed comparisons leaking into pipeline logic.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[allow(dead_code)]
pub enum ValueDtype {
    U8,
    U16,
    U32,
    I8,
    I16,
    I32,
    F32,
    F64,
    Bool,
}

impl ValueDtype {
    /// Library-side string form (matches `pbzarr` `TrackConfig::dtype`).
    pub fn as_str(self) -> &'static str {
        match self {
            ValueDtype::U8 => "uint8",
            ValueDtype::U16 => "uint16",
            ValueDtype::U32 => "uint32",
            ValueDtype::I8 => "int8",
            ValueDtype::I16 => "int16",
            ValueDtype::I32 => "int32",
            ValueDtype::F32 => "float32",
            ValueDtype::F64 => "float64",
            ValueDtype::Bool => "bool",
        }
    }
}

impl std::fmt::Display for ValueDtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}
