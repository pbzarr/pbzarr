use color_eyre::Result;
use ndarray::Array1;

/// Source of per-base depth (or similar) values for ingestion into a PBZ track.
///
/// Implementations may hold mutable cursors (e.g. a D4 streaming view), so
/// `read_chunk` takes `&mut self` rather than `&self`.
pub trait DepthReader: Send {
    fn dtype(&self) -> &str;
    fn contigs(&self) -> &[(String, u64)];
    fn read_chunk(&mut self, contig: &str, start: u64, end: u64) -> Result<DepthChunk>;
}

#[derive(Debug)]
// Variants beyond U32 cover dtypes that future readers (BigWig, methylBED, BED
// masks) will produce. Keeping the full set in place now means new readers can
// land without churning the trait surface.
#[allow(dead_code)]
pub enum DepthChunk {
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
