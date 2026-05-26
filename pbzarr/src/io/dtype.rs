/// Runtime tag for the  numeric types supported by `ValueReader` impls.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Dtype {
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

impl Dtype {
    /// Matches `pbzarr` `TrackConfig::dtype` (Zarr v3 dtype strings).
    pub fn as_str(self) -> &'static str {
        match self {
            Dtype::U8 => "uint8",
            Dtype::U16 => "uint16",
            Dtype::U32 => "uint32",
            Dtype::I8 => "int8",
            Dtype::I16 => "int16",
            Dtype::I32 => "int32",
            Dtype::F32 => "float32",
            Dtype::F64 => "float64",
            Dtype::Bool => "bool",
        }
    }
}

impl std::fmt::Display for Dtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Numeric value kinds a `ValueReader` may produce.
///
/// `const DTYPE` lets generic code recover the runtime tag — needed at the
/// dtype-erased Zarr-writer boundary.
///
/// The `zarrs::array::Element + ElementOwned` supertrait bounds are required by
/// the zarrs I/O methods in `Track::read_region` / `write_region`.
pub trait Numeric:
    Copy + Send + Sync + 'static + zarrs::array::Element + zarrs::array::ElementOwned
{
    const DTYPE: Dtype;
}

impl Numeric for u8   { const DTYPE: Dtype = Dtype::U8;   }
impl Numeric for u16  { const DTYPE: Dtype = Dtype::U16;  }
impl Numeric for u32  { const DTYPE: Dtype = Dtype::U32;  }
impl Numeric for i8   { const DTYPE: Dtype = Dtype::I8;   }
impl Numeric for i16  { const DTYPE: Dtype = Dtype::I16;  }
impl Numeric for i32  { const DTYPE: Dtype = Dtype::I32;  }
impl Numeric for f32  { const DTYPE: Dtype = Dtype::F32;  }
impl Numeric for f64  { const DTYPE: Dtype = Dtype::F64;  }
impl Numeric for bool { const DTYPE: Dtype = Dtype::Bool; }