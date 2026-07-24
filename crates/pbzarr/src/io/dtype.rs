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

    /// In-memory element size in bytes, matching the pipeline's per-chunk byte
    /// accounting (positions × columns × this).
    pub fn size_bytes(self) -> usize {
        match self {
            Dtype::U8 | Dtype::I8 | Dtype::Bool => 1,
            Dtype::U16 | Dtype::I16 => 2,
            Dtype::U32 | Dtype::I32 | Dtype::F32 => 4,
            Dtype::F64 => 8,
        }
    }

    /// Parse a Zarr v3 dtype string into a `Dtype`.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Result<Self, crate::error::PbzError> {
        Ok(match s {
            "uint8" => Dtype::U8,
            "uint16" => Dtype::U16,
            "uint32" => Dtype::U32,
            "int8" => Dtype::I8,
            "int16" => Dtype::I16,
            "int32" => Dtype::I32,
            "float32" => Dtype::F32,
            "float64" => Dtype::F64,
            "bool" => Dtype::Bool,
            other => {
                return Err(crate::error::PbzError::InvalidDtype {
                    dtype: other.to_owned(),
                });
            }
        })
    }
}

impl Dtype {
    /// Inverse of the store-side `dtype_to_zarrs`: map an opened array's
    /// zarrs `DataType` back to a `Dtype` tag. Errors on any dtype the
    /// library doesn't support (custom extension types, complex, etc.).
    pub fn from_zarrs(dt: &zarrs::array::DataType) -> Result<Self, crate::error::PbzError> {
        use zarrs::array::data_type;
        if *dt == data_type::uint8() {
            Ok(Dtype::U8)
        } else if *dt == data_type::uint16() {
            Ok(Dtype::U16)
        } else if *dt == data_type::uint32() {
            Ok(Dtype::U32)
        } else if *dt == data_type::int8() {
            Ok(Dtype::I8)
        } else if *dt == data_type::int16() {
            Ok(Dtype::I16)
        } else if *dt == data_type::int32() {
            Ok(Dtype::I32)
        } else if *dt == data_type::float32() {
            Ok(Dtype::F32)
        } else if *dt == data_type::float64() {
            Ok(Dtype::F64)
        } else if *dt == data_type::bool() {
            Ok(Dtype::Bool)
        } else {
            Err(crate::error::PbzError::InvalidDtype {
                dtype: dt.to_string(),
            })
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
/// dtype-erased Zarr-writer boundary. `ZERO` is the type's additive identity
/// (or `false` for bool); used as scratch-buffer fill in the import pipeline.
///
/// The `zarrs::array::Element + ElementOwned` supertrait bounds are required by
/// the zarrs I/O methods in `Track::read_region` / `write_region`.
pub trait Numeric:
    Copy + Send + Sync + 'static + zarrs::array::Element + zarrs::array::ElementOwned
{
    const DTYPE: Dtype;
    const ZERO: Self;
}

impl Numeric for u8 {
    const DTYPE: Dtype = Dtype::U8;
    const ZERO: Self = 0;
}
impl Numeric for u16 {
    const DTYPE: Dtype = Dtype::U16;
    const ZERO: Self = 0;
}
impl Numeric for u32 {
    const DTYPE: Dtype = Dtype::U32;
    const ZERO: Self = 0;
}
impl Numeric for i8 {
    const DTYPE: Dtype = Dtype::I8;
    const ZERO: Self = 0;
}
impl Numeric for i16 {
    const DTYPE: Dtype = Dtype::I16;
    const ZERO: Self = 0;
}
impl Numeric for i32 {
    const DTYPE: Dtype = Dtype::I32;
    const ZERO: Self = 0;
}
impl Numeric for f32 {
    const DTYPE: Dtype = Dtype::F32;
    const ZERO: Self = 0.0;
}
impl Numeric for f64 {
    const DTYPE: Dtype = Dtype::F64;
    const ZERO: Self = 0.0;
}
impl Numeric for bool {
    const DTYPE: Dtype = Dtype::Bool;
    const ZERO: Self = false;
}
