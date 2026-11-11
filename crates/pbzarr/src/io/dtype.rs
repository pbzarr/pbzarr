/// Runtime tag for the numeric types supported by `ValueReader` impls.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Dtype {
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
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
            Dtype::U64 => "uint64",
            Dtype::I8 => "int8",
            Dtype::I16 => "int16",
            Dtype::I32 => "int32",
            Dtype::I64 => "int64",
            Dtype::F32 => "float32",
            Dtype::F64 => "float64",
            Dtype::Bool => "bool",
        }
    }

    /// Parse a Zarr v3 dtype string into a `Dtype`.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Result<Self, crate::error::PbzError> {
        Ok(match s {
            "uint8" => Dtype::U8,
            "uint16" => Dtype::U16,
            "uint32" => Dtype::U32,
            "uint64" => Dtype::U64,
            "int8" => Dtype::I8,
            "int16" => Dtype::I16,
            "int32" => Dtype::I32,
            "int64" => Dtype::I64,
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

    /// Element size in bytes.
    pub fn size(self) -> usize {
        match self {
            Dtype::U8 | Dtype::I8 | Dtype::Bool => 1,
            Dtype::U16 | Dtype::I16 => 2,
            Dtype::U32 | Dtype::I32 | Dtype::F32 => 4,
            Dtype::U64 | Dtype::I64 | Dtype::F64 => 8,
        }
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
        } else if *dt == data_type::uint64() {
            Ok(Dtype::U64)
        } else if *dt == data_type::int8() {
            Ok(Dtype::I8)
        } else if *dt == data_type::int16() {
            Ok(Dtype::I16)
        } else if *dt == data_type::int32() {
            Ok(Dtype::I32)
        } else if *dt == data_type::int64() {
            Ok(Dtype::I64)
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

impl Dtype {
    /// The value that means missing in a track of this dtype: NaN for floats,
    /// the type maximum for integers. Bool has no null.
    pub fn null(self) -> Option<Null> {
        Some(match self {
            Dtype::U8 => Null::Int(u8::MAX.into()),
            Dtype::U16 => Null::Int(u16::MAX.into()),
            Dtype::U32 => Null::Int(u32::MAX.into()),
            Dtype::U64 => Null::Int(u64::MAX.into()),
            Dtype::I8 => Null::Int(i8::MAX.into()),
            Dtype::I16 => Null::Int(i16::MAX.into()),
            Dtype::I32 => Null::Int(i32::MAX.into()),
            Dtype::I64 => Null::Int(i64::MAX.into()),
            Dtype::F32 | Dtype::F64 => Null::Nan,
            Dtype::Bool => return None,
        })
    }
}

/// A value that marks a missing position.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Null {
    Nan,
    Int(i128),
}

impl Null {
    /// The CF `_FillValue` attribute that declares this null: base64 of the
    /// little-endian f64 for NaN, as xarray writes it, else a JSON integer.
    pub fn attribute(self) -> serde_json::Value {
        use base64::Engine as _;
        match self {
            Null::Nan => base64::engine::general_purpose::STANDARD
                .encode(f64::NAN.to_le_bytes())
                .into(),
            Null::Int(v) => match (i64::try_from(v), u64::try_from(v)) {
                (Ok(v), _) => v.into(),
                (_, Ok(v)) => v.into(),
                _ => serde_json::Value::Null,
            },
        }
    }
}

impl std::fmt::Display for Dtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Numeric element kinds a track can store.
///
/// `const DTYPE` lets generic code recover the runtime tag — needed at the
/// dtype-erased Zarr-writer boundary.
///
/// The `zarrs::array::Element + ElementOwned` supertrait bounds are required by
/// the zarrs array read and write methods.
pub trait Numeric:
    Copy + Send + Sync + 'static + zarrs::array::Element + zarrs::array::ElementOwned
{
    const DTYPE: Dtype;

    /// `null` as a value of this type, when the type can hold it.
    fn from_null(null: Null) -> Option<Self>;
}

macro_rules! impl_numeric_int {
    ($($ty:ty => $dtype:ident),*) => {
        $(impl Numeric for $ty {
            const DTYPE: Dtype = Dtype::$dtype;
            fn from_null(null: Null) -> Option<Self> {
                match null {
                    Null::Int(v) => Self::try_from(v).ok(),
                    Null::Nan => None,
                }
            }
        })*
    };
}

impl_numeric_int!(
    u8 => U8, u16 => U16, u32 => U32, u64 => U64,
    i8 => I8, i16 => I16, i32 => I32, i64 => I64
);
impl Numeric for f32 {
    const DTYPE: Dtype = Dtype::F32;
    fn from_null(null: Null) -> Option<Self> {
        (null == Null::Nan).then_some(Self::NAN)
    }
}
impl Numeric for f64 {
    const DTYPE: Dtype = Dtype::F64;
    fn from_null(null: Null) -> Option<Self> {
        (null == Null::Nan).then_some(Self::NAN)
    }
}
impl Numeric for bool {
    const DTYPE: Dtype = Dtype::Bool;
    fn from_null(_: Null) -> Option<Self> {
        None
    }
}
