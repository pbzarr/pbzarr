//! User-supplied Zarr v3 array metadata: the `codecs` list and optional
//! `chunk_grid` that replace pbzarr's default pipeline at track creation.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use zarrs::array::codec::api::{
    ArrayToArrayCodecTraits, ArrayToBytesCodecTraits, BytesToBytesCodecTraits, Codec,
};
use zarrs::metadata::v3::MetadataV3;

use crate::Result;
use crate::error::PbzError;

/// Explicit array spec parsed from the CLI `--codecs` flag or the Python
/// `codecs=` keyword: standard Zarr v3 metadata, stored verbatim and applied
/// when the track's `values` array is created.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExplicitArraySpec {
    /// Outer chunk grid (the `zarr.json` `chunk_grid` member). `None` keeps
    /// pbzarr's default grid.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunk_grid: Option<MetadataV3>,
    /// Codec chain (the `zarr.json` `codecs` member).
    pub codecs: Vec<MetadataV3>,
}

impl ExplicitArraySpec {
    /// Parse a JSON string: either a bare `codecs` array, or a
    /// `{chunk_grid?, codecs}` fragment.
    pub fn parse(text: &str) -> Result<Self> {
        let value: Value = serde_json::from_str(text)
            .map_err(|e| PbzError::Metadata(format!("codecs spec is not valid JSON: {e}")))?;
        Self::from_value(value)
    }

    /// Same as [`parse`](Self::parse), from an already-parsed JSON value.
    pub fn from_value(value: Value) -> Result<Self> {
        match value {
            Value::Array(_) => Ok(Self {
                chunk_grid: None,
                codecs: serde_json::from_value(value)
                    .map_err(|e| PbzError::Metadata(format!("codecs list: {e}")))?,
            }),
            Value::Object(_) => serde_json::from_value(value)
                .map_err(|e| PbzError::Metadata(format!("codecs spec: {e}"))),
            other => Err(PbzError::Metadata(format!(
                "codecs spec must be a JSON array or object, got {other}"
            ))),
        }
    }
}

/// The spec's codec entries resolved through the zarrs registry and split the
/// way `ArrayBuilder` consumes them.
pub(crate) struct ResolvedCodecs {
    pub(crate) array_to_array: Vec<Arc<dyn ArrayToArrayCodecTraits>>,
    /// `None` when the spec names no array-to-bytes codec; the caller then
    /// supplies the default little-endian `bytes` codec.
    pub(crate) array_to_bytes: Option<Arc<dyn ArrayToBytesCodecTraits>>,
    pub(crate) bytes_to_bytes: Vec<Arc<dyn BytesToBytesCodecTraits>>,
}

pub(crate) fn resolve_codecs(track: &str, codecs: &[MetadataV3]) -> Result<ResolvedCodecs> {
    let mut resolved = ResolvedCodecs {
        array_to_array: Vec::new(),
        array_to_bytes: None,
        bytes_to_bytes: Vec::new(),
    };
    for metadata in codecs {
        let codec = Codec::from_metadata(metadata).map_err(|e| {
            PbzError::Metadata(format!("track {track:?}: codec {:?}: {e}", metadata.name()))
        })?;
        match codec {
            Codec::ArrayToArray(c) => resolved.array_to_array.push(c),
            Codec::ArrayToBytes(c) => {
                if resolved.array_to_bytes.is_some() {
                    return Err(PbzError::Metadata(format!(
                        "track {track:?}: more than one array-to-bytes codec in codecs spec"
                    )));
                }
                resolved.array_to_bytes = Some(c);
            }
            Codec::BytesToBytes(c) => resolved.bytes_to_bytes.push(c),
        }
    }
    Ok(resolved)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_bare_codecs_array() {
        let spec = ExplicitArraySpec::parse(
            r#"[{"name":"transpose","configuration":{"order":[1,0]}},
                {"name":"zstd","configuration":{"level":3,"checksum":false}}]"#,
        )
        .unwrap();
        assert!(spec.chunk_grid.is_none());
        assert_eq!(spec.codecs.len(), 2);
    }

    #[test]
    fn parses_fragment_with_chunk_grid() {
        let spec = ExplicitArraySpec::parse(
            r#"{"chunk_grid":{"name":"regular","configuration":{"chunk_shape":[8,4]}},
                "codecs":[{"name":"zstd","configuration":{"level":3,"checksum":false}}]}"#,
        )
        .unwrap();
        assert!(spec.chunk_grid.is_some());
        assert_eq!(spec.codecs.len(), 1);
    }

    #[test]
    fn rejects_scalars_and_unknown_keys() {
        assert!(ExplicitArraySpec::parse("3").is_err());
        assert!(ExplicitArraySpec::parse(r#"{"codec":[]}"#).is_err());
        assert!(ExplicitArraySpec::parse("not json").is_err());
    }

    #[test]
    fn resolve_splits_by_kind_and_rejects_bad_chains() {
        let spec = ExplicitArraySpec::parse(
            r#"[{"name":"transpose","configuration":{"order":[1,0]}},
                {"name":"bytes","configuration":{"endian":"little"}},
                {"name":"zstd","configuration":{"level":3,"checksum":false}}]"#,
        )
        .unwrap();
        let resolved = resolve_codecs("t", &spec.codecs).unwrap();
        assert_eq!(resolved.array_to_array.len(), 1);
        assert!(resolved.array_to_bytes.is_some());
        assert_eq!(resolved.bytes_to_bytes.len(), 1);

        let doubled = ExplicitArraySpec::parse(
            r#"[{"name":"bytes","configuration":{"endian":"little"}},
                {"name":"bytes","configuration":{"endian":"little"}}]"#,
        )
        .unwrap();
        assert!(resolve_codecs("t", &doubled.codecs).is_err());

        let unknown = ExplicitArraySpec::parse(r#"[{"name":"no-such-codec"}]"#).unwrap();
        assert!(resolve_codecs("t", &unknown.codecs).is_err());
    }
}
