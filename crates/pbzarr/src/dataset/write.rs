//! Minimal creation and import support for the root Dataset convention.
use super::{Dataset, FORMAT_VERSION, Reference};
use crate::{PbzError, Result};
use std::{path::Path, sync::Arc};
use zarrs::{
    array::{ArrayBuilder, data_type},
    storage::{ReadableWritableListableStorage, StorePrefix},
};

fn storage_error(error: impl std::fmt::Display) -> PbzError {
    PbzError::Store(error.to_string())
}

impl Dataset {
    /// Create a new core-draft-1 Dataset with a fixed ordered reference.
    ///
    /// The storage namespace must be empty. No conversion or replacement is performed.
    pub fn create(path: impl AsRef<Path>, reference: Reference) -> Result<Self> {
        let storage = zarrs::filesystem::FilesystemStore::new(path).map_err(storage_error)?;
        Self::create_with_storage(Arc::new(storage), reference)
    }

    /// Create a Dataset in an empty storage namespace.
    ///
    /// Reference arrays are written before root metadata. Errors may leave
    /// incomplete output; automatic cleanup and recovery are not implemented.
    pub fn create_with_storage(
        storage: ReadableWritableListableStorage,
        reference: Reference,
    ) -> Result<Self> {
        let entries = storage
            .list_dir(&StorePrefix::root())
            .map_err(storage_error)?;
        if !entries.keys().is_empty() || !entries.prefixes().is_empty() {
            return Err(PbzError::Metadata(
                "Dataset creation requires an empty storage namespace".into(),
            ));
        }
        let n = reference.contigs().len() as u64;
        let names = ArrayBuilder::new(vec![n], vec![n.max(1)], data_type::string(), "")
            .dimension_names(["contig"].into())
            .build(storage.clone(), "/contig")
            .map_err(storage_error)?;
        let lengths = ArrayBuilder::new(vec![n], vec![n.max(1)], data_type::int64(), 0_i64)
            .dimension_names(["contig"].into())
            .build(storage.clone(), "/contig_length")
            .map_err(storage_error)?;
        if n > 0 {
            let values: Vec<String> = reference.contigs().iter().map(|c| c.name.clone()).collect();
            names
                .store_chunk(&[0], ndarray::Array1::from(values))
                .map_err(storage_error)?;
            let values: Vec<i64> = reference
                .contigs()
                .iter()
                .map(|c| c.length as i64)
                .collect();
            lengths
                .store_chunk(&[0], ndarray::Array1::from(values))
                .map_err(storage_error)?;
        }
        names.store_metadata().map_err(storage_error)?;
        lengths.store_metadata().map_err(storage_error)?;
        let mut root = zarrs::group::GroupBuilder::new()
            .build(storage.clone(), "/")
            .map_err(storage_error)?;
        root.attributes_mut().extend(
            serde_json::from_value::<serde_json::Map<String, serde_json::Value>>(
                serde_json::json!({
                    "zarr_conventions": [{"uuid":crate::PERBASE_CONVENTION_UUID,"name":"pbz:"}],
                    "pbz:version":FORMAT_VERSION,
                    "pbz:reference":{"names":"contig","lengths":"contig_length"},
                    "coordinates":"contig_length"
                }),
            )
            .expect("literal attribute object"),
        );
        root.store_metadata().map_err(storage_error)?;
        Self::open_with_storage(storage)
    }
}

use super::{GenomicAxis, Role, Support, VariableInfo};
use crate::import::destination::Destination;
use crate::store::{apply_explicit_spec, default_data_codecs, dtype_to_zarrs, resolve_fill_value};
use crate::{Genome, Track, TrackConfig, validate_node_name};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{RwLock, atomic::AtomicBool};
use zarrs::array::codec::api::ArrayToBytesCodecTraits;
use zarrs::storage::StoreKey;

pub(crate) struct PreparedImport {
    pub(crate) targets: Vec<Destination<'static>>,
    coordinates: Vec<(Arc<super::read::ArrayHandle>, Vec<String>)>,
}

impl Dataset {
    /// Prepare existing 1D/2D import destinations without writing the store.
    pub(crate) fn prepare_import(&self, specs: &[(String, TrackConfig)]) -> Result<PreparedImport> {
        if specs.is_empty() {
            return Err(PbzError::Metadata("import: no destination tracks".into()));
        }
        let reference = self.reference()?;
        let total = reference.extent(&GenomicAxis::Position)?;
        let genome = Arc::new(Genome::new(reference.contigs().to_vec())?);
        let entries = self
            .storage
            .list_dir(&StorePrefix::root())
            .map_err(storage_error)?;
        let occupied: BTreeSet<String> = entries
            .keys()
            .iter()
            .map(|k| k.as_str().to_owned())
            .chain(
                entries
                    .prefixes()
                    .iter()
                    .map(|p| p.as_str().trim_end_matches('/').to_owned()),
            )
            .collect();
        let mut names = BTreeSet::new();
        let dimensions: BTreeSet<&str> = self
            .variables
            .values()
            .flat_map(|v| v.dims.iter().map(String::as_str))
            .collect();
        for (name, _) in specs {
            validate_node_name(name)?;
            if !names.insert(name.clone())
                || occupied.contains(name)
                || name == "position"
                || dimensions.contains(name.as_str())
            {
                return Err(PbzError::Metadata(format!(
                    "import: destination {name:?} is occupied or conflicts with a dimension"
                )));
            }
            for attrs in std::iter::once(&self.root["attributes"])
                .chain(self.variables.values().map(|v| v.attrs()))
            {
                if attrs["coordinates"]
                    .as_str()
                    .is_some_and(|s| s.split_whitespace().any(|v| v == name))
                {
                    return Err(PbzError::Metadata(format!(
                        "import: {name:?} is declared as a coordinate"
                    )));
                }
            }
        }
        let mut shared_labels: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let mut targets = Vec::with_capacity(specs.len());
        for (name, config) in specs {
            let mut shape = vec![total];
            let mut dims = vec!["position".to_owned()];
            let column_dim = config
                .columns
                .as_ref()
                .map(|_| config.column_dim.clone().unwrap_or_else(|| "column".into()));
            if let (Some(labels), Some(dim)) = (&config.columns, &column_dim) {
                validate_node_name(dim)?;
                if GenomicAxis::from_dimension(dim)?.is_some() || names.contains(dim) {
                    return Err(PbzError::Metadata(format!(
                        "import: column dimension {dim:?} conflicts with genomic geometry or a track"
                    )));
                }
                let unique: BTreeSet<_> = labels.iter().collect();
                if unique.len() != labels.len() {
                    return Err(PbzError::Metadata(format!(
                        "import: duplicate labels for {dim:?}"
                    )));
                }
                if let Some(previous) = shared_labels.get(dim) {
                    if previous != labels {
                        return Err(PbzError::Metadata(format!(
                            "import: conflicting labels for {dim:?}"
                        )));
                    }
                } else {
                    if let Some(info) = self.variables.get(dim) {
                        self.validate_labels(dim)?;
                        let values = self.coordinate_values(info)?;
                        let actual: Option<Vec<&str>> = values.iter().map(|v| v.as_str()).collect();
                        // Decode into owned strings before dropping the coordinate buffer.
                        let actual = actual
                            .map(|v| v.into_iter().map(str::to_owned).collect::<Vec<_>>())
                            .ok_or_else(|| {
                                PbzError::UnsupportedEncoding(format!(
                                    "import: {dim:?} requires string labels"
                                ))
                            })?;
                        if actual != *labels {
                            return Err(PbzError::Metadata(format!(
                                "import: labels for {dim:?} differ from the Dataset"
                            )));
                        }
                    } else if occupied.contains(dim) {
                        return Err(PbzError::Metadata(format!(
                            "import: coordinate destination {dim:?} is occupied"
                        )));
                    }
                    shared_labels.insert(dim.clone(), labels.clone());
                }
                shape.push(labels.len() as u64);
                dims.push(dim.clone());
            }
            for info in self.variables.values() {
                for (dim, extent) in dims.iter().zip(&shape) {
                    if let Some(axis) = info.dims.iter().position(|d| d == dim)
                        && info.shape.get(axis) != Some(extent)
                    {
                        return Err(PbzError::Metadata(format!(
                            "import: inconsistent shared dimension {dim:?} on {:?}",
                            info.name
                        )));
                    }
                }
            }
            if config.extra.contains_key("coordinates") {
                return Err(PbzError::Metadata("import: coordinate declarations cannot be attached through track extra attributes".into()));
            }
            let chunk_pos = (config.chunk_size as u64).min(total.max(1)).max(1);
            let shard_pos = config
                .shard_size
                .map(|n| (n as u64).min(total.max(1)).max(1));
            let mut chunk = vec![chunk_pos];
            let mut outer = vec![shard_pos.unwrap_or(chunk_pos)];
            if let Some(labels) = &config.columns {
                let width = labels.len() as u64;
                let col_chunk = config
                    .column_chunk_size
                    .map(|n| n as u64)
                    .unwrap_or(width)
                    .min(width)
                    .max(1);
                let shard_col = config
                    .shard_column_size
                    .map(|n| n as u64)
                    .unwrap_or(width)
                    .min(width)
                    .max(1);
                chunk.push(col_chunk);
                outer.push(if shard_pos.is_some() {
                    shard_col
                } else {
                    col_chunk
                });
            }
            let mut builder = ArrayBuilder::new(
                shape.clone(),
                if config.codecs.is_some() {
                    chunk.clone()
                } else {
                    outer
                },
                dtype_to_zarrs(config.dtype),
                resolve_fill_value(config.dtype, config.fill_value.as_ref())?,
            );
            builder.dimension_names(Some(dims.clone()));
            builder.attributes(config.extra.clone());
            if let Some(value) = &config.description {
                builder
                    .attributes_mut()
                    .insert("description".into(), value.clone().into());
            }
            if let Some(value) = &config.source {
                builder
                    .attributes_mut()
                    .insert("source".into(), value.clone().into());
            }
            if let Some(spec) = &config.codecs {
                apply_explicit_spec(&mut builder, name, spec, &shape)?;
            } else {
                builder.bytes_to_bytes_codecs(default_data_codecs(config.dtype)?);
                if shard_pos.is_some() {
                    builder.subchunk_shape(Some(chunk));
                }
            }
            let metadata = builder.build_metadata().map_err(storage_error)?;
            let metadata =
                serde_json::to_value(metadata).map_err(|e| PbzError::Metadata(e.to_string()))?;
            if metadata["chunk_grid"]["name"] != "regular" {
                return Err(PbzError::UnsupportedEncoding(
                    "import requires a regular outer chunk grid".into(),
                ));
            }
            let info = VariableInfo {
                name: name.clone(),
                dims,
                shape,
                role: Role::Measurement,
                metadata,
                logical_support: Support::Supported,
            };
            super::metadata::storage_metadata(&info)?;
            if let Some(reason) = super::metadata::encoding_limit(&info) {
                return Err(PbzError::UnsupportedEncoding(reason));
            }
            let array = Arc::new(
                builder
                    .build(self.storage.clone(), &format!("/{name}"))
                    .map_err(storage_error)?,
            );
            let chunk_shape = array
                .chunk_shape(&vec![0; array.shape().len()])
                .map_err(storage_error)?;
            array
                .codecs()
                .encoded_representation(
                    chunk_shape.as_slice(),
                    array.data_type(),
                    array.fill_value(),
                )
                .map_err(storage_error)?;
            let track = Track {
                name: name.clone(),
                genome: genome.clone(),
                dtype: config.dtype,
                rank: array.shape().len(),
                column_dim,
                storage: self.storage.clone(),
                values: RwLock::new(Some(array)),
                sealed: AtomicBool::new(false),
            };
            targets.push(Destination::Root {
                track: Box::new(track),
                labels: config.columns.clone(),
            });
        }
        let mut coordinates = Vec::new();
        for (dim, labels) in shared_labels {
            if !self.variables.contains_key(&dim) {
                let n = labels.len() as u64;
                let array = ArrayBuilder::new(vec![n], vec![n.max(1)], data_type::string(), "")
                    .dimension_names(Some(vec![dim.clone()]))
                    .build(self.storage.clone(), &format!("/{dim}"))
                    .map_err(storage_error)?;
                coordinates.push((Arc::new(array), labels));
            }
        }
        Ok(PreparedImport {
            targets,
            coordinates,
        })
    }

    pub(crate) fn begin_import(&mut self, prepared: &PreparedImport) -> Result<()> {
        // Consolidation must not hide additions from native Xarray discovery.
        if self.root.get("consolidated_metadata").is_some() {
            let mut root = self.root.clone();
            root.as_object_mut()
                .expect("validated root object")
                .remove("consolidated_metadata");
            self.storage
                .set(
                    &StoreKey::new("zarr.json").expect("root key"),
                    serde_json::to_vec(&root)
                        .map_err(|e| PbzError::Metadata(e.to_string()))?
                        .into(),
                )
                .map_err(storage_error)?;
            self.root = root;
        }
        for (array, labels) in &prepared.coordinates {
            if !labels.is_empty() {
                array
                    .store_chunk(&[0], ndarray::Array1::from(labels.clone()))
                    .map_err(storage_error)?;
            }
            array.store_metadata().map_err(storage_error)?;
        }
        Ok(())
    }

    pub(crate) fn finish_import(&mut self, prepared: &PreparedImport) -> Result<()> {
        for target in &prepared.targets {
            target
                .values_array()?
                .store_metadata()
                .map_err(storage_error)?;
        }
        *self = Self::open_with_storage(self.storage.clone())?;
        Ok(())
    }
}
