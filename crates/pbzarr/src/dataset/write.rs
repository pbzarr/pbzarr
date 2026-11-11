//! Minimal creation and import support for the root Dataset convention.
use super::{CONTIG_DIM, CONTIG_ID, CONTIG_LENGTH, Dataset, FORMAT_VERSION, Reference};
use crate::{PbzError, Result};
use std::{path::Path, sync::Arc};
use zarrs::{
    array::{ArrayBuilder, data_type},
    storage::{ReadableWritableListableStorage, StorePrefix},
};

fn storage_error(error: impl std::fmt::Display) -> PbzError {
    PbzError::Store(error.to_string())
}

/// A shard edge must be a whole number of chunks, or the sharding codec
/// rejects the array. Reject a request that is not, then cap at the fewest
/// whole chunks that cover the array extent.
fn shard_extent(request: u64, chunk: u64, extent: u64, axis: &str) -> Result<u64> {
    if !request.is_multiple_of(chunk) {
        return Err(PbzError::Metadata(format!(
            "import: {axis} shard {request} must be a multiple of the {axis} chunk {chunk}"
        )));
    }
    let cover = extent.max(1).div_ceil(chunk) * chunk;
    Ok(request.min(cover).max(chunk))
}

impl Dataset {
    /// Create a new format-0.5 Dataset with a fixed ordered reference.
    ///
    /// The storage namespace must be empty. No conversion or replacement is performed.
    pub fn create(path: impl AsRef<Path>, reference: Reference) -> Result<Self> {
        let storage =
            zarrs::filesystem::FilesystemStore::new(path.as_ref()).map_err(storage_error)?;
        Self::create_in(
            Arc::new(storage),
            Some(path.as_ref().to_path_buf()),
            reference,
        )
    }

    /// Create a new format-0.5 Dataset that becomes a single-file zip store
    /// at `path`. Nothing appears at `path` until [`Dataset::close`]; a
    /// Dataset dropped before then leaves no file.
    ///
    /// The zip store is written once: each data key takes one write. Use a
    /// sharded layout, so that the archive holds few entries.
    pub fn create_zip(path: impl AsRef<Path>, reference: Reference) -> Result<Self> {
        let zip = Arc::new(crate::zip_store::ZipStore::create(path.as_ref())?);
        let mut dataset = Self::create_in(zip.clone(), None, reference)?;
        dataset.zip = Some(zip);
        Ok(dataset)
    }

    /// Write out a store made by [`Dataset::create_zip`] and move it to its
    /// path. A directory store needs no close, and this does nothing for it.
    pub fn close(self) -> Result<()> {
        match &self.zip {
            Some(zip) => zip.finish(),
            None => Ok(()),
        }
    }

    /// Create a Dataset in an empty storage namespace.
    ///
    /// Reference arrays are written before root metadata. Errors may leave
    /// incomplete output; automatic cleanup and recovery are not implemented.
    pub fn create_with_storage(
        storage: ReadableWritableListableStorage,
        reference: Reference,
    ) -> Result<Self> {
        Self::create_in(storage, None, reference)
    }

    fn create_in(
        storage: ReadableWritableListableStorage,
        dir: Option<std::path::PathBuf>,
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
            .dimension_names([CONTIG_DIM].into())
            .build(storage.clone(), &format!("/{CONTIG_ID}"))
            .map_err(storage_error)?;
        let lengths = ArrayBuilder::new(vec![n], vec![n.max(1)], data_type::int64(), 0_i64)
            .dimension_names([CONTIG_DIM].into())
            .build(storage.clone(), &format!("/{CONTIG_LENGTH}"))
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
                    "zarr_conventions": [{
                        "uuid":crate::PBZ_CONVENTION_UUID,
                        "name":crate::PBZ_CONVENTION_NAME,
                    }],
                    "pbz:version":FORMAT_VERSION,
                    "coordinates":format!("{CONTIG_ID} {CONTIG_LENGTH}")
                }),
            )
            .expect("literal attribute object"),
        );
        root.store_metadata().map_err(storage_error)?;
        let mut dataset = Self::open_in(storage, dir)?;
        dataset.write_snapshot()?;
        Ok(dataset)
    }

    /// Reopen after a write, then replace the root consolidated metadata.
    /// This is the last step of every write: until it lands, the old copy
    /// stays a correct view of the store before the write.
    fn publish(&mut self) -> Result<()> {
        let zip = self.zip.take();
        *self = Self::open_in(self.writable()?.clone(), self.dir.clone())?;
        self.zip = zip;
        self.write_snapshot()
    }

    fn write_snapshot(&mut self) -> Result<()> {
        let mut root = self.root.clone();
        let storage = self.writable()?;
        crate::consolidated::insert_snapshot(storage, &mut root)?;
        crate::consolidated::replace_root(storage, self.dir.as_deref(), &root)?;
        self.root = root;
        Ok(())
    }
}

use super::{GenomicAxis, Role, Support, VariableInfo};
use crate::array::{
    TrackConfig, apply_explicit_spec, default_data_codecs, dtype_to_zarrs, resolve_fill,
    validate_node_name,
};
use crate::import::destination::Destination;
use std::collections::{BTreeMap, BTreeSet};
use zarrs::array::codec::api::ArrayToBytesCodecTraits;
use zarrs::storage::{ReadableWritableListableStorageTraits, StoreKey};

type WritableArray = zarrs::array::Array<dyn ReadableWritableListableStorageTraits>;

pub(crate) struct PreparedImport {
    pub(crate) targets: Vec<Destination>,
    coordinates: Vec<(Arc<WritableArray>, Vec<String>)>,
}

/// Refuse a track or dimension name that the store keeps for its genomic axes
/// and reference.
fn refuse_reserved(name: &str) -> Result<()> {
    if name == "position"
        || name.starts_with("bin_")
        || name == super::CONTIG_DIM
        || super::REFERENCE_ARRAYS.contains(&name)
        || name == "reference"
    {
        return Err(PbzError::Metadata(format!(
            "import: {name:?} is a reserved name. Do not use position, bin_N, contig, \
             contig_id, contig_length, contig_md5 or reference as a track or dimension name"
        )));
    }
    Ok(())
}

impl Dataset {
    /// Prepare existing 1D/2D import destinations without writing the store.
    fn prepare_import(&self, specs: &[(String, TrackConfig)]) -> Result<PreparedImport> {
        if specs.is_empty() {
            return Err(PbzError::Metadata("import: no destination tracks".into()));
        }
        let reference = self.reference()?;
        let total = reference.extent(&GenomicAxis::Position)?;
        let shared_reference = Arc::new(reference.clone());
        let entries = self
            .writable()?
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
            if self.variables.contains_key(name) && !dimensions.contains(name.as_str()) {
                return Err(PbzError::Metadata(format!(
                    "import: the store already has a track named {name:?}"
                )));
            }
            if !names.insert(name.clone())
                || occupied.contains(name)
                || dimensions.contains(name.as_str())
            {
                return Err(PbzError::Metadata(format!(
                    "import: destination {name:?} is occupied or conflicts with a dimension"
                )));
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
                if names.contains(dim) {
                    return Err(PbzError::Metadata(format!(
                        "import: column dimension {dim:?} conflicts with a track"
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
                .map(|n| shard_extent(n as u64, chunk_pos, total, "position"))
                .transpose()?;
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
                // One column chunk per shard by default, so widening the column
                // axis later appends whole shards instead of rewriting one.
                let shard_col = shard_extent(
                    config
                        .shard_column_size
                        .map(|n| n as u64)
                        .unwrap_or(col_chunk),
                    col_chunk,
                    width,
                    "column",
                )?;
                if self.zip.is_none() && !width.is_multiple_of(col_chunk) {
                    log::warn!(
                        "import: track {name:?} is {width} column(s) wide, which is not a \
                         multiple of its column chunk {col_chunk}; this store cannot be \
                         grown later with pbz append"
                    );
                }
                chunk.push(col_chunk);
                outer.push(if shard_pos.is_some() {
                    shard_col
                } else {
                    col_chunk
                });
            }
            let (fill, null) = resolve_fill(config.dtype, config.fill.as_ref())?;
            let mut builder = ArrayBuilder::new(
                shape.clone(),
                if config.codecs.is_some() {
                    chunk.clone()
                } else {
                    outer
                },
                dtype_to_zarrs(config.dtype),
                fill,
            );
            builder.dimension_names(Some(dims.clone()));
            builder.attributes(config.extra.clone());
            if let Some(null) = null {
                builder.attributes_mut().insert("_FillValue".into(), null);
            }
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
                    .build(self.writable()?.clone(), &format!("/{name}"))
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
            targets.push(Destination {
                name: name.clone(),
                reference: Arc::clone(&shared_reference),
                dtype: config.dtype,
                values: array,
                labels: config.columns.clone(),
                column_offset: 0,
            });
        }
        let mut coordinates = Vec::new();
        for (dim, labels) in shared_labels {
            if !self.variables.contains_key(&dim) {
                let n = labels.len() as u64;
                let array = ArrayBuilder::new(vec![n], vec![n.max(1)], data_type::string(), "")
                    .dimension_names(Some(vec![dim.clone()]))
                    .build(self.writable()?.clone(), &format!("/{dim}"))
                    .map_err(storage_error)?;
                coordinates.push((Arc::new(array), labels));
            }
        }
        Ok(PreparedImport {
            targets,
            coordinates,
        })
    }

    fn begin_import(&mut self, prepared: &PreparedImport) -> Result<()> {
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

    fn finish_import(&mut self, prepared: &PreparedImport) -> Result<()> {
        for target in &prepared.targets {
            target
                .values_array()
                .store_metadata()
                .map_err(storage_error)?;
        }
        self.publish()
    }
}

use crate::io::Dtype;
use zarrs::array::{Array, ArrayShardedExt, ArraySubset};
use zarrs::metadata::ArrayMetadata;

/// Which shape a Dataset run writes into: new root arrays, or more columns on
/// existing ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DatasetPlan {
    Create,
    Append,
}

pub(crate) enum Prepared {
    Import(PreparedImport),
    Append(PreparedAppend),
}

impl Prepared {
    pub(crate) fn targets(&self) -> &[Destination] {
        match self {
            Self::Import(prepared) => &prepared.targets,
            Self::Append(prepared) => &prepared.targets,
        }
    }
}

pub(crate) struct PreparedAppend {
    pub(crate) targets: Vec<Destination>,
    /// Every widened variable and its stored metadata carrying the new shape,
    /// published only after the engine's chunk writes land. Writing the stored
    /// JSON back with one field replaced keeps the rest byte for byte; a round
    /// trip through zarrs would respell codecs that zarr-python then rejects.
    grown: Vec<(String, serde_json::Value)>,
    /// The appended dimension's coordinate. It grows last, so a torn append
    /// reads back as a dimension-size conflict instead of silent fill.
    coordinate: (String, serde_json::Value, Arc<WritableArray>),
    /// Labels for the appended columns.
    labels: Vec<String>,
    /// Column width before the append.
    width: u64,
}

/// `info`'s stored metadata with `shape` replaced.
fn grown_metadata(info: &VariableInfo, axis: usize, extent: u64) -> (String, serde_json::Value) {
    let mut shape = info.shape.clone();
    shape[axis] = extent;
    let mut metadata = info.metadata.clone();
    metadata["shape"] = serde_json::json!(shape);
    (info.name.clone(), metadata)
}

/// A rank-2 array's column chunk, and its column shard when sharded.
fn column_grid(array: &WritableArray, name: &str) -> Result<(u64, Option<u64>)> {
    let outer = array.chunk_shape(&[0, 0]).map_err(storage_error)?;
    let shard = outer.as_slice()[1].get();
    if !array.is_sharded() {
        return Ok((shard, None));
    }
    let sub = array.effective_subchunk_shape().ok_or_else(|| {
        PbzError::Metadata(format!(
            "append: track {name:?} is sharded with an indeterminate subchunk shape"
        ))
    })?;
    Ok((sub.as_slice()[1].get(), Some(shard)))
}

impl Dataset {
    fn open_growable(&self, name: &str) -> Result<WritableArray> {
        let info = self.variables.get(name).ok_or_else(|| {
            PbzError::Metadata(format!("append: {name:?} is not a variable of this store"))
        })?;
        let meta = super::metadata::storage_metadata(info)?;
        Array::new_with_metadata(
            self.writable()?.clone(),
            &format!("/{name}"),
            ArrayMetadata::V3(meta),
        )
        .map_err(|e| super::metadata::array_error(name, e))
    }

    fn append_target(&self, name: &str) -> Result<&VariableInfo> {
        let info = self.variables.get(name).ok_or_else(|| {
            PbzError::Metadata(format!("append: {name:?} is not a variable of this store"))
        })?;
        if !info.is_track() {
            return Err(PbzError::Metadata(format!(
                "append: {name:?} is not a track: its first dimension is not position or bin_N"
            )));
        }
        if let Support::Unsupported { reason } = &info.logical_support {
            return Err(PbzError::UnsupportedEncoding(format!(
                "append: {name:?}: {reason}"
            )));
        }
        if info.dims.len() != 2 || info.dims[0] != "position" {
            return Err(PbzError::Metadata(format!(
                "append: {name:?} has dimensions {:?}; a column append needs (position, <column>)",
                info.dims
            )));
        }
        Ok(info)
    }

    /// Prepare an append of new columns onto existing root variables.
    ///
    /// Validates the whole request, then widens the in-memory shape of every
    /// root variable over the appended dimension. Nothing is written here.
    pub(crate) fn prepare_append(&self, specs: &[(String, TrackConfig)]) -> Result<PreparedAppend> {
        if specs.is_empty() {
            return Err(PbzError::Metadata("append: no destination tracks".into()));
        }
        let reference = self.reference()?;
        let total = reference.extent(&GenomicAxis::Position)?;
        let shared_reference = Arc::new(reference.clone());

        let mut dim = String::new();
        let mut labels: Vec<String> = Vec::new();
        let mut width = 0;
        let mut infos = Vec::with_capacity(specs.len());
        for (index, (name, config)) in specs.iter().enumerate() {
            let info = self.append_target(name)?;
            infos.push(info);
            let new = config.columns.clone().ok_or_else(|| {
                PbzError::Metadata(format!(
                    "append: {name:?} has a column dimension but the run supplies no new labels"
                ))
            })?;
            if info.shape[0] != total {
                return Err(PbzError::Metadata(format!(
                    "append: {name:?} spans {} position(s) but the store reference spans {total}",
                    info.shape[0]
                )));
            }
            let dtype = Dtype::from_str(info.metadata["data_type"].as_str().unwrap_or(""))?;
            if dtype != config.dtype {
                return Err(PbzError::InvalidDtype {
                    dtype: format!(
                        "append: {name:?} is {dtype} but the sources supply {}",
                        config.dtype
                    ),
                });
            }
            if let Some(configured) = &config.column_dim
                && *configured != info.dims[1]
            {
                return Err(PbzError::Metadata(format!(
                    "append: {name:?} has column dimension {:?}, not {configured:?}",
                    info.dims[1]
                )));
            }
            if index == 0 {
                dim = info.dims[1].clone();
                labels = new;
                width = info.shape[1];
            } else if info.dims[1] != dim {
                return Err(PbzError::Metadata(format!(
                    "append: {name:?} grows {:?} but {:?} grows {dim:?}; one run grows one \
                     dimension",
                    info.dims[1], specs[0].0
                )));
            } else if new != labels {
                return Err(PbzError::Metadata(
                    "append: every destination track must receive the same new labels".into(),
                ));
            }
        }
        if labels.is_empty() {
            return Err(PbzError::Metadata("append: no new columns".into()));
        }
        let batch = labels.len() as u64;
        if labels.iter().collect::<BTreeSet<_>>().len() != labels.len() {
            return Err(PbzError::Metadata(format!(
                "append: duplicate new labels for {dim:?}"
            )));
        }

        self.validate_labels(&dim)?;
        let coordinate_info = &self.variables[&dim];
        if coordinate_info.shape[0] != width {
            return Err(PbzError::Metadata(format!(
                "append: coordinate {dim:?} carries {} label(s) for {width} column(s)",
                coordinate_info.shape[0]
            )));
        }
        let existing = self.coordinate_values(coordinate_info)?;
        let existing: Vec<String> = existing
            .iter()
            .map(|value| value.as_str().map(str::to_owned))
            .collect::<Option<_>>()
            .ok_or_else(|| {
                PbzError::UnsupportedEncoding(format!(
                    "append: {dim:?} requires string labels to grow"
                ))
            })?;
        for label in &labels {
            if existing.contains(label) {
                return Err(PbzError::Metadata(format!(
                    "append: label {label:?} is already on dimension {dim:?}"
                )));
            }
        }
        let mut grown_labels = existing.clone();
        grown_labels.extend(labels.iter().cloned());

        let mut grown = Vec::with_capacity(specs.len());
        let mut targets = Vec::with_capacity(specs.len());
        for ((name, config), info) in specs.iter().zip(infos) {
            let mut array = self.open_growable(name)?;
            let (col_chunk, shard_col) = column_grid(&array, name)?;
            if !width.is_multiple_of(col_chunk) {
                return Err(PbzError::Metadata(format!(
                    "append: track {name:?} is {width} column(s) wide, which is not a multiple \
                     of its column chunk {col_chunk}; only a width that is a multiple of \
                     {col_chunk} can be appended to"
                )));
            }
            if !batch.is_multiple_of(col_chunk) {
                return Err(PbzError::Metadata(format!(
                    "append: {batch} new column(s) is not a multiple of the column chunk \
                     {col_chunk} of track {name:?}"
                )));
            }
            if let Some(shard_col) = shard_col
                && shard_col != col_chunk
            {
                return Err(PbzError::Metadata(format!(
                    "append: track {name:?} has column shard {shard_col} and column chunk \
                     {col_chunk}; an append requires them equal"
                )));
            }
            array
                .set_shape(vec![total, width + batch])
                .map_err(|e| super::metadata::array_error(name, e))?;
            grown.push(grown_metadata(info, 1, width + batch));
            let array = Arc::new(array);
            targets.push(Destination {
                name: name.clone(),
                reference: Arc::clone(&shared_reference),
                dtype: config.dtype,
                values: array,
                labels: Some(grown_labels.clone()),
                column_offset: width,
            });
        }

        // Every other root variable over the appended dimension grows with
        // the targets. Only the targets receive values; the rest read their
        // new columns as fill.
        let destinations: BTreeSet<&str> = specs.iter().map(|(name, _)| name.as_str()).collect();
        for info in self.variables.values() {
            let Some(axis) = info.dims.iter().position(|d| *d == dim) else {
                continue;
            };
            if info.shape[axis] != width {
                return Err(PbzError::Metadata(format!(
                    "append: {:?} is {} wide on {dim:?} but the appended tracks are {width}",
                    info.name, info.shape[axis]
                )));
            }
            if info.name == dim || destinations.contains(info.name.as_str()) {
                continue;
            }
            if info.metadata["chunk_grid"]["name"] != "regular" {
                return Err(PbzError::UnsupportedEncoding(format!(
                    "append: {:?} does not use a regular chunk grid",
                    info.name
                )));
            }
            grown.push(grown_metadata(info, axis, width + batch));
        }

        let mut coordinate = self.open_growable(&dim)?;
        coordinate
            .set_shape(vec![width + batch])
            .map_err(|e| super::metadata::array_error(&dim, e))?;
        let (name, metadata) = grown_metadata(coordinate_info, 0, width + batch);
        Ok(PreparedAppend {
            targets,
            grown,
            coordinate: (name, metadata, Arc::new(coordinate)),
            labels,
            width,
        })
    }

    fn publish_metadata(&self, name: &str, metadata: &serde_json::Value) -> Result<()> {
        let key = StoreKey::new(format!("{name}/zarr.json"))
            .map_err(|e| PbzError::Metadata(e.to_string()))?;
        self.writable()?
            .set(
                &key,
                serde_json::to_vec(metadata)
                    .map_err(|e| PbzError::Metadata(e.to_string()))?
                    .into(),
            )
            .map_err(storage_error)
    }

    /// Publish the grown shapes, then the grown coordinate. The coordinate
    /// length is the committed column width.
    pub(crate) fn finish_append(&mut self, prepared: &PreparedAppend) -> Result<()> {
        for (name, metadata) in &prepared.grown {
            self.publish_metadata(name, metadata)?;
        }
        let (name, metadata, array) = &prepared.coordinate;
        let end = prepared.width + prepared.labels.len() as u64;
        #[allow(clippy::single_range_in_vec_init)]
        let appended = ArraySubset::new_with_ranges(&[prepared.width..end]);
        array
            .store_array_subset(&appended, ndarray::Array1::from(prepared.labels.clone()))
            .map_err(storage_error)?;
        self.publish_metadata(name, metadata)?;
        self.publish()
    }

    pub(crate) fn prepare(
        &self,
        specs: &[(String, TrackConfig)],
        plan: DatasetPlan,
    ) -> Result<Prepared> {
        self.writable()?;
        for (name, config) in specs {
            refuse_reserved(name)?;
            if let Some(dim) = &config.column_dim {
                refuse_reserved(dim)?;
            }
        }
        match plan {
            DatasetPlan::Create => self.prepare_import(specs).map(Prepared::Import),
            DatasetPlan::Append => self.prepare_append(specs).map(Prepared::Append),
        }
    }

    pub(crate) fn begin(&mut self, prepared: &Prepared) -> Result<()> {
        match prepared {
            Prepared::Import(prepared) => self.begin_import(prepared),
            Prepared::Append(_) => Ok(()),
        }
    }

    pub(crate) fn finish(&mut self, prepared: &Prepared) -> Result<()> {
        match prepared {
            Prepared::Import(prepared) => self.finish_import(prepared),
            Prepared::Append(prepared) => self.finish_append(prepared),
        }
    }
}
