use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use zarrs::array::ArrayBuilder;
use zarrs::array::data_type;
use zarrs::filesystem::FilesystemStore;
use zarrs::group::{Group, GroupBuilder};
use zarrs::storage::ReadableWritableListableStorage;

use crate::PERBASE_ZARR_VERSION;
use crate::error::{PbzError, Result};
use crate::track::{Track, TrackConfig};

/// Attribute key on the root Zarr group identifying a PBZ store.
const ROOT_ATTR_KEY: &str = "perbase_zarr";

/// Attribute key on track groups identifying them as PBZ tracks.
const TRACK_ATTR_KEY: &str = "perbase_zarr_track";

/// Layouts this implementation knows how to enumerate. Tracks with any other
/// layout are silently skipped (with a `log::warn!`) to allow forward-compat
/// readers to coexist with older writers that don't recognize new layouts.
const KNOWN_LAYOUTS: &[&str] = &[crate::track::PER_BASE_LAYOUT];

/// A PBZ (Per-Base Zarr) store.
///
/// Wraps a Zarr v3 root group and provides access to contigs and tracks.
/// Created via [`PbzStore::create`] or opened via [`PbzStore::open`].
pub struct PbzStore {
    store: ReadableWritableListableStorage,
    path: PathBuf,
    contigs: Vec<String>,
    contig_lengths: HashMap<String, u64>,
    coordinate_space: Option<String>,
}

/// Builder for [`PbzStore::create`] — required when supplying optional
/// attributes such as `coordinate_space`.
pub struct PbzStoreBuilder<'a> {
    path: &'a Path,
    contigs: Option<(&'a [String], &'a [u64])>,
    coordinate_space: Option<String>,
}

impl<'a> PbzStoreBuilder<'a> {
    pub fn new(path: &'a Path) -> Self {
        Self {
            path,
            contigs: None,
            coordinate_space: None,
        }
    }

    pub fn contigs(mut self, names: &'a [String], lengths: &'a [u64]) -> Self {
        self.contigs = Some((names, lengths));
        self
    }

    /// Optional `coordinate_space` root attribute (e.g. `"GRCh38"`).
    pub fn coordinate_space(mut self, name: impl Into<String>) -> Self {
        self.coordinate_space = Some(name.into());
        self
    }

    pub fn create(self) -> Result<PbzStore> {
        let (contigs, lengths) = self
            .contigs
            .ok_or_else(|| PbzError::Metadata("contigs are required".into()))?;
        PbzStore::create_inner(self.path, contigs, lengths, self.coordinate_space)
    }
}

impl fmt::Debug for PbzStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PbzStore")
            .field("path", &self.path)
            .field("contigs", &self.contigs)
            .field("contig_lengths", &self.contig_lengths)
            .finish_non_exhaustive()
    }
}

impl PbzStore {
    /// Create a new PBZ store on disk.
    ///
    /// This creates the Zarr v3 root group with PBZ metadata, writes the contig
    /// and contig_lengths arrays, and creates an empty `/tracks` group.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `contigs` is empty
    /// - `contigs` and `contig_lengths` have different lengths
    /// - The path already exists or cannot be created
    pub fn create(
        path: impl AsRef<Path>,
        contigs: &[String],
        contig_lengths: &[u64],
    ) -> Result<Self> {
        Self::create_inner(path.as_ref(), contigs, contig_lengths, None)
    }

    /// Builder entry point for [`PbzStore`]. Use when you need to set
    /// optional root attributes such as `coordinate_space`.
    pub fn builder(path: &Path) -> PbzStoreBuilder<'_> {
        PbzStoreBuilder::new(path)
    }

    fn create_inner(
        path: &Path,
        contigs: &[String],
        contig_lengths: &[u64],
        coordinate_space: Option<String>,
    ) -> Result<Self> {
        if contigs.is_empty() {
            return Err(PbzError::Metadata("contigs must not be empty".into()));
        }
        if contigs.len() != contig_lengths.len() {
            return Err(PbzError::Metadata(format!(
                "contigs ({}) and contig_lengths ({}) must have the same length",
                contigs.len(),
                contig_lengths.len()
            )));
        }

        let path = path.to_path_buf();
        let store: ReadableWritableListableStorage =
            Arc::new(FilesystemStore::new(&path).map_err(|e| PbzError::Store(e.to_string()))?);

        // Root group with PBZ version attribute
        // pbz[impl group.attrs.namespace]
        // pbz[impl group.attrs.version]
        let mut root_attrs = serde_json::Map::new();
        let mut pbz_meta = serde_json::json!({ "version": PERBASE_ZARR_VERSION });
        // pbz[impl group.attrs.coordinate_space]
        if let Some(ref cs) = coordinate_space {
            pbz_meta["coordinate_space"] = serde_json::Value::String(cs.clone());
        }
        root_attrs.insert(ROOT_ATTR_KEY.into(), pbz_meta);

        let group = GroupBuilder::new()
            .attributes(root_attrs)
            .build(store.clone(), "/")
            .map_err(|e| PbzError::Store(e.to_string()))?;
        group
            .store_metadata()
            .map_err(|e| PbzError::Store(e.to_string()))?;

        // /contigs — variable-length string array
        // pbz[impl contigs.array]
        // pbz[impl contigs.order]
        // pbz[impl coords.contig-name-match]
        let num_contigs = contigs.len() as u64;

        let contigs_array = ArrayBuilder::new(
            vec![num_contigs],
            vec![num_contigs],
            data_type::string(),
            "",
        )
        .build(store.clone(), "/contigs")
        .map_err(|e| PbzError::Store(e.to_string()))?;

        contigs_array
            .store_metadata()
            .map_err(|e| PbzError::Store(e.to_string()))?;
        let contig_strings: Vec<String> = contigs.to_vec();
        contigs_array
            .store_chunk(&[0], contig_strings)
            .map_err(|e| PbzError::Store(e.to_string()))?;

        // /contig_lengths — int64 array
        // pbz[impl contigs.lengths]
        let lengths_array = ArrayBuilder::new(
            vec![num_contigs],
            vec![num_contigs],
            data_type::int64(),
            0i64,
        )
        .build(store.clone(), "/contig_lengths")
        .map_err(|e| PbzError::Store(e.to_string()))?;

        lengths_array
            .store_metadata()
            .map_err(|e| PbzError::Store(e.to_string()))?;
        let length_values: Vec<i64> = contig_lengths.iter().map(|&l| l as i64).collect();
        lengths_array
            .store_chunk(&[0], length_values)
            .map_err(|e| PbzError::Store(e.to_string()))?;

        // /tracks — empty group
        let tracks_group = GroupBuilder::new()
            .build(store.clone(), "/tracks")
            .map_err(|e| PbzError::Store(e.to_string()))?;
        tracks_group
            .store_metadata()
            .map_err(|e| PbzError::Store(e.to_string()))?;

        let contig_lengths_map: HashMap<String, u64> = contigs
            .iter()
            .zip(contig_lengths.iter())
            .map(|(c, &l)| (c.clone(), l))
            .collect();

        Ok(Self {
            store,
            path,
            contigs: contigs.to_vec(),
            contig_lengths: contig_lengths_map,
            coordinate_space,
        })
    }

    /// Open an existing PBZ store from disk.
    ///
    /// Validates the root group has a `perbase_zarr` attribute with a `version`
    /// field, then reads and caches the contig arrays.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The path does not contain a valid Zarr v3 store
    /// - The root group is missing the `perbase_zarr` attribute
    /// - The contig arrays cannot be read
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let store: ReadableWritableListableStorage =
            Arc::new(FilesystemStore::new(&path).map_err(|e| PbzError::Store(e.to_string()))?);

        // Validate root metadata
        let root = Group::open(store.clone(), "/").map_err(|e| PbzError::Store(e.to_string()))?;

        let pbz_meta = root.attributes().get(ROOT_ATTR_KEY).ok_or_else(|| {
            PbzError::Metadata(format!(
                "root group missing '{ROOT_ATTR_KEY}' attribute — not a PBZ store"
            ))
        })?;

        // Validate version field exists (we don't enforce specific versions yet)
        pbz_meta
            .get("version")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                PbzError::Metadata("root 'perbase_zarr' attribute missing 'version' field".into())
            })?;

        // Optional coordinate_space attribute (added in spec v0.1).
        let coordinate_space = pbz_meta
            .get("coordinate_space")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        // Read /contigs array
        let contigs_array = zarrs::array::Array::open(store.clone(), "/contigs")
            .map_err(|e| PbzError::Store(e.to_string()))?;

        let contigs: Vec<String> = contigs_array
            .retrieve_chunk::<Vec<String>>(&[0])
            .map_err(|e| PbzError::Store(e.to_string()))?;

        // Read /contig_lengths array
        let lengths_array = zarrs::array::Array::open(store.clone(), "/contig_lengths")
            .map_err(|e| PbzError::Store(e.to_string()))?;

        let lengths: Vec<i64> = lengths_array
            .retrieve_chunk::<Vec<i64>>(&[0])
            .map_err(|e| PbzError::Store(e.to_string()))?;

        if contigs.len() != lengths.len() {
            return Err(PbzError::Metadata(format!(
                "contigs ({}) and contig_lengths ({}) arrays have different lengths",
                contigs.len(),
                lengths.len()
            )));
        }

        let contig_lengths: HashMap<String, u64> = contigs
            .iter()
            .zip(lengths.iter())
            .map(|(c, &l)| (c.clone(), l as u64))
            .collect();

        Ok(Self {
            store,
            path,
            contigs,
            contig_lengths,
            coordinate_space,
        })
    }

    /// Optional `coordinate_space` root attribute (e.g. `"GRCh38"`).
    /// Returns `None` if the store was created without one.
    pub fn coordinate_space(&self) -> Option<&str> {
        self.coordinate_space.as_deref()
    }

    /// The filesystem path of this store.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The underlying zarrs storage handle.
    ///
    /// Escape hatch for advanced usage — callers can use this to interact
    /// with the Zarr store directly via the `zarrs` API.
    pub fn storage(&self) -> &ReadableWritableListableStorage {
        &self.store
    }

    /// Ordered list of contig names in this store.
    pub fn contigs(&self) -> &[String] {
        &self.contigs
    }

    /// Map of contig name to length (in base pairs).
    pub fn contig_lengths(&self) -> &HashMap<String, u64> {
        &self.contig_lengths
    }

    /// Get the length of a specific contig.
    pub fn contig_length(&self, name: &str) -> Result<u64> {
        self.contig_lengths
            .get(name)
            .copied()
            .ok_or_else(|| PbzError::ContigNotFound {
                contig: name.into(),
                available: self.contigs.clone(),
            })
    }

    /// Validate that a contig name exists in this store.
    pub fn validate_contig(&self, name: &str) -> Result<()> {
        if self.contig_lengths.contains_key(name) {
            Ok(())
        } else {
            Err(PbzError::ContigNotFound {
                contig: name.into(),
                available: self.contigs.clone(),
            })
        }
    }

    /// Create a new track in this store.
    ///
    /// Errors with [`PbzError::Metadata`] if a track with this name already
    /// exists. To replace a track, drop it first via
    /// [`PbzStore::drop_track`].
    pub fn create_track(&self, name: &str, config: TrackConfig) -> Result<Track> {
        if self.tracks()?.iter().any(|t| t == name) {
            return Err(PbzError::Metadata(format!("track '{name}' already exists")));
        }
        Track::create(self.store.clone(), name, &config, &self.contig_lengths)
    }

    /// Open an existing track by name.
    ///
    /// Returns `TrackNotFound` if the track does not exist.
    pub fn track(&self, name: &str) -> Result<Track> {
        let available = self.tracks()?;
        if !available.contains(&name.to_string()) {
            return Err(PbzError::TrackNotFound {
                name: name.into(),
                available,
            });
        }
        Track::open(self.store.clone(), name, &self.contig_lengths)
    }

    /// Recursively delete a track group. Errors if `name` doesn't exist.
    pub fn drop_track(&self, name: &str) -> Result<()> {
        let _ = self.track(name)?;
        let path = self.path.join("tracks").join(name);
        fs_err::remove_dir_all(&path)
            .map_err(|e| PbzError::Store(format!("remove_dir_all failed: {e}")))?;
        Ok(())
    }

    /// Rename a track group on disk. Errors if `old` doesn't exist or `new` already does.
    pub fn rename_track(&self, old: &str, new: &str) -> Result<()> {
        let _ = self.track(old)?;
        let old_path = self.path.join("tracks").join(old);
        let new_path = self.path.join("tracks").join(new);
        // Cheaper than `self.tracks()` (which walks the whole zarr hierarchy
        // again); we only need to know whether the destination exists.
        if fs_err::metadata(&new_path).is_ok() {
            return Err(PbzError::Metadata(format!("track '{new}' already exists")));
        }
        fs_err::rename(&old_path, &new_path)
            .map_err(|e| PbzError::Store(format!("rename failed: {e}")))?;
        Ok(())
    }

    /// List all track names in this store.
    ///
    /// Walks the `/tracks/` group for child groups that have the
    /// `perbase_zarr_track` attribute. Returns track names relative to
    /// `/tracks/` (e.g. `"depths"`, `"masks/callable"`).
    pub fn tracks(&self) -> Result<Vec<String>> {
        let tracks_group = Group::open(self.store.clone(), "/tracks")
            .map_err(|e| PbzError::Store(e.to_string()))?;

        let mut track_names = Vec::new();
        collect_tracks(&tracks_group, "", &mut track_names)?;
        track_names.sort();
        Ok(track_names)
    }
}

/// Recursively walk child groups looking for ones with the track attribute.
// pbz[impl group.tracks.nesting]
fn collect_tracks<T>(group: &Group<T>, prefix: &str, out: &mut Vec<String>) -> Result<()>
where
    T: zarrs::storage::ReadableStorageTraits + zarrs::storage::ListableStorageTraits + ?Sized,
{
    let children = group
        .child_groups()
        .map_err(|e| PbzError::Store(e.to_string()))?;

    for child in children {
        let child_path_str = child.path().as_str();

        // The child path is absolute (e.g. "/tracks/depths"), extract the name part
        let name_part = child_path_str
            .rsplit_once('/')
            .map(|(_, n)| n)
            .unwrap_or(child_path_str);

        let relative_name = if prefix.is_empty() {
            name_part.to_string()
        } else {
            format!("{prefix}/{name_part}")
        };

        if let Some(meta) = child.attributes().get(TRACK_ATTR_KEY) {
            // pbz[impl forward.layout.no-error]
            // pbz[impl forward.layout.warn-skip]
            let layout = meta.get("layout").and_then(|v| v.as_str());
            match layout {
                None => {
                    log::warn!("track {relative_name:?} missing 'layout' attribute; skipping");
                }
                Some(l) if !KNOWN_LAYOUTS.contains(&l) => {
                    log::warn!("track {relative_name:?} has unrecognized layout {l:?}; skipping");
                }
                Some(_) => {
                    out.push(relative_name.clone());
                }
            }
        }

        // Recurse into subgroups to find nested tracks
        collect_tracks(&child, &relative_name, out)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::track::TrackConfig;
    use tempfile::TempDir;

    fn test_contigs() -> (Vec<String>, Vec<u64>) {
        let names = vec!["chr1".to_string(), "chr2".to_string()];
        let lengths = vec![248_956_422, 242_193_529];
        (names, lengths)
    }

    #[test]
    fn create_and_reopen() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.pbz.zarr");
        let (contigs, lengths) = test_contigs();

        PbzStore::create(&path, &contigs, &lengths).unwrap();
        assert!(path.exists());

        let store = PbzStore::open(&path).unwrap();
        assert_eq!(store.contigs(), &contigs);
        assert_eq!(store.contig_length("chr1").unwrap(), 248_956_422);
        assert_eq!(store.contig_length("chr2").unwrap(), 242_193_529);
    }

    #[test]
    fn contigs_accessor() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.pbz.zarr");
        let (contigs, lengths) = test_contigs();

        let store = PbzStore::create(&path, &contigs, &lengths).unwrap();
        assert_eq!(store.contigs(), &contigs);
        assert_eq!(store.contig_lengths().len(), 2);
    }

    #[test]
    fn contig_length_missing() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.pbz.zarr");
        let (contigs, lengths) = test_contigs();

        let store = PbzStore::create(&path, &contigs, &lengths).unwrap();

        let err = store.contig_length("chrX").unwrap_err();
        match err {
            PbzError::ContigNotFound { contig, available } => {
                assert_eq!(contig, "chrX");
                assert_eq!(available, contigs);
            }
            other => panic!("expected ContigNotFound, got: {other}"),
        }
    }

    #[test]
    fn validate_contig_ok() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.pbz.zarr");
        let (contigs, lengths) = test_contigs();

        let store = PbzStore::create(&path, &contigs, &lengths).unwrap();
        store.validate_contig("chr1").unwrap();
        store.validate_contig("chr2").unwrap();
    }

    #[test]
    fn validate_contig_missing() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.pbz.zarr");
        let (contigs, lengths) = test_contigs();

        let store = PbzStore::create(&path, &contigs, &lengths).unwrap();
        assert!(store.validate_contig("chrX").is_err());
    }

    #[test]
    fn empty_tracks() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.pbz.zarr");
        let (contigs, lengths) = test_contigs();

        let store = PbzStore::create(&path, &contigs, &lengths).unwrap();
        let tracks = store.tracks().unwrap();
        assert!(tracks.is_empty());
    }

    #[test]
    fn err_empty_contigs() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.pbz.zarr");

        let err = PbzStore::create(&path, &[], &[]).unwrap_err();
        assert!(matches!(err, PbzError::Metadata(_)));
    }

    #[test]
    fn err_mismatched_lengths() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.pbz.zarr");

        let err = PbzStore::create(&path, &["chr1".into()], &[100, 200]).unwrap_err();
        assert!(matches!(err, PbzError::Metadata(_)));
    }

    #[test]
    fn err_open_not_pbz() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("not_pbz.zarr");

        // Create a plain zarr group without PBZ metadata
        let store: ReadableWritableListableStorage = Arc::new(FilesystemStore::new(&path).unwrap());
        let group = GroupBuilder::new().build(store.clone(), "/").unwrap();
        group.store_metadata().unwrap();

        let err = PbzStore::open(&path).unwrap_err();
        assert!(matches!(err, PbzError::Metadata(_)));
    }

    #[test]
    fn path_accessor() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.pbz.zarr");
        let (contigs, lengths) = test_contigs();

        let store = PbzStore::create(&path, &contigs, &lengths).unwrap();
        assert_eq!(store.path(), path);
    }

    #[test]
    fn tracks_with_manual_track_group() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.pbz.zarr");
        let (contigs, lengths) = test_contigs();

        let store = PbzStore::create(&path, &contigs, &lengths).unwrap();

        // Manually create a track group to test tracks() discovery
        let mut track_attrs = serde_json::Map::new();
        track_attrs.insert(
            TRACK_ATTR_KEY.into(),
            serde_json::json!({
                "layout": "per_base",
                "dtype": "uint32",
                "chunk_size": 1_000_000,
                "column_chunk_size": 16,
            }),
        );

        let group = GroupBuilder::new()
            .attributes(track_attrs)
            .build(store.storage().clone(), "/tracks/depths")
            .unwrap();
        group.store_metadata().unwrap();

        let tracks = store.tracks().unwrap();
        assert_eq!(tracks, vec!["depths"]);
    }

    // pbz[verify group.tracks.nesting]
    #[test]
    fn tracks_nested() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.pbz.zarr");
        let (contigs, lengths) = test_contigs();

        let store = PbzStore::create(&path, &contigs, &lengths).unwrap();

        // Create /tracks/masks/ (a non-track intermediate group)
        let masks_group = GroupBuilder::new()
            .build(store.storage().clone(), "/tracks/masks")
            .unwrap();
        masks_group.store_metadata().unwrap();

        // Create /tracks/masks/callable (a nested track)
        let mut track_attrs = serde_json::Map::new();
        track_attrs.insert(
            TRACK_ATTR_KEY.into(),
            serde_json::json!({
                "layout": "per_base",
                "dtype": "bool",
                "chunk_size": 1_000_000,
            }),
        );

        let callable_group = GroupBuilder::new()
            .attributes(track_attrs)
            .build(store.storage().clone(), "/tracks/masks/callable")
            .unwrap();
        callable_group.store_metadata().unwrap();

        let tracks = store.tracks().unwrap();
        assert_eq!(tracks, vec!["masks/callable"]);
    }

    #[test]
    fn create_and_get_track() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.pbz.zarr");
        let (contigs, lengths) = test_contigs();

        let store = PbzStore::create(&path, &contigs, &lengths).unwrap();

        let config = TrackConfig {
            dtype: "uint32".into(),
            columns: Some(vec!["s1".into(), "s2".into()]),
            chunk_size: 1_000_000,
            ..Default::default()
        };
        let track = store.create_track("depths", config).unwrap();
        assert_eq!(track.name(), "depths");

        let track_names = store.tracks().unwrap();
        assert_eq!(track_names, vec!["depths"]);

        let track2 = store.track("depths").unwrap();
        assert_eq!(track2.metadata().dtype, "uint32");
        assert!(track2.has_columns());
    }

    #[test]
    fn track_not_found() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.pbz.zarr");
        let (contigs, lengths) = test_contigs();

        let store = PbzStore::create(&path, &contigs, &lengths).unwrap();
        let err = store.track("nonexistent").unwrap_err();
        assert!(matches!(err, PbzError::TrackNotFound { .. }));
    }

    #[test]
    fn rename_track_round_trip() {
        let dir = TempDir::new().unwrap();
        let store_path = dir.path().join("s.pbz.zarr");
        let store = PbzStore::create(&store_path, &["chr1".to_string()], &[100u64]).unwrap();
        let cfg = crate::track::TrackConfig {
            dtype: "uint32".into(),
            columns: None,
            chunk_size: 100,
            column_chunk_size: 1,
            description: None,
            source: None,
            extra: serde_json::Map::new(),
            column_dim_name: None,
        };
        store.create_track("old", cfg).unwrap();
        store.rename_track("old", "new").unwrap();

        let mut tracks = store.tracks().unwrap();
        tracks.sort();
        assert_eq!(tracks, vec!["new"]);
        assert!(store.track("new").is_ok());
        assert!(matches!(
            store.track("old"),
            Err(PbzError::TrackNotFound { .. })
        ));
    }

    #[test]
    fn rename_track_old_missing() {
        let dir = TempDir::new().unwrap();
        let store_path = dir.path().join("s.pbz.zarr");
        let store = PbzStore::create(&store_path, &["chr1".to_string()], &[100u64]).unwrap();
        let err = store.rename_track("missing", "new").unwrap_err();
        assert!(matches!(err, PbzError::TrackNotFound { .. }));
    }

    #[test]
    fn rename_track_new_already_exists() {
        let dir = TempDir::new().unwrap();
        let store_path = dir.path().join("s.pbz.zarr");
        let store = PbzStore::create(&store_path, &["chr1".to_string()], &[100u64]).unwrap();
        let cfg_a = crate::track::TrackConfig {
            dtype: "uint32".into(),
            columns: None,
            chunk_size: 100,
            column_chunk_size: 1,
            description: None,
            source: None,
            extra: serde_json::Map::new(),
            column_dim_name: None,
        };
        let cfg_b = crate::track::TrackConfig {
            dtype: "uint32".into(),
            columns: None,
            chunk_size: 100,
            column_chunk_size: 1,
            description: None,
            source: None,
            extra: serde_json::Map::new(),
            column_dim_name: None,
        };
        store.create_track("a", cfg_a).unwrap();
        store.create_track("b", cfg_b).unwrap();
        let err = store.rename_track("a", "b").unwrap_err();
        assert!(matches!(err, PbzError::Metadata(_)));
    }

    #[test]
    fn drop_track_removes_directory() {
        let dir = TempDir::new().unwrap();
        let store_path = dir.path().join("s.pbz.zarr");
        let store = PbzStore::create(&store_path, &["chr1".to_string()], &[100u64]).unwrap();
        let cfg = crate::track::TrackConfig {
            dtype: "uint32".into(),
            columns: None,
            chunk_size: 100,
            column_chunk_size: 1,
            description: None,
            source: None,
            extra: serde_json::Map::new(),
            column_dim_name: None,
        };
        store.create_track("t", cfg).unwrap();
        assert!(store_path.join("tracks/t").exists());

        store.drop_track("t").unwrap();
        assert!(!store_path.join("tracks/t").exists());
        assert!(store.tracks().unwrap().is_empty());
    }

    #[test]
    fn drop_track_missing() {
        let dir = TempDir::new().unwrap();
        let store_path = dir.path().join("s.pbz.zarr");
        let store = PbzStore::create(&store_path, &["chr1".to_string()], &[100u64]).unwrap();
        let err = store.drop_track("nope").unwrap_err();
        assert!(matches!(err, PbzError::TrackNotFound { .. }));
    }
}
