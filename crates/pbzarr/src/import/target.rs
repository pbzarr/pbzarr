//! Where a format entry point writes: a format-0.5 root Dataset, or a path
//! that holds one or will hold a new one.

use std::path::Path;

use log::{debug, info};

use super::builder::{Import, ImportBuilder};
use super::config::{Config, Report};
use super::routing::SourceAxis;
use crate::dataset::{Dataset, DatasetPlan, Reference};
use crate::io::ValueReader;
use crate::{Contig, PbzError, Result, TrackConfig};

/// A destination for one import run.
///
/// Every variant takes the same track specifications and the same [`Axes`]:
/// `Dataset` writes new root variables over shared root coordinates, and
/// `Append` widens root variables that already exist. The format entry points
/// accept anything that converts into this, so `&mut dataset` and a path both
/// pass directly.
pub enum Target<'a> {
    Dataset(&'a mut Dataset),
    /// A format-0.5 Dataset whose existing root variables grow along their
    /// column dimension. [`Target::import`] routes this variant to the append
    /// path, so every format entry point appends without knowing about it.
    ///
    /// Every other root variable over that dimension grows with them, and its
    /// new columns read as the Zarr fill value. A variable with no such
    /// dimension, such as a reduction across columns, keeps a legal shape and
    /// goes stale; nothing detects or marks that.
    Append(&'a mut Dataset),
    /// A store path. When the path does not exist, the import creates a
    /// format-0.5 store whose reference is the contig list of the sources.
    /// When it exists, the import adds new tracks to that store.
    Path(&'a Path),
    /// A new single-file zip store at a path that must not exist. The import
    /// writes a temporary file beside the path and moves it there at the end,
    /// so a failed import leaves no file. A track with no shard size and no
    /// codecs gets shards of 64 position chunks and all its columns. A track
    /// whose codecs have no sharding codec is refused.
    Zip(&'a Path),
}

impl<'a> From<&'a Path> for Target<'a> {
    fn from(path: &'a Path) -> Self {
        Self::Path(path)
    }
}

impl<'a> From<&'a mut Dataset> for Target<'a> {
    fn from(dataset: &'a mut Dataset) -> Self {
        Self::Dataset(dataset)
    }
}

/// Which source axis fans out over tracks and which over the column
/// dimension: the part of an import request that does not depend on the
/// destination. The default states neither axis, which is one track of
/// scalar width.
#[derive(Default)]
pub struct Axes {
    tracks_from_fields: bool,
    columns_from: Option<(SourceAxis, Vec<String>)>,
}

impl Axes {
    /// Schema field `i` becomes track `i`.
    pub fn fields_as_tracks(mut self) -> Self {
        self.tracks_from_fields = true;
        self
    }

    /// Reader `i` becomes column `i`, over the coordinate `labels`.
    pub fn readers_as_columns(mut self, labels: Vec<String>) -> Self {
        self.columns_from = Some((SourceAxis::Readers, labels));
        self
    }

    /// Schema field `i` becomes column `i`, over the coordinate `labels`.
    pub fn fields_as_columns(mut self, labels: Vec<String>) -> Self {
        self.columns_from = Some((SourceAxis::Fields, labels));
        self
    }

    fn apply<'t, R: ValueReader>(self, builder: ImportBuilder<'t, R>) -> ImportBuilder<'t, R> {
        let builder = if self.tracks_from_fields {
            builder.fields_as_tracks()
        } else {
            builder
        };
        match self.columns_from {
            Some((SourceAxis::Readers, labels)) => {
                builder.readers_as_columns().expect_column_labels(labels)
            }
            Some((SourceAxis::Fields, labels)) => {
                builder.fields_as_columns().expect_column_labels(labels)
            }
            None => builder,
        }
    }
}

impl Target<'_> {
    /// Create `specs` at this destination and populate them from `readers`.
    pub fn import<R: ValueReader>(
        self,
        specs: Vec<(String, TrackConfig)>,
        readers: Vec<R>,
        axes: Axes,
        config: Config,
    ) -> Result<Report> {
        match self {
            Self::Dataset(dataset) => {
                dataset_run(dataset, specs, readers, axes, config, DatasetPlan::Create)
            }
            Self::Append(dataset) => {
                dataset_run(dataset, specs, readers, axes, config, DatasetPlan::Append)
            }
            Self::Path(path) => import_at(path, specs, readers, axes, config),
            Self::Zip(path) => import_zip(path, specs, readers, axes, config),
        }
    }
}

/// Import into the store at `path`, or into a new store there. A new store
/// takes its reference from the first source. A failed import into a new
/// store removes that store.
fn import_at<R: ValueReader>(
    path: &Path,
    specs: Vec<(String, TrackConfig)>,
    readers: Vec<R>,
    axes: Axes,
    config: Config,
) -> Result<Report> {
    refuse_zip(path)?;
    if path.exists() {
        info!("{} exists; adding track(s) to the store", path.display());
        let mut dataset = Dataset::open(path)?;
        return Target::Dataset(&mut dataset).import(specs, readers, axes, config);
    }
    let reference = readers
        .first()
        .ok_or_else(|| PbzError::Metadata("import: no sources".into()))?
        .contigs()
        .clone();
    debug!("creating store {}", path.display());
    let mut dataset = Dataset::create(path, reference)?;
    let report = Target::Dataset(&mut dataset).import(specs, readers, axes, config);
    if report.is_err() {
        drop(dataset);
        if let Err(error) = std::fs::remove_dir_all(path) {
            debug!("remove failed store {}: {error}", path.display());
        }
    }
    report
}

/// Position chunks in one shard of a zip store track that names no shard.
const ZIP_SHARD_CHUNKS: u64 = 64;

fn import_zip<R: ValueReader>(
    path: &Path,
    specs: Vec<(String, TrackConfig)>,
    readers: Vec<R>,
    axes: Axes,
    config: Config,
) -> Result<Report> {
    let reference = readers
        .first()
        .ok_or_else(|| PbzError::Metadata("import: no sources".into()))?
        .contigs()
        .clone();
    let specs = zip_layout(specs, &reference)?;
    debug!("creating zip store {}", path.display());
    let mut dataset = Dataset::create_zip(path, reference)?;
    let report = Target::Dataset(&mut dataset).import(specs, readers, axes, config)?;
    dataset.close()?;
    Ok(report)
}

/// Give every track of a zip store a sharded layout, so that the archive
/// holds few entries.
fn zip_layout(
    specs: Vec<(String, TrackConfig)>,
    reference: &Reference,
) -> Result<Vec<(String, TrackConfig)>> {
    let total: u64 = reference.contigs().iter().map(|c| c.length).sum();
    specs
        .into_iter()
        .map(|(name, mut track)| {
            match &track.codecs {
                Some(spec) => {
                    if !spec.codecs.iter().any(|c| c.name() == "sharding_indexed") {
                        return Err(PbzError::UnsupportedEncoding(format!(
                            "track {name:?}: a zip store needs sharded arrays, \
                             but the codecs have no sharding_indexed codec"
                        )));
                    }
                }
                None if track.shard_size.is_none() => {
                    let chunk = (track.chunk_size as u64).min(total.max(1)).max(1);
                    track.shard_size = Some((chunk * ZIP_SHARD_CHUNKS) as usize);
                    if track.shard_column_size.is_none()
                        && let Some(columns) = &track.columns
                    {
                        let width = columns.len().max(1);
                        let column_chunk = track.column_chunk_size.unwrap_or(width).clamp(1, width);
                        track.shard_column_size = Some(width.div_ceil(column_chunk) * column_chunk);
                    }
                }
                None => {}
            }
            Ok((name, track))
        })
        .collect()
}

/// A zip store is written once, so no import adds tracks to it.
fn refuse_zip(path: &Path) -> Result<()> {
    if crate::is_zip_store(path) {
        return Err(PbzError::Store(format!(
            "{} is a zip store. Zip stores are written once, so nothing adds tracks to them",
            path.display()
        )));
    }
    Ok(())
}

/// The contig list for an import into `path` whose sources carry no contig
/// lengths, such as BED. An existing store gives its own reference, and a
/// `genome` given with it must match that reference. A new store needs
/// `genome`.
pub fn store_reference(path: &Path, genome: Option<Reference>) -> Result<Reference> {
    refuse_zip(path)?;
    if !path.exists() {
        return genome.ok_or_else(|| {
            PbzError::Metadata(format!(
                "a genome file is needed to create the store {}",
                path.display()
            ))
        });
    }
    let dataset = Dataset::open(path)?;
    let store = dataset.reference()?;
    if let Some(genome) = &genome {
        same_reference(genome.contigs(), store.contigs())?;
    }
    Ok(store.clone())
}

fn dataset_run<R: ValueReader>(
    dataset: &mut Dataset,
    specs: Vec<(String, TrackConfig)>,
    readers: Vec<R>,
    axes: Axes,
    config: Config,
    plan: DatasetPlan,
) -> Result<Report> {
    let store = dataset.reference()?;
    for reader in &readers {
        same_reference(reader.contigs().contigs(), store.contigs())?;
    }
    axes.apply(Import::from_readers(readers)?.into_dataset_plan(dataset, specs, plan))
        .config(config)
        .run()
}

/// The store fixes the position axis, so every source must describe the
/// same contigs in the same order.
fn same_reference(source: &[Contig], store: &[Contig]) -> Result<()> {
    if source == store {
        return Ok(());
    }
    let message = source
        .iter()
        .zip(store)
        .enumerate()
        .find(|(_, (a, b))| a != b)
        .map(|(index, (source, store))| {
            format!(
                "contig {index} is {}:{} in the source but {}:{} in the store",
                source.name, source.length, store.name, store.length
            )
        })
        .unwrap_or_else(|| {
            format!(
                "the source has {} contig(s) and the store has {}",
                source.len(),
                store.len()
            )
        });
    Err(PbzError::Metadata(format!(
        "source reference differs from the store reference: {message}"
    )))
}
