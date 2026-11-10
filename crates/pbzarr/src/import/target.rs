//! Where a format entry point writes: a format-0.5 root Dataset, or a path
//! that holds one or will hold a new one.

use std::path::Path;

use log::{debug, info};

use super::builder::{Import, ImportBuilder};
use super::config::Report;
use super::engine::PipelineOptions;
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
    Append(&'a mut Dataset),
    /// A store path. When the path does not exist, the import creates a
    /// format-0.5 store whose reference is the contig list of the sources.
    /// When it exists, the import adds new tracks to that store.
    Path(&'a Path),
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
        options: PipelineOptions,
    ) -> Result<Report> {
        match self {
            Self::Dataset(dataset) => {
                dataset_run(dataset, specs, readers, axes, options, DatasetPlan::Create)
            }
            Self::Append(dataset) => {
                dataset_run(dataset, specs, readers, axes, options, DatasetPlan::Append)
            }
            Self::Path(path) => import_at(path, specs, readers, axes, options),
        }
    }

    /// Grow the `specs` tracks at this destination along their column
    /// dimension and fill the new columns from `readers`.
    ///
    /// Every other root variable over that dimension grows with them, and its
    /// new columns read as the Zarr fill value. A variable with no such
    /// dimension, such as a reduction across columns, keeps a legal shape and
    /// goes stale; nothing detects or marks that.
    pub fn append<R: ValueReader>(
        self,
        specs: Vec<(String, TrackConfig)>,
        readers: Vec<R>,
        axes: Axes,
        options: PipelineOptions,
    ) -> Result<Report> {
        match self {
            Self::Path(path) => {
                let mut dataset = Dataset::open(path)?;
                dataset_run(
                    &mut dataset,
                    specs,
                    readers,
                    axes,
                    options,
                    DatasetPlan::Append,
                )
            }
            Self::Dataset(dataset) | Self::Append(dataset) => {
                dataset_run(dataset, specs, readers, axes, options, DatasetPlan::Append)
            }
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
    options: PipelineOptions,
) -> Result<Report> {
    if path.exists() {
        info!("{} exists; adding track(s) to the store", path.display());
        let mut dataset = Dataset::open(path)?;
        return Target::Dataset(&mut dataset).import(specs, readers, axes, options);
    }
    let reference = readers
        .first()
        .ok_or_else(|| PbzError::Metadata("import: no sources".into()))?
        .contigs()
        .clone();
    debug!("creating store {}", path.display());
    let mut dataset = Dataset::create(path, reference)?;
    let report = Target::Dataset(&mut dataset).import(specs, readers, axes, options);
    if report.is_err() {
        drop(dataset);
        if let Err(error) = std::fs::remove_dir_all(path) {
            debug!("remove failed store {}: {error}", path.display());
        }
    }
    report
}

/// The contig list for an import into `path` whose sources carry no contig
/// lengths, such as BED. An existing store gives its own reference, and a
/// `genome` given with it must match that reference. A new store needs
/// `genome`.
pub fn store_reference(path: &Path, genome: Option<Reference>) -> Result<Reference> {
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
    options: PipelineOptions,
    plan: DatasetPlan,
) -> Result<Report> {
    let store = dataset.reference()?;
    for reader in &readers {
        same_reference(reader.contigs().contigs(), store.contigs())?;
    }
    axes.apply(Import::from_readers(readers)?.into_dataset_plan(dataset, specs, plan))
        .options(options)
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
