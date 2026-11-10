//! Where a format entry point writes: a format-0.4 store, a format-0.5
//! root Dataset, or a path that holds one or will hold a new one.

use std::path::Path;

use log::{debug, info};

use super::builder::{Import, ImportBuilder};
use super::config::Report;
use super::engine::PipelineOptions;
use super::routing::SourceAxis;
use crate::dataset::{Dataset, DatasetPlan, Reference};
use crate::io::ValueReader;
use crate::{Contig, Genome, PbzError, PbzStore, Result, TrackConfig};

/// A destination for one import run.
///
/// Every variant takes the same track specifications and the same [`Axes`]:
/// `Store` writes one Zarr group per track, `Dataset` writes new root
/// variables over shared root coordinates, and `Append` widens root variables
/// that already exist. The format entry points accept anything that converts
/// into this, so `&mut store` and `&mut dataset` both pass directly.
pub enum Target<'a> {
    Store(&'a mut PbzStore),
    Dataset(&'a mut Dataset),
    /// A format-0.5 Dataset whose existing root variables grow along their
    /// column dimension. [`Target::import`] routes this variant to the append
    /// path, so every format entry point appends without knowing about it.
    Append(&'a mut Dataset),
    /// A store path. When the path does not exist, the import creates a
    /// format-0.5 store whose reference is the genome of the checked
    /// sources. When it exists, the import adds new tracks to that store.
    Path(&'a Path),
}

impl<'a> From<&'a Path> for Target<'a> {
    fn from(path: &'a Path) -> Self {
        Self::Path(path)
    }
}

impl<'a> From<&'a mut PbzStore> for Target<'a> {
    fn from(store: &'a mut PbzStore) -> Self {
        Self::Store(store)
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
        specs: Vec<(String, Genome, TrackConfig)>,
        readers: Vec<R>,
        axes: Axes,
        options: PipelineOptions,
    ) -> Result<Report> {
        match self {
            Self::Store(store) => store.create_tracks_with(specs, move |tracks| {
                axes.apply(Import::from_readers(readers)?.into_tracks(tracks))
                    .options(options)
                    .run()
            }),
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
        specs: Vec<(String, Genome, TrackConfig)>,
        readers: Vec<R>,
        axes: Axes,
        options: PipelineOptions,
    ) -> Result<Report> {
        match self {
            Self::Store(_) => Err(PbzError::LegacyFormat),
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
/// takes its reference from the first spec, whose genome the format entry
/// point has already checked against every source. A failed import into a
/// new store removes that store.
fn import_at<R: ValueReader>(
    path: &Path,
    specs: Vec<(String, Genome, TrackConfig)>,
    readers: Vec<R>,
    axes: Axes,
    options: PipelineOptions,
) -> Result<Report> {
    if path.exists() {
        info!("{} exists; adding track(s) to the store", path.display());
        return match Dataset::open(path) {
            Ok(mut dataset) => Target::Dataset(&mut dataset).import(specs, readers, axes, options),
            Err(PbzError::LegacyFormat) => {
                let mut store = PbzStore::open(path)?;
                Target::Store(&mut store).import(specs, readers, axes, options)
            }
            Err(error) => Err(error),
        };
    }
    let (_, genome, _) = specs
        .first()
        .ok_or_else(|| PbzError::Metadata("import: no destination tracks".into()))?;
    let reference = Reference::new(genome.contigs().to_vec())?;
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
pub fn reference_genome(path: &Path, genome: Option<Genome>) -> Result<Genome> {
    if !path.exists() {
        return genome.ok_or_else(|| {
            PbzError::Metadata(format!(
                "a genome file is needed to create the store {}",
                path.display()
            ))
        });
    }
    let dataset = match Dataset::open(path) {
        Ok(dataset) => dataset,
        Err(PbzError::LegacyFormat) => return genome.ok_or(PbzError::LegacyFormat),
        Err(error) => return Err(error),
    };
    let store = dataset.reference()?.contigs().to_vec();
    if let Some(genome) = &genome {
        same_reference(genome.contigs(), &store)?;
    }
    Genome::new(store)
}

fn dataset_run<R: ValueReader>(
    dataset: &mut Dataset,
    specs: Vec<(String, Genome, TrackConfig)>,
    readers: Vec<R>,
    axes: Axes,
    options: PipelineOptions,
    plan: DatasetPlan,
) -> Result<Report> {
    if plan == DatasetPlan::Append {
        let reference = dataset.reference()?.contigs().to_vec();
        for (_, genome, _) in &specs {
            same_reference(genome.contigs(), &reference)?;
        }
    }
    // A Dataset fixes its reference at creation, so the per-spec genome is
    // dropped here; routing checks every reader against the reference before
    // the run writes anything.
    let specs = specs
        .into_iter()
        .map(|(name, _, cfg)| (name, cfg))
        .collect();
    axes.apply(Import::from_readers(readers)?.into_dataset_plan(dataset, specs, plan))
        .options(options)
        .run()
}

/// An append reads the store's own geometry, so the source must describe the
/// same reference in the same order.
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
