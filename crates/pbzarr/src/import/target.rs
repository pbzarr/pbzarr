//! Where a format entry point writes: a format-0.4 store, or a format-0.5
//! root Dataset.

use super::builder::{Import, ImportBuilder};
use super::config::{Config, Report};
use super::engine::PipelineOptions;
use super::routing::SourceAxis;
use crate::dataset::{Dataset, DatasetPlan};
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
    ///
    /// A store also builds the pyramid `config` asks for. A Dataset rejects
    /// `config.scales` before it writes anything: a multiscale layout under
    /// one root Dataset is not defined.
    pub fn import<R: ValueReader>(
        self,
        specs: Vec<(String, Genome, TrackConfig)>,
        readers: Vec<R>,
        axes: Axes,
        options: PipelineOptions,
        config: &Config,
    ) -> Result<Report> {
        match self {
            Self::Store(store) => {
                let names: Vec<String> = specs.iter().map(|(name, _, _)| name.clone()).collect();
                let report = store.create_tracks_with(specs, move |tracks| {
                    axes.apply(Import::from_readers(readers)?.into_tracks(tracks))
                        .options(options)
                        .run()
                })?;
                config.scale_tracks(store, &names)?;
                Ok(report)
            }
            Self::Dataset(dataset) => dataset_run(
                dataset,
                specs,
                readers,
                axes,
                options,
                config,
                DatasetPlan::Create,
            ),
            Self::Append(dataset) => dataset_run(
                dataset,
                specs,
                readers,
                axes,
                options,
                config,
                DatasetPlan::Append,
            ),
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
        config: &Config,
    ) -> Result<Report> {
        match self {
            Self::Store(_) => Err(PbzError::UnsupportedFormat(
                "append grows a format 0.5 root Dataset; this destination is a format 0.4 store"
                    .into(),
            )),
            Self::Dataset(dataset) | Self::Append(dataset) => dataset_run(
                dataset,
                specs,
                readers,
                axes,
                options,
                config,
                DatasetPlan::Append,
            ),
        }
    }
}

fn dataset_run<R: ValueReader>(
    dataset: &mut Dataset,
    specs: Vec<(String, Genome, TrackConfig)>,
    readers: Vec<R>,
    axes: Axes,
    options: PipelineOptions,
    config: &Config,
    plan: DatasetPlan,
) -> Result<Report> {
    if !config.scales.is_empty() {
        return Err(PbzError::Metadata(
            "format 0.5 import: scales are not supported; a multiscale pyramid under one root Dataset is not defined".into(),
        ));
    }
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
        "append: source reference differs from the store reference: {message}"
    )))
}
