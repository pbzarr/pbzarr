//! Declarative builder over the import engine.
//!
//! ```ignore
//! Import::from_readers(readers)?
//!     .into_dataset(&mut dataset, specs)
//!     .readers_as_columns()        // or .fields_as_tracks() / .fields_as_columns()
//!     .config(Config::default())
//!     .run()?;
//! ```
//!
//! An axis with no fan-out needs no statement: one reader, one schema field,
//! one scalar track is just `from_readers(..).into_dataset(..).run()`.

use crate::dataset::DatasetPlan;
use crate::error::{PbzError, Result};
use crate::import::config::{Config, Report};
use crate::import::engine;
use crate::import::routing::{ImportRouting, SourceAxis};
use crate::io::ValueReader;
use crate::{TrackConfig, dataset::Dataset};

/// A Dataset destination: the store, its track specifications, and whether
/// the run creates those tracks or widens them.
type DatasetTarget<'t> = (&'t mut Dataset, Vec<(String, TrackConfig)>, DatasetPlan);

pub struct Import;

impl Import {
    /// Start an import. Readers are positional: reader `i` is column `i`.
    pub fn from_readers<'t, R: ValueReader>(readers: Vec<R>) -> Result<ImportBuilder<'t, R>> {
        if readers.is_empty() {
            return Err(PbzError::Metadata("import: no readers".into()));
        }
        Ok(ImportBuilder {
            readers,
            dataset: None,
            tracks_from: None,
            columns_from: None,
            config: Config::default(),
            handle_budget: 0,
            expected_labels: None,
        })
    }
}

pub struct ImportBuilder<'t, R: ValueReader> {
    readers: Vec<R>,
    dataset: Option<DatasetTarget<'t>>,
    tracks_from: Option<SourceAxis>,
    columns_from: Option<SourceAxis>,
    config: Config,
    /// Reader handles across all readers; 0 derives it from the fd limit.
    handle_budget: usize,
    expected_labels: Option<Vec<String>>,
}

impl<'t, R: ValueReader> ImportBuilder<'t, R> {
    /// Import into new root arrays in a format-0.5 Dataset.
    ///
    /// Configuration is read-only; run validates the whole request
    /// before writing. Existing dimensions require identical ordered labels.
    /// Errors may leave incomplete output; reopen the Dataset after an error.
    pub fn into_dataset(
        self,
        dataset: &'t mut Dataset,
        tracks: Vec<(String, TrackConfig)>,
    ) -> Self {
        self.into_dataset_plan(dataset, tracks, DatasetPlan::Create)
    }

    pub(crate) fn into_dataset_plan(
        mut self,
        dataset: &'t mut Dataset,
        tracks: Vec<(String, TrackConfig)>,
        plan: DatasetPlan,
    ) -> Self {
        self.dataset = Some((dataset, tracks, plan));
        self
    }

    pub fn fields_as_tracks(mut self) -> Self {
        self.tracks_from = Some(SourceAxis::Fields);
        self
    }

    pub fn fields_as_columns(mut self) -> Self {
        self.columns_from = Some(SourceAxis::Fields);
        self
    }

    pub fn readers_as_columns(mut self) -> Self {
        self.columns_from = Some(SourceAxis::Readers);
        self
    }

    /// The run reads the worker count, `decode_chunks`, the progress hook and
    /// the cancel flag.
    /// The tracks take their sizes from their `TrackConfig`.
    pub fn config(mut self, config: Config) -> Self {
        self.config = config;
        self
    }

    #[cfg(test)]
    pub(crate) fn handle_budget(mut self, budget: usize) -> Self {
        self.handle_budget = budget;
        self
    }

    /// Cross-check the destination tracks' on-disk column coordinate against
    /// `labels` before writing anything.
    pub fn expect_column_labels(mut self, labels: Vec<String>) -> Self {
        self.expected_labels = Some(labels);
        self
    }

    pub fn run(mut self) -> Result<Report> {
        let prepared = self
            .dataset
            .as_ref()
            .map(|(ds, specs, plan)| ds.prepare(specs, *plan))
            .transpose()?;
        let Some(prepared) = prepared else {
            return Err(PbzError::Metadata(
                "import: no destination; call into_dataset first".into(),
            ));
        };
        let tracks = prepared.targets();
        if tracks.is_empty() {
            return Err(PbzError::Metadata("import: no destination tracks".into()));
        }
        let schemas: Vec<_> = self
            .readers
            .iter()
            .map(ValueReader::output_schema)
            .collect();
        let references: Vec<_> = self.readers.iter().map(ValueReader::contigs).collect();
        let routing = ImportRouting::resolve(
            &schemas,
            &references,
            tracks,
            self.tracks_from,
            self.columns_from,
            self.expected_labels.as_deref(),
        )?;
        engine::validate_layout(tracks, &routing)?;
        if let Some((ds, ..)) = self.dataset.as_mut() {
            ds.begin(&prepared)?;
        }
        let report = engine::run_import(
            self.readers,
            tracks,
            &routing,
            &self.config,
            self.handle_budget,
        )?;
        if let Some((ds, ..)) = self.dataset.as_mut() {
            ds.finish(&prepared)?;
        }
        Ok(report)
    }
}
