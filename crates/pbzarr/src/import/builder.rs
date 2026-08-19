//! Declarative builder over the import engine.
//!
//! ```ignore
//! Import::from_readers(readers)?
//!     .into_track(&track)          // or .into_tracks(&[&a, &b])
//!     .readers_as_columns()        // or .fields_as_tracks() / .fields_as_columns()
//!     .options(PipelineOptions::default())
//!     .run()?;
//! ```
//!
//! An axis with no fan-out needs no statement: one reader, one schema field,
//! one scalar track is just `from_readers(..).into_track(..).run()`.

use std::sync::Arc;

use crossbeam_channel::Sender;
use zarrs::array::ArrayShardedExt;

use crate::error::{PbzError, Result};
use crate::import::config::Report;
use crate::import::engine::{self, CostModel, PipelineOptions, TapMessage};
use crate::import::estimate::{GenomeGeometry, LayoutEstimate, LayoutKnobs, TrackShape};
use crate::import::routing::{ImportRouting, SourceAxis};
use crate::io::ValueReader;
use crate::track::Track;

pub struct Import;

impl Import {
    /// Start an import. Readers are positional: reader `i` is column `i`.
    pub fn from_readers<'t, R: ValueReader>(readers: Vec<R>) -> Result<ImportBuilder<'t, R>> {
        if readers.is_empty() {
            return Err(PbzError::Metadata("import: no readers".into()));
        }
        Ok(ImportBuilder {
            readers,
            tracks: Vec::new(),
            tracks_from: None,
            columns_from: None,
            options: PipelineOptions::default(),
            cost_model: None,
            tap: None,
            expected_labels: None,
        })
    }
}

pub struct ImportBuilder<'t, R: ValueReader> {
    readers: Vec<R>,
    tracks: Vec<&'t Track>,
    tracks_from: Option<SourceAxis>,
    columns_from: Option<SourceAxis>,
    options: PipelineOptions,
    cost_model: Option<Arc<dyn CostModel>>,
    tap: Option<Sender<TapMessage>>,
    expected_labels: Option<Vec<String>>,
}

impl<'t, R: ValueReader> ImportBuilder<'t, R> {
    pub fn into_track(mut self, track: &'t Track) -> Self {
        self.tracks = vec![track];
        self
    }

    pub fn into_tracks(mut self, tracks: &[&'t Track]) -> Self {
        self.tracks = tracks.to_vec();
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

    pub fn options(mut self, options: PipelineOptions) -> Self {
        self.options = options;
        self
    }

    /// Attach a span cost model: spans then open in descending summed-cost
    /// order instead of genome order.
    pub fn with_cost_model(mut self, model: Arc<dyn CostModel>) -> Self {
        self.cost_model = Some(model);
        self
    }

    /// Attach a write tap: one [`TapMessage`] per closed buffer. Consume the
    /// channel concurrently, or size it for the whole run; a dropped receiver
    /// silently detaches the tap.
    pub fn with_tap(mut self, tap: Sender<TapMessage>) -> Self {
        self.tap = Some(tap);
        self
    }

    /// Cross-check the destination tracks' on-disk column coordinate against
    /// `labels` before writing anything.
    pub fn expect_column_labels(mut self, labels: Vec<String>) -> Self {
        self.expected_labels = Some(labels);
        self
    }

    /// Predict this import's physical layout and resource use without
    /// running it. Validates the routing exactly like [`run`](Self::run),
    /// then maps the resolved track geometry onto
    /// [`LayoutEstimate::compute`].
    pub fn estimate(&self) -> Result<LayoutEstimate> {
        if self.tracks.is_empty() {
            return Err(PbzError::Metadata(
                "import: no destination tracks; call into_track(s) first".into(),
            ));
        }
        let schemas: Vec<_> = self
            .readers
            .iter()
            .map(ValueReader::output_schema)
            .collect();
        let genomes: Vec<_> = self.readers.iter().map(ValueReader::contigs).collect();
        ImportRouting::resolve(
            &schemas,
            &genomes,
            &self.tracks,
            self.tracks_from,
            self.columns_from,
            self.expected_labels.as_deref(),
        )?;

        let mut shapes = Vec::with_capacity(self.tracks.len());
        for track in &self.tracks {
            shapes.push(TrackShape {
                name: track.name().to_owned(),
                dtype: track.dtype(),
                columns: if track.rank() == 2 {
                    Some(track.columns_count()? as u64)
                } else {
                    None
                },
            });
        }

        // The engine requires every target to share the position grid, so
        // the first track's geometry stands for the knobs; multi-track
        // routings only ever target scalar tracks, so the column knobs can
        // only come from a lone rank-2 track.
        let first = self.tracks[0];
        let rank2 = first.rank() == 2;
        let array = first.values_array()?;
        let write_unit = first.write_unit_shape()?;
        let sharded = array.is_sharded();
        let (chunk_size, column_chunk_size) = if sharded {
            let sub = array.effective_subchunk_shape().ok_or_else(|| {
                PbzError::Metadata(format!(
                    "track {:?}: sharded values array with indeterminate subchunk shape",
                    first.name()
                ))
            })?;
            let sub = sub.as_slice();
            (sub[0].get(), rank2.then(|| sub[1].get()))
        } else {
            (write_unit[0] as u64, rank2.then(|| write_unit[1] as u64))
        };
        let knobs = LayoutKnobs {
            chunk_size,
            column_chunk_size,
            shard_size: sharded.then(|| write_unit[0] as u64),
            shard_column_size: (sharded && rank2).then(|| write_unit[1] as u64),
        };

        Ok(LayoutEstimate::compute(
            GenomeGeometry {
                total_len: first.total_len(),
                contigs: first.genome().len() as u64,
            },
            &shapes,
            &knobs,
            self.readers.len(),
            self.options.workers,
            self.options.in_flight_spans,
            &[],
        ))
    }

    pub fn run(self) -> Result<Report> {
        if self.tracks.is_empty() {
            return Err(PbzError::Metadata(
                "import: no destination tracks; call into_track(s) first".into(),
            ));
        }
        let schemas: Vec<_> = self
            .readers
            .iter()
            .map(ValueReader::output_schema)
            .collect();
        let genomes: Vec<_> = self.readers.iter().map(ValueReader::contigs).collect();
        let routing = ImportRouting::resolve(
            &schemas,
            &genomes,
            &self.tracks,
            self.tracks_from,
            self.columns_from,
            self.expected_labels.as_deref(),
        )?;
        engine::run_import(
            self.readers,
            &self.tracks,
            &routing,
            &self.options,
            self.cost_model.as_deref(),
            self.tap.as_ref(),
        )
    }
}
