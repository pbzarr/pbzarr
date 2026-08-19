//! `ImportRouting`: which source axis fans out over tracks, which over
//! columns, and one [`TrackTarget`] per destination track.
//!
//! Always built by [`ImportRouting::resolve`], never by hand: the constructor
//! owns every cross-source and source-vs-track check, so the engine can trust
//! what it gets.

use std::ops::Range;

use crate::error::{PbzError, Result};
use crate::genome::Genome;
use crate::io::error::ReaderError;
use crate::io::{Dtype, OutputSchema};
use crate::track::Track;

/// A source axis that can fan out over tracks or columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceAxis {
    Fields,
    /// The readers themselves, positionally: reader `i` is column `i`.
    Readers,
}

/// One destination track of a routing.
#[derive(Debug, Clone)]
pub struct TrackTarget {
    pub track: String,
    /// Schema field feeding this target. Under `columns_from = Fields` this
    /// is the field for the first column of `columns`; successive columns
    /// take successive fields.
    pub field: usize,
    /// Element dtype; matches both the schema field and the track.
    pub dtype: Dtype,
    /// Destination range on the track's column axis, not assumed to start at
    /// 0 (a column-append can target e.g. `3..5`). Scalar targets use `0..1`.
    pub columns: Range<u64>,
    /// Expected on-disk labels over `columns`, checked at resolve time
    /// against the track's column coordinate array.
    pub expected_labels: Option<Vec<String>>,
}

/// Axis mapping plus one target per destination track. `None` on an axis
/// means that axis has no fan-out: a single track, or scalar width.
#[derive(Debug, Clone)]
pub struct ImportRouting {
    pub tracks_from: Option<SourceAxis>,
    pub columns_from: Option<SourceAxis>,
    pub targets: Vec<TrackTarget>,
}

impl ImportRouting {
    /// Derive and validate a routing from the reader schemas and genomes, the
    /// destination tracks, and the builder's axis statements.
    pub fn resolve(
        schemas: &[&OutputSchema],
        genomes: &[&Genome],
        tracks: &[&Track],
        tracks_from: Option<SourceAxis>,
        columns_from: Option<SourceAxis>,
        expected_labels: Option<&[String]>,
    ) -> Result<Self> {
        if schemas.is_empty() {
            return Err(PbzError::Metadata("import routing: no readers".into()));
        }
        if schemas.len() != genomes.len() {
            return Err(PbzError::Metadata(
                "import routing: schema/genome count mismatch".into(),
            ));
        }
        if tracks.is_empty() {
            return Err(PbzError::Metadata("import routing: no tracks".into()));
        }
        if tracks_from == Some(SourceAxis::Readers) {
            return Err(PbzError::Metadata(
                "import routing: readers_as_tracks is not implemented".into(),
            ));
        }
        if tracks_from.is_some() && tracks_from == columns_from {
            return Err(PbzError::Metadata(
                "import routing: tracks and columns cannot come from the same source axis".into(),
            ));
        }

        let schema = schemas[0];
        if schema.is_empty() {
            return Err(PbzError::Metadata(
                "import routing: reader schema has no fields".into(),
            ));
        }
        for (i, other) in schemas.iter().enumerate().skip(1) {
            if *other != schema {
                return Err(PbzError::Reader(ReaderError::SchemaMismatch {
                    message: format!(
                        "reader {i} schema {:?} differs from reader 0 schema {:?}",
                        other.fields, schema.fields
                    ),
                }));
            }
        }

        // Genomes must match as ordered contig sequences: the flat position
        // axis is defined by contig order, so the order-insensitive checksum
        // cannot stand in for this check.
        let reference = genomes[0];
        for (i, genome) in genomes.iter().enumerate().skip(1) {
            if genome.contigs() != reference.contigs() {
                return Err(PbzError::Metadata(format!(
                    "import routing: reader {i} genome differs from reader 0 \
                     (ordered contig sequences must be identical)"
                )));
            }
        }
        for track in tracks {
            if track.genome().contigs() != reference.contigs() {
                return Err(PbzError::Metadata(format!(
                    "import routing: track {:?} genome differs from the readers \
                     (ordered contig sequences must be identical)",
                    track.name()
                )));
            }
        }

        let n_readers = schemas.len() as u64;
        let n_fields = schema.len();

        let field_of_target: Vec<usize> = match tracks_from {
            Some(SourceAxis::Fields) => {
                if tracks.len() != n_fields {
                    return Err(PbzError::Metadata(format!(
                        "import routing: {n_fields} schema field(s) for {} track(s)",
                        tracks.len()
                    )));
                }
                (0..n_fields).collect()
            }
            None => {
                if tracks.len() != 1 {
                    return Err(PbzError::Metadata(format!(
                        "import routing: {} tracks but no tracks_from axis",
                        tracks.len()
                    )));
                }
                if columns_from != Some(SourceAxis::Fields) && n_fields != 1 {
                    return Err(PbzError::Metadata(format!(
                        "import routing: schema has {n_fields} fields but neither \
                         tracks_from nor columns_from consumes the field axis"
                    )));
                }
                vec![0]
            }
            Some(SourceAxis::Readers) => unreachable!("rejected above"),
        };

        if columns_from == Some(SourceAxis::Fields) && schemas.len() != 1 {
            return Err(PbzError::Metadata(format!(
                "import routing: fields_as_columns requires exactly one reader, got {}",
                schemas.len()
            )));
        }

        let mut targets = Vec::with_capacity(tracks.len());
        for (track, &field) in tracks.iter().zip(&field_of_target) {
            let width = match columns_from {
                Some(SourceAxis::Readers) => n_readers,
                Some(SourceAxis::Fields) => n_fields as u64,
                None => 1,
            };
            let columns = 0..width;

            if columns_from.is_none() {
                if track.rank() != 1 {
                    return Err(PbzError::Metadata(format!(
                        "import routing: track {:?} is rank {} but the routing is scalar",
                        track.name(),
                        track.rank()
                    )));
                }
            } else {
                if track.rank() != 2 {
                    return Err(PbzError::Metadata(format!(
                        "import routing: track {:?} is rank {} but the routing has a column axis",
                        track.name(),
                        track.rank()
                    )));
                }
                let on_disk = track.columns_count()? as u64;
                if on_disk != width {
                    return Err(PbzError::Metadata(format!(
                        "import routing: track {:?} has {on_disk} column(s) but the \
                         routing supplies {width}",
                        track.name()
                    )));
                }
            }

            let feeding_fields: Range<usize> = if columns_from == Some(SourceAxis::Fields) {
                field..field + n_fields
            } else {
                field..field + 1
            };
            for f in feeding_fields {
                let fd = schema.fields[f].dtype;
                if fd != track.dtype() {
                    return Err(PbzError::InvalidDtype {
                        dtype: format!(
                            "import routing: track {:?} is {} but schema field {f} is {fd}",
                            track.name(),
                            track.dtype()
                        ),
                    });
                }
            }

            let expected = expected_labels.map(<[String]>::to_vec);
            if let Some(expected) = &expected {
                if track.rank() != 2 {
                    return Err(PbzError::Metadata(format!(
                        "import routing: expected_labels given for scalar track {:?}",
                        track.name()
                    )));
                }
                if expected.len() as u64 != columns.end - columns.start {
                    return Err(PbzError::Metadata(format!(
                        "import routing: {} expected label(s) for {} column(s) on track {:?}",
                        expected.len(),
                        columns.end - columns.start,
                        track.name()
                    )));
                }
                let on_disk = track.column_labels()?;
                let lo = columns.start as usize;
                let hi = columns.end as usize;
                if on_disk.get(lo..hi) != Some(expected.as_slice()) {
                    return Err(PbzError::Metadata(format!(
                        "import routing: track {:?} columns {lo}..{hi} carry labels {:?} \
                         but the import expected {:?}",
                        track.name(),
                        on_disk.get(lo..hi),
                        expected
                    )));
                }
            }

            targets.push(TrackTarget {
                track: track.name().to_owned(),
                field,
                dtype: track.dtype(),
                columns,
                expected_labels: expected,
            });
        }

        Ok(Self {
            tracks_from,
            columns_from,
            targets,
        })
    }
}
