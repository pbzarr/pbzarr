//! Predict an import's physical layout and resource use before any byte
//! moves.
//!
//! [`LayoutEstimate::compute`] is pure arithmetic over the same grid rules
//! the store applies at track creation and the engine applies at run time:
//! no I/O, no zarrs calls. Callers can therefore unit-test it directly and
//! search over candidate layouts without touching storage. File counts
//! assume full coverage; a sparse import elides chunk files and lands below
//! them.

use std::fmt;

use zarrs::metadata::v3::MetadataV3;

use crate::codec_spec::ExplicitArraySpec;
use crate::io::Dtype;

/// Mirrors the level-array position chunk length in `scale.rs`.
const LEVEL_CHUNK_LEN: u64 = 262_144;

const GIB: u64 = 1 << 30;

/// The genome facts the estimate needs. Per-contig lengths are not inputs,
/// so contig-dependent figures (scale-level bins) are upper bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenomeGeometry {
    /// Flat position-axis length: the sum of the contig lengths.
    pub total_len: u64,
    /// Number of contigs.
    pub contigs: u64,
}

/// One destination track's shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrackShape {
    pub name: String,
    pub dtype: Dtype,
    /// Column count; `None` for a scalar (rank-1) track.
    pub columns: Option<u64>,
}

/// The layout knobs shared by every destination track. All position and
/// column sizes clamp exactly as track creation clamps them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayoutKnobs {
    /// Inner position chunk length.
    pub chunk_size: u64,
    /// Inner column chunk width; `None` means full column width.
    pub column_chunk_size: Option<u64>,
    /// Position shard length; `None` leaves the tracks unsharded.
    pub shard_size: Option<u64>,
    /// Column shard width; `None` means full column width. Ignored when
    /// `shard_size` is `None`.
    pub shard_column_size: Option<u64>,
}

impl LayoutKnobs {
    /// The knobs as track creation resolves the write grid from user sizing.
    /// Unset sizes take the track-creation defaults (1M positions, 16
    /// columns). An explicit `spec` replaces the shard sizes: its
    /// `chunk_grid` (when present) is the outer grid, and a
    /// `sharding_indexed` codec makes that outer grid the shard with the
    /// codec's `chunk_shape` as the inner chunk; without one, the outer grid
    /// is the inner chunk and nothing shards.
    pub fn resolve(
        chunk_size: Option<u64>,
        column_chunk_size: Option<u64>,
        shard_size: Option<u64>,
        shard_column_size: Option<u64>,
        spec: Option<&ExplicitArraySpec>,
    ) -> Self {
        let chunk = chunk_size.unwrap_or(1_000_000);
        let column_chunk = column_chunk_size.unwrap_or(16);
        let Some(spec) = spec else {
            return Self {
                chunk_size: chunk,
                column_chunk_size: Some(column_chunk),
                shard_size,
                shard_column_size,
            };
        };
        let outer = spec
            .chunk_grid
            .as_ref()
            .and_then(|grid| metadata_chunk_shape(grid, "regular"))
            .unwrap_or_else(|| vec![chunk, column_chunk]);
        let outer_pos = outer.first().copied().unwrap_or(chunk);
        let outer_col = outer.get(1).copied();
        match spec
            .codecs
            .iter()
            .find_map(|codec| metadata_chunk_shape(codec, "sharding_indexed"))
        {
            Some(inner) => Self {
                chunk_size: inner.first().copied().unwrap_or(outer_pos),
                column_chunk_size: inner.get(1).copied().or(outer_col),
                shard_size: Some(outer_pos),
                shard_column_size: outer_col,
            },
            None => Self {
                chunk_size: outer_pos,
                column_chunk_size: outer_col.or(Some(column_chunk)),
                shard_size: None,
                shard_column_size: None,
            },
        }
    }
}

/// `chunk_shape` of `metadata` when it names `name` (a `regular` grid or a
/// `sharding_indexed` codec; both spell their shape the same way).
fn metadata_chunk_shape(metadata: &MetadataV3, name: &str) -> Option<Vec<u64>> {
    if metadata.name() != name {
        return None;
    }
    metadata
        .configuration()?
        .get("chunk_shape")?
        .as_array()?
        .iter()
        .map(serde_json::Value::as_u64)
        .collect()
}

/// Predicted size of one multiscale level array.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LevelEstimate {
    pub factor: u64,
    /// Level length in bins. Bins never straddle contigs, so this is
    /// `ceil(total / factor)` plus at most one ragged bin per extra contig.
    pub bins: u64,
    /// Files under `scales/<factor>/`: the group and array `zarr.json`
    /// plus the level chunk files.
    pub files: u64,
    /// Uncompressed float32 level data bytes.
    pub bytes: u64,
}

/// Predicted physical layout of one destination track.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrackEstimate {
    pub name: String,
    /// Inner chunks over both axes: the engine's write grid.
    pub chunks: u64,
    /// Shard count; `None` when unsharded.
    pub shards: Option<u64>,
    /// Files under `values/` other than `zarr.json`: shards when sharded,
    /// inner chunks otherwise.
    pub value_files: u64,
    /// Inner chunks one full shard holds; `None` when unsharded.
    pub subchunks_per_shard: Option<u64>,
    /// Uncompressed bytes of one on-disk write unit (a shard when sharded,
    /// an inner chunk otherwise).
    pub write_unit_bytes: u64,
    /// Bytes of one open import buffer (one inner chunk).
    pub buffer_bytes: u64,
    /// One entry per requested scale factor, in input order.
    pub levels: Vec<LevelEstimate>,
}

/// Predicted layout and resource use of one import run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayoutEstimate {
    pub tracks: Vec<TrackEstimate>,
    /// Every file the import creates: per track group, its `zarr.json`, the
    /// values array, the offsets/contigs (and label) coordinate arrays, and
    /// any scale levels.
    pub total_files: u64,
    /// Position windows the engine opens: the inner position chunk count.
    pub spans: u64,
    /// Work units: one `(reader, span)` piece per reader per span.
    pub pieces: u64,
    /// Open-buffer bytes across all tracks for `min(in_flight_spans, spans)`
    /// spans: the engine's scratch-memory bound.
    pub peak_scratch_bytes: u64,
    /// Pieces per worker, rounded up.
    pub tasks_per_worker: u64,
    pub warnings: Vec<String>,
}

fn dtype_size(d: Dtype) -> u64 {
    match d {
        Dtype::U8 | Dtype::I8 | Dtype::Bool => 1,
        Dtype::U16 | Dtype::I16 => 2,
        Dtype::U32 | Dtype::I32 | Dtype::F32 => 4,
        Dtype::F64 => 8,
    }
}

impl LayoutEstimate {
    /// Predict the layout of an import of `tracks` over `genome` with the
    /// given knobs, `readers` sources, `workers` threads, and the engine's
    /// `in_flight_spans` bound. `scale_factors` adds one multiscale level
    /// estimate per factor to every track.
    pub fn compute(
        genome: GenomeGeometry,
        tracks: &[TrackShape],
        knobs: &LayoutKnobs,
        readers: usize,
        workers: usize,
        in_flight_spans: usize,
        scale_factors: &[u64],
    ) -> Self {
        let total = genome.total_len;
        let chunk_pos = knobs.chunk_size.min(total.max(1)).max(1);
        let shard_pos = knobs.shard_size.map(|s| s.min(total.max(1)).max(1));
        let spans = total.div_ceil(chunk_pos);
        let workers = workers.max(1) as u64;

        let mut entries = Vec::with_capacity(tracks.len());
        let mut total_files = 0u64;
        let mut per_span_bytes = 0u64;
        let mut warnings = Vec::new();

        if spans < workers {
            warnings.push(format!(
                "parallelism-starved: {spans} span(s) feed {workers} worker(s); \
                 a smaller chunk_size adds spans"
            ));
        }

        for shape in tracks {
            let width = shape.columns.unwrap_or(1);
            let elem = dtype_size(shape.dtype);
            let col_chunk = match shape.columns {
                Some(w) => knobs.column_chunk_size.unwrap_or(w).min(w).max(1),
                None => 1,
            };
            let col_chunks = width.div_ceil(col_chunk);
            let chunks = spans * col_chunks;

            let (shards, subchunks_per_shard, write_unit_bytes) = match shard_pos {
                Some(sp) => {
                    let shard_col = match shape.columns {
                        Some(w) => knobs.shard_column_size.unwrap_or(w).min(w).max(1),
                        None => 1,
                    };
                    if shape.columns == Some(shard_col) {
                        warnings.push(format!(
                            "track {:?}: full-column-width shards on a 2D track block \
                             cheap column append; set shard_column_size below the \
                             column count",
                            shape.name
                        ));
                    }
                    (
                        Some(total.div_ceil(sp) * width.div_ceil(shard_col)),
                        Some(sp.div_ceil(chunk_pos) * shard_col.div_ceil(col_chunk)),
                        sp * shard_col * elem,
                    )
                }
                None => (None, None, chunk_pos * col_chunk * elem),
            };
            let value_files = shards.unwrap_or(chunks);

            let levels: Vec<LevelEstimate> = scale_factors
                .iter()
                .map(|&factor| {
                    let factor = factor.max(1);
                    let bins = total.div_ceil(factor) + genome.contigs.saturating_sub(1);
                    let chunk_len = LEVEL_CHUNK_LEN.min(bins.max(1));
                    LevelEstimate {
                        factor,
                        bins,
                        files: 2 + bins.div_ceil(chunk_len),
                        bytes: bins * width * 4,
                    }
                })
                .collect();

            // Group + values zarr.json, the value chunk files, offsets
            // (zarr.json + one chunk), contigs, the label coordinate for 2D
            // tracks, and the scales subtree (its group zarr.json + levels).
            total_files += 2
                + value_files
                + 2
                + 1
                + u64::from(genome.contigs > 0)
                + match shape.columns {
                    Some(w) => 1 + u64::from(w > 0),
                    None => 0,
                }
                + if levels.is_empty() {
                    0
                } else {
                    1 + levels.iter().map(|l| l.files).sum::<u64>()
                };
            per_span_bytes += chunk_pos * width * elem;

            entries.push(TrackEstimate {
                name: shape.name.clone(),
                chunks,
                shards,
                value_files,
                subchunks_per_shard,
                write_unit_bytes,
                buffer_bytes: chunk_pos * col_chunk * elem,
                levels,
            });
        }

        let pieces = spans * readers as u64;
        let peak_scratch_bytes = per_span_bytes * spans.min(in_flight_spans.max(1) as u64);
        if peak_scratch_bytes / workers > GIB {
            warnings.push(format!(
                "per-worker buffer bytes {} exceed 1 GiB; lower in_flight_spans \
                 or the chunk sizes",
                human_bytes(peak_scratch_bytes / workers)
            ));
        }

        Self {
            tracks: entries,
            total_files,
            spans,
            pieces,
            peak_scratch_bytes,
            tasks_per_worker: pieces.div_ceil(workers),
            warnings,
        }
    }
}

fn human_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        return format!("{bytes} B");
    }
    const UNITS: [&str; 4] = ["KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64 / 1024.0;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    format!("{value:.1} {}", UNITS[unit])
}

fn write_table(f: &mut fmt::Formatter<'_>, header: &[&str], rows: &[Vec<String>]) -> fmt::Result {
    let mut widths: Vec<usize> = header.iter().map(|h| h.len()).collect();
    for row in rows {
        for (w, cell) in widths.iter_mut().zip(row) {
            *w = (*w).max(cell.len());
        }
    }
    for (i, (h, &w)) in header.iter().zip(&widths).enumerate() {
        if i > 0 {
            f.write_str("  ")?;
        }
        if i == 0 {
            write!(f, "{h:<w$}")?;
        } else {
            write!(f, "{h:>w$}")?;
        }
    }
    writeln!(f)?;
    for row in rows {
        for (i, (cell, &w)) in row.iter().zip(&widths).enumerate() {
            if i > 0 {
                f.write_str("  ")?;
            }
            if i == 0 {
                write!(f, "{cell:<w$}")?;
            } else {
                write!(f, "{cell:>w$}")?;
            }
        }
        writeln!(f)?;
    }
    Ok(())
}

impl fmt::Display for LayoutEstimate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rows: Vec<Vec<String>> = self
            .tracks
            .iter()
            .map(|t| {
                vec![
                    t.name.clone(),
                    t.chunks.to_string(),
                    t.shards.map_or_else(|| "-".to_owned(), |s| s.to_string()),
                    t.value_files.to_string(),
                    t.subchunks_per_shard
                        .map_or_else(|| "-".to_owned(), |s| s.to_string()),
                    human_bytes(t.write_unit_bytes),
                    human_bytes(t.buffer_bytes),
                ]
            })
            .collect();
        write_table(
            f,
            &[
                "track",
                "chunks",
                "shards",
                "value files",
                "subchunks/shard",
                "write unit",
                "buffer",
            ],
            &rows,
        )?;

        let level_rows: Vec<Vec<String>> = self
            .tracks
            .iter()
            .flat_map(|t| {
                t.levels.iter().map(|l| {
                    vec![
                        t.name.clone(),
                        l.factor.to_string(),
                        l.bins.to_string(),
                        l.files.to_string(),
                        human_bytes(l.bytes),
                    ]
                })
            })
            .collect();
        if !level_rows.is_empty() {
            writeln!(f)?;
            write_table(
                f,
                &["track", "factor", "bins", "level files", "level bytes"],
                &level_rows,
            )?;
        }

        writeln!(f)?;
        writeln!(
            f,
            "files {}  spans {}  pieces {}  peak scratch {}  tasks/worker {}",
            self.total_files,
            self.spans,
            self.pieces,
            human_bytes(self.peak_scratch_bytes),
            self.tasks_per_worker,
        )?;
        for warning in &self.warnings {
            writeln!(f, "warning: {warning}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;
    use crate::genome::{Contig, Genome};
    use crate::import::Import;
    use crate::io::{OutputSchema, OutputSinkMut, ValueReader};
    use crate::store::PbzStore;
    use crate::track::TrackConfig;

    #[test]
    fn unsharded_scalar_track_layout() {
        let est = LayoutEstimate::compute(
            GenomeGeometry {
                total_len: 4_000_000,
                contigs: 2,
            },
            &[TrackShape {
                name: "depth".into(),
                dtype: Dtype::I32,
                columns: None,
            }],
            &LayoutKnobs {
                chunk_size: 250_000,
                column_chunk_size: None,
                shard_size: None,
                shard_column_size: None,
            },
            1,
            4,
            8,
            &[],
        );

        let t = &est.tracks[0];
        assert_eq!(t.chunks, 16);
        assert_eq!(t.shards, None);
        assert_eq!(t.value_files, 16);
        assert_eq!(t.subchunks_per_shard, None);
        // One 250_000-position i32 chunk is both the write unit and the
        // buffer.
        assert_eq!(t.write_unit_bytes, 1_000_000);
        assert_eq!(t.buffer_bytes, 1_000_000);
        assert!(t.levels.is_empty());

        // group + values zarr.json (2) + 16 chunks + offsets (2) +
        // contigs (2).
        assert_eq!(est.total_files, 22);
        assert_eq!(est.spans, 16);
        assert_eq!(est.pieces, 16);
        assert_eq!(est.tasks_per_worker, 4);
        // 8 in-flight spans of one 1 MB buffer each.
        assert_eq!(est.peak_scratch_bytes, 8_000_000);
        assert!(est.warnings.is_empty(), "{:?}", est.warnings);
    }

    #[test]
    fn sharded_cohort_with_scales_and_warnings() {
        let genome = GenomeGeometry {
            total_len: 4_000_000,
            contigs: 2,
        };
        let track = TrackShape {
            name: "depth".into(),
            dtype: Dtype::I32,
            columns: Some(3),
        };
        let knobs = LayoutKnobs {
            chunk_size: 250_000,
            column_chunk_size: Some(2),
            shard_size: Some(1_000_000),
            shard_column_size: None,
        };
        let est = LayoutEstimate::compute(
            genome,
            std::slice::from_ref(&track),
            &knobs,
            3,
            4,
            8,
            &[32, 256],
        );

        let t = &est.tracks[0];
        // 16 position chunks x 2 column chunks; 4 position shards x 1
        // column shard.
        assert_eq!(t.chunks, 32);
        assert_eq!(t.shards, Some(4));
        assert_eq!(t.value_files, 4);
        assert_eq!(t.subchunks_per_shard, Some(8));
        assert_eq!(t.write_unit_bytes, 12_000_000);
        assert_eq!(t.buffer_bytes, 2_000_000);
        assert_eq!(
            t.levels,
            vec![
                // ceil(4M/32) + 1 ragged bin for the second contig; one
                // level chunk file + 2 zarr.json.
                LevelEstimate {
                    factor: 32,
                    bins: 125_001,
                    files: 3,
                    bytes: 1_500_012,
                },
                LevelEstimate {
                    factor: 256,
                    bins: 15_626,
                    files: 3,
                    bytes: 187_512,
                },
            ]
        );

        // group + values zarr.json (2) + 4 shards + offsets (2) + contigs
        // (2) + labels (2) + scales group (1) + two levels (3 each).
        assert_eq!(est.total_files, 19);
        assert_eq!(est.spans, 16);
        assert_eq!(est.pieces, 48);
        assert_eq!(est.tasks_per_worker, 12);
        // 8 in-flight spans x 250_000 positions x 3 columns x 4 bytes.
        assert_eq!(est.peak_scratch_bytes, 24_000_000);
        // Only the full-column-width shard warning fires here.
        assert_eq!(est.warnings.len(), 1, "{:?}", est.warnings);
        assert!(est.warnings[0].contains("column append"));

        // 16 spans feeding 32 workers adds the parallelism warning.
        let est =
            LayoutEstimate::compute(genome, std::slice::from_ref(&track), &knobs, 3, 32, 8, &[]);
        assert_eq!(est.warnings.len(), 2, "{:?}", est.warnings);
        assert!(est.warnings[0].contains("parallelism-starved"));

        // A wide f64 cohort on one worker exceeds the 1 GiB per-worker
        // buffer bound: 250_000 x 3000 x 8 = 6 GB for a single span.
        let est = LayoutEstimate::compute(
            genome,
            &[TrackShape {
                name: "wide".into(),
                dtype: Dtype::F64,
                columns: Some(3000),
            }],
            &LayoutKnobs {
                chunk_size: 250_000,
                column_chunk_size: Some(100),
                shard_size: None,
                shard_column_size: None,
            },
            1,
            1,
            1,
            &[],
        );
        assert_eq!(est.warnings.len(), 1, "{:?}", est.warnings);
        assert!(est.warnings[0].contains("1 GiB"));
    }

    #[test]
    fn knobs_resolve_mirrors_track_creation_grid() {
        // Size flags pass through; unset sizes take the creation defaults.
        assert_eq!(
            LayoutKnobs::resolve(None, None, Some(64), Some(2), None),
            LayoutKnobs {
                chunk_size: 1_000_000,
                column_chunk_size: Some(16),
                shard_size: Some(64),
                shard_column_size: Some(2),
            }
        );

        // A sharding codec makes the explicit grid the shard and its
        // chunk_shape the inner chunk.
        let sharded = ExplicitArraySpec::parse(
            r#"{"chunk_grid":{"name":"regular","configuration":{"chunk_shape":[32,4]}},
                "codecs":[{"name":"sharding_indexed","configuration":
                    {"chunk_shape":[8,2],"codecs":[{"name":"bytes"}]}}]}"#,
        )
        .unwrap();
        assert_eq!(
            LayoutKnobs::resolve(None, None, Some(999), None, Some(&sharded)),
            LayoutKnobs {
                chunk_size: 8,
                column_chunk_size: Some(2),
                shard_size: Some(32),
                shard_column_size: Some(4),
            }
        );

        // Without one, the explicit grid is the inner chunk and nothing
        // shards, the shard flags included.
        let flat = ExplicitArraySpec::parse(
            r#"{"chunk_grid":{"name":"regular","configuration":{"chunk_shape":[32,4]}},
                "codecs":[{"name":"zstd","configuration":{"level":3,"checksum":false}}]}"#,
        )
        .unwrap();
        assert_eq!(
            LayoutKnobs::resolve(None, None, Some(999), None, Some(&flat)),
            LayoutKnobs {
                chunk_size: 32,
                column_chunk_size: Some(4),
                shard_size: None,
                shard_column_size: None,
            }
        );
    }

    /// Estimate-only reader: geometry and schema, never any data.
    struct StubReader {
        genome: Genome,
        schema: OutputSchema,
    }

    impl ValueReader for StubReader {
        fn contigs(&self) -> &Genome {
            &self.genome
        }

        fn output_schema(&self) -> &OutputSchema {
            &self.schema
        }

        fn read_into(
            &mut self,
            _contig: &str,
            _start: u64,
            _end: u64,
            _outputs: &mut [OutputSinkMut<'_>],
        ) -> std::result::Result<(), crate::io::ReaderError> {
            Ok(())
        }

        fn fork(&self) -> std::result::Result<Self, crate::io::ReaderError> {
            Ok(Self {
                genome: self.genome.clone(),
                schema: self.schema.clone(),
            })
        }
    }

    #[test]
    fn builder_estimate_maps_real_track_geometry() {
        let dir = TempDir::new().unwrap();
        let genome = Genome::new(vec![Contig {
            name: "chr1".into(),
            length: 100,
        }])
        .unwrap();
        let mut store = PbzStore::create(dir.path().join("est.pbz")).unwrap();
        store
            .create_track(
                "depth",
                genome.clone(),
                TrackConfig::new(Dtype::I32)
                    .columns((0..4).map(|i| format!("s{i}")).collect())
                    .chunk_size(16)
                    .column_chunk_size(2)
                    .shard_size(32)
                    .shard_column_size(4),
            )
            .unwrap();

        let readers: Vec<StubReader> = (0..4)
            .map(|_| StubReader {
                genome: genome.clone(),
                schema: OutputSchema::single("value", Dtype::I32),
            })
            .collect();
        let est = Import::from_readers(readers)
            .unwrap()
            .into_track(store.track("depth").unwrap())
            .readers_as_columns()
            .estimate()
            .unwrap();

        let t = &est.tracks[0];
        assert_eq!(est.spans, 7);
        assert_eq!(t.chunks, 14);
        assert_eq!(t.shards, Some(4));
        assert_eq!(t.value_files, 4);
        assert_eq!(t.subchunks_per_shard, Some(4));
        assert_eq!(t.write_unit_bytes, 512);
        assert_eq!(t.buffer_bytes, 128);
        assert_eq!(est.pieces, 28);
        // shard_column_size 4 covers all 4 columns.
        assert_eq!(est.warnings.len(), 1, "{:?}", est.warnings);
        assert!(est.warnings[0].contains("column append"));
    }
}
