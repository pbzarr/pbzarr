//! Single-pass, multi-column import of one tabix-indexed BED into N scalar
//! tracks. A [`BedSchema`] names which file columns become which tracks; each
//! BED line is parsed once and scattered into every target track's buffer.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use noodles_bgzf as bgzf;
use noodles_core::Position;
use noodles_core::region::Interval;
use noodles_csi::BinningIndex;

use pbzarr::genome::Genome;
use pbzarr::import::{Config, Report, run_matrix_pipeline, run_multi_pipeline, run_wide_pipeline};
use pbzarr::io::{ColumnSinkMut, Dtype, MultiValueReader, ReaderError};
use pbzarr::{PbzError, PbzStore, Result, TrackConfig};

use super::import::zero_fill;
use super::reader::column_index_by_name;
use super::reader::open_bgzf;
use super::schema::{BedImportOptions, ResolvedImport, resolve_sources};

/// Reader-side result alias (parse/IO errors), distinct from `pbzarr::Result`.
type IoResult<T> = std::result::Result<T, ReaderError>;

/// How a schema entry picks a BED value column.
pub enum ColumnSelector {
    /// Absolute 0-based file column index (`>= 3`).
    Index(usize),
    /// A name from the `#`-prefixed header line.
    Name(String),
}

/// One column of a [`BedSchema`]: which file column, its dtype, and the target
/// track. `track_name` defaults to the header name for `Name` selectors and is
/// required for `Index` selectors. Uncovered positions always take the dtype's
/// zero fill; a non-zero fill knob is deliberately not offered (gaps and true
/// zeros are indistinguishable, matching `from_bed`).
pub struct BedColumnSpec {
    pub selector: ColumnSelector,
    pub dtype: Dtype,
    pub track_name: Option<String>,
    pub description: Option<String>,
}

impl BedColumnSpec {
    /// Select a column by header name; the track takes the same name.
    pub fn named(name: impl Into<String>, dtype: Dtype) -> Self {
        Self {
            selector: ColumnSelector::Name(name.into()),
            dtype,
            track_name: None,
            description: None,
        }
    }

    /// Select a column by absolute file index; a track name is required.
    pub fn indexed(index: usize, dtype: Dtype, track_name: impl Into<String>) -> Self {
        Self {
            selector: ColumnSelector::Index(index),
            dtype,
            track_name: Some(track_name.into()),
            description: None,
        }
    }
}

/// An ordered set of columns to import from one BED file.
pub struct BedSchema(pub Vec<BedColumnSpec>);

/// A schema entry with its selector resolved to a concrete file column + name.
pub(super) struct Resolved {
    pub file_col: usize,
    pub dtype: Dtype,
    pub track_name: String,
    pub description: Option<String>,
}

/// Resolve every schema entry: map `Name` selectors to file columns via the
/// header, default track names, and reject unsupported dtypes, coordinate
/// columns, and duplicate track names.
pub(super) fn resolve(bed_gz: &Path, schema: &BedSchema) -> Result<Vec<Resolved>> {
    if schema.0.is_empty() {
        return Err(PbzError::Metadata("bed multi import: empty schema".into()));
    }
    let mut out: Vec<Resolved> = Vec::with_capacity(schema.0.len());
    let mut seen: HashSet<String> = HashSet::new();
    for spec in &schema.0 {
        let (file_col, default_name) = match &spec.selector {
            ColumnSelector::Index(i) => (*i, None),
            ColumnSelector::Name(n) => {
                let idx = column_index_by_name(bed_gz, n)
                    .map_err(|e| PbzError::Store(format!("resolve column {n:?}: {e}")))?;
                (idx, Some(n.clone()))
            }
        };
        if file_col < 3 {
            return Err(PbzError::Metadata(format!(
                "bed multi import: column {file_col} is a coordinate column (need >= 3)"
            )));
        }
        let track_name = spec.track_name.clone().or(default_name).ok_or_else(|| {
            PbzError::Metadata(format!(
                "bed multi import: index column {file_col} needs an explicit track_name"
            ))
        })?;
        if !seen.insert(track_name.clone()) {
            return Err(PbzError::Metadata(format!(
                "bed multi import: duplicate track name {track_name:?}"
            )));
        }
        out.push(Resolved {
            file_col,
            dtype: spec.dtype,
            track_name,
            description: spec.description.clone(),
        });
    }
    Ok(out)
}

/// Shared, thread-safe reader state: index + name→id map + resolved columns.
struct MultiShared {
    path: PathBuf,
    index: noodles_tabix::Index,
    ref_ids: HashMap<String, usize>,
    columns: Vec<(Option<usize>, Dtype)>,
    dtypes: Vec<Dtype>,
    genome: Genome,
}

/// Multi-column BED reader. One tabix seek per region; each line's cells are
/// scattered into the matching per-track sink.
pub struct BedMultiReader {
    shared: Arc<MultiShared>,
    bgzf: Mutex<bgzf::io::Reader<File>>,
}

impl BedMultiReader {
    /// Open a bgzipped, tabix-indexed BED. `columns` is `(file column, dtype)`
    /// in track order; every file column must be `>= 3`. A `None` column is a
    /// presence column: covered positions read as `1` (BED3 masks). `genome`'s
    /// contig names must match the BED's.
    pub fn open<P: AsRef<Path>>(
        bed_gz: P,
        columns: Vec<(Option<usize>, Dtype)>,
        genome: Genome,
    ) -> IoResult<Self> {
        let path = bed_gz.as_ref().to_path_buf();
        let tbi = PathBuf::from(format!("{}.tbi", path.display()));
        let index = noodles_tabix::fs::read(&tbi).map_err(|source| ReaderError::Io {
            path: tbi.clone(),
            source,
        })?;
        let header = index.header().ok_or_else(|| {
            ReaderError::Other(anyhow::anyhow!(
                "tabix index {} has no header",
                tbi.display()
            ))
        })?;
        let ref_ids = header
            .reference_sequence_names()
            .iter()
            .enumerate()
            .map(|(i, n)| (String::from_utf8_lossy(n.as_ref()).into_owned(), i))
            .collect();
        let dtypes = columns.iter().map(|(_, d)| *d).collect();
        let bgzf = open_bgzf(&path)?;
        Ok(Self {
            shared: Arc::new(MultiShared {
                path,
                index,
                ref_ids,
                columns,
                dtypes,
                genome,
            }),
            bgzf: Mutex::new(bgzf),
        })
    }

    /// Parse one BED line and fill each target sink across the clipped run.
    fn scatter_line(
        &self,
        line: &[u8],
        contig_name: &str,
        win_start: u64,
        win_end: u64,
        sinks: &mut [ColumnSinkMut<'_>],
    ) -> IoResult<()> {
        let text = std::str::from_utf8(line).map_err(|e| {
            ReaderError::Other(anyhow::anyhow!(
                "non-utf8 line in {}: {e}",
                self.shared.path.display()
            ))
        })?;
        let mut fields = text.trim_end().split('\t');
        let chrom = fields.next().unwrap_or_default();
        if chrom != contig_name {
            return Ok(()); // tabix bins may overlap neighbors; skip foreign contigs
        }
        let cols: Vec<&str> = std::iter::once(chrom).chain(fields).collect();
        let parse_u64 = |s: &str| {
            s.parse::<u64>().map_err(|e| {
                ReaderError::Other(anyhow::anyhow!(
                    "bad coord {s:?} in {}: {e}",
                    self.shared.path.display()
                ))
            })
        };
        let bed_start = parse_u64(cols.get(1).copied().unwrap_or_default())?;
        let bed_end = parse_u64(cols.get(2).copied().unwrap_or_default())?;

        let lo = bed_start.max(win_start);
        let hi = bed_end.min(win_end);
        if lo >= hi {
            return Ok(());
        }
        let (rel_lo, rel_hi) = ((lo - win_start) as usize, (hi - win_start) as usize);

        for (col_idx, (file_col, _dt)) in self.shared.columns.iter().enumerate() {
            let cell = match file_col {
                Some(file_col) => cols.get(*file_col).copied().ok_or_else(|| {
                    ReaderError::Other(anyhow::anyhow!(
                        "line has no column {} in {}",
                        file_col,
                        self.shared.path.display()
                    ))
                })?,
                None => "1",
            };
            sinks[col_idx].fill_run(rel_lo, rel_hi, cell)?;
        }
        Ok(())
    }
}

impl MultiValueReader for BedMultiReader {
    fn contigs(&self) -> &Genome {
        &self.shared.genome
    }

    fn columns(&self) -> &[Dtype] {
        &self.shared.dtypes
    }

    fn read_into(
        &self,
        contig_name: &str,
        start: u64,
        end: u64,
        sinks: &mut [ColumnSinkMut<'_>],
    ) -> IoResult<()> {
        if end <= start {
            return Ok(());
        }
        let Some(&ref_id) = self.shared.ref_ids.get(contig_name) else {
            return Ok(()); // contig absent from this BED -> leave as caller's fill
        };

        // tabix/csi intervals are 1-based inclusive; our range is 0-based half-open.
        let q_start = Position::try_from(start as usize + 1)
            .map_err(|e| ReaderError::Other(anyhow::anyhow!("bad start {start}: {e}")))?;
        let q_end = Position::try_from(end as usize)
            .map_err(|e| ReaderError::Other(anyhow::anyhow!("bad end {end}: {e}")))?;
        let interval = Interval::from(q_start..=q_end);

        let chunks = self
            .shared
            .index
            .query(ref_id, interval)
            .map_err(|source| ReaderError::Io {
                path: self.shared.path.clone(),
                source,
            })?;

        let mut bgzf = self.bgzf.lock().expect("bed multi reader mutex poisoned");
        let mut line = Vec::new();
        for chunk in chunks {
            bgzf.seek(chunk.start()).map_err(|source| ReaderError::Io {
                path: self.shared.path.clone(),
                source,
            })?;
            while bgzf.virtual_position() < chunk.end() {
                line.clear();
                let n = bgzf
                    .read_until(b'\n', &mut line)
                    .map_err(|source| ReaderError::Io {
                        path: self.shared.path.clone(),
                        source,
                    })?;
                if n == 0 {
                    break;
                }
                if line.first() == Some(&b'#') {
                    continue;
                }
                self.scatter_line(&line, contig_name, start, end, sinks)?;
            }
        }
        Ok(())
    }

    fn fork(&self) -> IoResult<Self> {
        let bgzf = open_bgzf(&self.shared.path)?;
        Ok(Self {
            shared: Arc::clone(&self.shared),
            bgzf: Mutex::new(bgzf),
        })
    }
}

/// Import every column named in `schema` from one tabix-indexed BED into its
/// own scalar track, in a single decode pass. `genome` supplies the contig
/// lengths (BED files declare none). Uncovered positions become the track's
/// zero fill.
pub fn from_bed_multi(
    store: &mut PbzStore,
    bed_gz: &Path,
    schema: &BedSchema,
    genome: Genome,
    config: Config,
) -> Result<Report> {
    let resolved = resolve(bed_gz, schema)?;

    let columns: Vec<(Option<usize>, Dtype)> = resolved
        .iter()
        .map(|r| (Some(r.file_col), r.dtype))
        .collect();
    let reader = BedMultiReader::open(bed_gz, columns, genome.clone())
        .map_err(|e| PbzError::Store(format!("open {}: {e}", bed_gz.display())))?;
    let specs = resolved
        .iter()
        .map(|r| {
            // Uncovered positions read back as zero; the on-disk fill must match the
            // zero-filled scratch buffers the pipeline writes, so all-gap chunks elide.
            let mut cfg = config.track_config(r.dtype, Some(zero_fill(r.dtype)), None, "sample");
            if let Some(desc) = &r.description {
                cfg = cfg.description(desc.clone());
            }
            (r.track_name.clone(), genome.clone(), cfg)
        })
        .collect();

    store.create_tracks_with(specs, move |tracks| {
        run_multi_pipeline(tracks, reader, &config)
    })
}

/// Import BED sources following the default shape table: BED3 becomes a bool
/// presence track, headerless BED4 a single value track, headered fields their
/// own tracks (2D over sources when there are several), and a single
/// multi-field source one wide 2D track whose columns are the fields. Any 2D
/// output requires `Config::column_dim`.
pub fn from_bed_matrix(
    store: &mut PbzStore,
    sources: &[pbzarr::import::Source],
    genome: Genome,
    options: &BedImportOptions,
    config: Config,
) -> Result<Report> {
    let resolved = resolve_sources(sources, options)?;
    let two_d = matches!(resolved, ResolvedImport::Wide { .. }) || sources.len() > 1;
    if two_d && config.column_dim.is_none() {
        return Err(PbzError::Metadata(
            "bed import: 2D output requires a column dimension name".into(),
        ));
    }

    match resolved {
        ResolvedImport::SingleTrack {
            track,
            dtype,
            column,
            labels,
        } => {
            // Uncovered positions read back as zero; the on-disk fill must match
            // the zero-filled scratch buffers the pipeline writes, so all-gap
            // chunks elide.
            let scalar = sources.len() <= 1 && config.column_dim.is_none();
            let column_labels = (!scalar).then_some(labels);
            let cfg = config.track_config(dtype, Some(zero_fill(dtype)), column_labels, "sample");
            let readers = open_readers(sources, &[(column, dtype)], &genome)?;
            run_single_or_matrix(store, vec![(track, genome, cfg)], readers, scalar, config)
        }
        ResolvedImport::Wide {
            track,
            dtype,
            columns,
            column_labels,
        } => {
            // `two_d` above already requires `column_dim` whenever `Wide` is
            // resolved, so the column axis is unconditional here.
            let cfg =
                config.track_config(dtype, Some(zero_fill(dtype)), Some(column_labels), "column");
            let reader_columns: Vec<(Option<usize>, Dtype)> =
                columns.into_iter().map(|c| (Some(c), dtype)).collect();
            let reader = BedMultiReader::open(&sources[0].path, reader_columns, genome.clone())
                .map_err(|e| PbzError::Store(format!("open {}: {e}", sources[0].path.display())))?;
            store.create_tracks_with(vec![(track, genome, cfg)], move |tracks| {
                run_wide_pipeline(tracks[0], reader, &config)
            })
        }
        ResolvedImport::PerField {
            fields,
            columns,
            labels,
        } => {
            let scalar = sources.len() <= 1 && config.column_dim.is_none();
            let specs = fields
                .iter()
                .map(|field| {
                    let column_labels = (!scalar).then(|| labels.clone());
                    let cfg = config.track_config(
                        field.dtype,
                        Some(zero_fill(field.dtype)),
                        column_labels,
                        "sample",
                    );
                    (field.name.clone(), genome.clone(), cfg)
                })
                .collect();
            let reader_columns: Vec<(Option<usize>, Dtype)> = columns
                .iter()
                .zip(&fields)
                .map(|(column, field)| (Some(*column), field.dtype))
                .collect();
            let readers = open_readers(sources, &reader_columns, &genome)?;
            run_single_or_matrix(store, specs, readers, scalar, config)
        }
    }
}

fn open_readers(
    sources: &[pbzarr::import::Source],
    columns: &[(Option<usize>, Dtype)],
    genome: &Genome,
) -> Result<Vec<BedMultiReader>> {
    sources
        .iter()
        .map(|source| {
            BedMultiReader::open(&source.path, columns.to_vec(), genome.clone())
                .map_err(|e| PbzError::Store(format!("open {}: {e}", source.path.display())))
        })
        .collect()
}

fn run_single_or_matrix(
    store: &mut PbzStore,
    specs: Vec<(String, Genome, TrackConfig)>,
    readers: Vec<BedMultiReader>,
    // Whether the specs built above are rank-1: source count alone isn't
    // enough (D-1: an explicit `column_dim` makes even a single source
    // rank-2), so the caller passes the same scalarity it used to build the
    // `TrackConfig`s, and `run_multi_pipeline`'s rank-1 requirement is
    // dispatched on that instead.
    scalar: bool,
    config: Config,
) -> Result<Report> {
    store.create_tracks_with(specs, move |tracks| {
        if scalar {
            let reader = readers
                .into_iter()
                .next()
                .ok_or_else(|| PbzError::Metadata("bed import lost its only reader".into()))?;
            run_multi_pipeline(tracks, reader, &config)
        } else {
            run_matrix_pipeline(tracks, readers, &config)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // Index-selector resolution is pure (no file read), so these run without htslib.

    #[test]
    fn index_selectors_resolve_and_default_nothing() {
        let schema = BedSchema(vec![
            BedColumnSpec::indexed(3, Dtype::I32, "cov"),
            BedColumnSpec::indexed(4, Dtype::Bool, "mask"),
        ]);
        let r = resolve(&PathBuf::from("unused.bed.gz"), &schema).unwrap();
        assert_eq!(r.len(), 2);
        assert_eq!(r[0].file_col, 3);
        assert_eq!(r[0].track_name, "cov");
        assert_eq!(r[1].dtype, Dtype::Bool);
    }

    #[test]
    fn coordinate_column_is_rejected() {
        let schema = BedSchema(vec![BedColumnSpec::indexed(2, Dtype::I32, "bad")]);
        assert!(resolve(&PathBuf::from("unused.bed.gz"), &schema).is_err());
    }

    #[test]
    fn duplicate_track_names_are_rejected() {
        let schema = BedSchema(vec![
            BedColumnSpec::indexed(3, Dtype::I32, "dup"),
            BedColumnSpec::indexed(4, Dtype::F32, "dup"),
        ]);
        assert!(resolve(&PathBuf::from("unused.bed.gz"), &schema).is_err());
    }

    #[test]
    fn empty_schema_is_rejected() {
        assert!(resolve(&PathBuf::from("unused.bed.gz"), &BedSchema(vec![])).is_err());
    }

    #[test]
    fn index_selector_without_track_name_is_rejected() {
        // Bypasses the `indexed()` constructor to hit the "no name to default to" branch.
        let schema = BedSchema(vec![BedColumnSpec {
            selector: ColumnSelector::Index(3),
            dtype: Dtype::I32,
            track_name: None,
            description: None,
        }]);
        assert!(resolve(&PathBuf::from("unused.bed.gz"), &schema).is_err());
    }
}
