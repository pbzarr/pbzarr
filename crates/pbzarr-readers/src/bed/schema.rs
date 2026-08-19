use std::collections::{BTreeMap, HashSet};
use std::fs::File;
use std::io::BufRead;
use std::path::Path;

use log::debug;
use noodles_bgzf as bgzf;
use pbzarr::import::Source;
use pbzarr::io::{Dtype, ReaderError};
use pbzarr::{PbzError, Result};

/// Open a bgzipped file for reading.
pub(super) fn open_bgzf(path: &Path) -> std::result::Result<bgzf::io::Reader<File>, ReaderError> {
    let file = File::open(path).map_err(|source| ReaderError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(bgzf::io::Reader::new(file))
}

/// Read the `#`-prefixed header line and return the 0-based index of `name`.
pub fn column_index_by_name<P: AsRef<Path>>(
    bed_gz: P,
    name: &str,
) -> std::result::Result<usize, ReaderError> {
    let path = bed_gz.as_ref();
    let mut reader = open_bgzf(path)?;
    let mut line = Vec::new();
    reader
        .read_until(b'\n', &mut line)
        .map_err(|source| ReaderError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    let line = String::from_utf8_lossy(&line);
    let header = line.trim_start_matches('#').trim_end();
    header.split('\t').position(|c| c == name).ok_or_else(|| {
        ReaderError::Other(anyhow::anyhow!(
            "column {name:?} not in header of {}",
            path.display()
        ))
    })
}

#[derive(Debug, Clone, Copy)]
pub enum InferRows {
    Sample(usize),
    All,
}

pub struct BedImportOptions {
    /// Header field names to import; `None` selects every value field.
    pub fields: Option<Vec<String>>,
    /// Per-field dtype overrides for headered per-field imports.
    pub dtype_overrides: BTreeMap<String, Dtype>,
    pub infer_rows: InferRows,
    /// Output track name; required for BED3, headerless BED4, and wide imports.
    pub track: Option<String>,
    /// Value dtype for headerless BED4 input (skips inference).
    pub dtype: Option<Dtype>,
}

impl Default for BedImportOptions {
    fn default() -> Self {
        Self {
            fields: None,
            dtype_overrides: BTreeMap::new(),
            infer_rows: InferRows::Sample(1_000),
            track: None,
            dtype: None,
        }
    }
}

pub(super) struct ResolvedField {
    pub name: String,
    pub dtype: Dtype,
}

/// The output shape a set of sources + options resolves to. Sources are always
/// the column axis when there are several; a single multi-field source turns
/// its fields into the column axis instead.
pub(super) enum ResolvedImport {
    /// Headered: each selected field becomes its own track (2D over sources
    /// when there are several).
    PerField {
        fields: Vec<ResolvedField>,
        columns: Vec<usize>,
        labels: Vec<String>,
    },
    /// BED3 presence (`column` = None) or headerless BED4 (`column` = Some(3)):
    /// one track, scalar for one source, 2D over sources otherwise.
    SingleTrack {
        track: String,
        dtype: Dtype,
        column: Option<usize>,
        labels: Vec<String>,
    },
    /// One source, several fields: one 2D track whose columns are the fields.
    Wide {
        track: String,
        dtype: Dtype,
        columns: Vec<usize>,
        column_labels: Vec<String>,
    },
}

/// The shape of a BED file's first line: its header field names (if the line
/// is a `#`-prefixed header) and total column count.
pub struct BedLayout {
    pub header: Option<Vec<String>>,
    pub n_cols: usize,
}

/// Sniff a BED's layout from its first line.
pub fn read_bed_layout(path: &Path) -> Result<BedLayout> {
    let mut reader = open_bgzf(path).map_err(|error| PbzError::Store(error.to_string()))?;
    let mut line = Vec::new();
    reader
        .read_until(b'\n', &mut line)
        .map_err(|error| PbzError::Store(format!("read BED {}: {error}", path.display())))?;
    let text = std::str::from_utf8(&line)
        .map_err(|error| {
            PbzError::Metadata(format!("BED {} is not UTF-8: {error}", path.display()))
        })?
        .trim_end();
    if text.is_empty() {
        return Err(PbzError::Metadata(format!(
            "BED {} is empty",
            path.display()
        )));
    }
    if let Some(header) = text.strip_prefix('#') {
        let fields = header.split('\t').map(str::to_owned).collect::<Vec<_>>();
        if fields.len() < 3 || fields[..3] != ["chrom", "start", "end"] {
            return Err(PbzError::Metadata(format!(
                "BED {} has invalid coordinate header",
                path.display()
            )));
        }
        Ok(BedLayout {
            n_cols: fields.len(),
            header: Some(fields),
        })
    } else {
        let n_cols = text.split('\t').count();
        if n_cols < 3 {
            return Err(PbzError::Metadata(format!(
                "BED {} has fewer than 3 columns",
                path.display()
            )));
        }
        Ok(BedLayout {
            header: None,
            n_cols,
        })
    }
}

pub(super) fn resolve_sources(
    sources: &[Source],
    options: &BedImportOptions,
) -> Result<ResolvedImport> {
    let first = sources
        .first()
        .ok_or_else(|| PbzError::Metadata("bed import: no sources".into()))?;
    let layout = read_bed_layout(&first.path)?;
    for source in &sources[1..] {
        let other = read_bed_layout(&source.path)?;
        if other.n_cols != layout.n_cols {
            return Err(PbzError::Metadata(format!(
                "bed import: {} has {} columns but {} has {}",
                source.path.display(),
                other.n_cols,
                first.path.display(),
                layout.n_cols
            )));
        }
        if other.header != layout.header {
            return Err(PbzError::Metadata(format!(
                "bed import: {} header differs from {}",
                source.path.display(),
                first.path.display()
            )));
        }
    }
    let labels = sources.iter().map(Source::label).collect::<Vec<_>>();

    if layout.n_cols == 3 {
        if options.fields.is_some() {
            return Err(PbzError::Metadata(
                "bed import: BED3 input has no BED value columns to select".into(),
            ));
        }
        if options.dtype.is_some() {
            return Err(PbzError::Metadata(
                "bed import: BED3 input is always bool; drop the dtype".into(),
            ));
        }
        reject_overrides(options)?;
        return Ok(ResolvedImport::SingleTrack {
            track: required_track(options, "for BED3 input")?,
            dtype: Dtype::Bool,
            column: None,
            labels,
        });
    }

    let Some(header) = &layout.header else {
        if layout.n_cols > 4 {
            return Err(PbzError::Metadata(format!(
                "bed import: {} has {} BED value columns and no header; add a header line or use a schema",
                first.path.display(),
                layout.n_cols - 3
            )));
        }
        if options.fields.is_some() {
            return Err(PbzError::Metadata(
                "bed import: `--field` selection requires a #-prefixed header".into(),
            ));
        }
        reject_overrides(options)?;
        let dtype = match options.dtype {
            Some(dtype) => dtype,
            None => {
                let (mut states, exact) = infer_states(sources, &[3], options.infer_rows)?;
                states
                    .pop()
                    .ok_or_else(|| {
                        PbzError::Metadata("bed import: inference produced no dtype".into())
                    })?
                    .finish(exact)?
            }
        };
        return Ok(ResolvedImport::SingleTrack {
            track: required_track(options, "for headerless input")?,
            dtype,
            column: Some(3),
            labels,
        });
    };

    if options.dtype.is_some() {
        return Err(PbzError::Metadata(
            "bed import: a bare dtype only applies to headerless BED4 input".into(),
        ));
    }
    let selected = options
        .fields
        .clone()
        .unwrap_or_else(|| header[3..].to_vec());
    if selected.is_empty() {
        return Err(PbzError::Metadata(
            "bed import: no BED value columns selected".into(),
        ));
    }
    let mut seen = HashSet::new();
    if selected.iter().any(|name| !seen.insert(name.clone())) {
        return Err(PbzError::Metadata(
            "bed import: duplicate selected BED column".into(),
        ));
    }
    let columns = selected
        .iter()
        .map(|name| {
            header
                .iter()
                .position(|field| field == name)
                .filter(|index| *index >= 3)
                .ok_or_else(|| {
                    PbzError::Metadata(format!(
                        "bed import: BED column {name:?} is not a value BED column of the header"
                    ))
                })
        })
        .collect::<Result<Vec<_>>>()?;

    if sources.len() == 1 && selected.len() > 1 {
        reject_overrides(options)?;
        let (states, exact) = infer_states(sources, &columns, options.infer_rows)?;
        let joint = states
            .iter()
            .cloned()
            .reduce(|mut joint, state| {
                joint.merge(&state);
                joint
            })
            .ok_or_else(|| PbzError::Metadata("bed import: inference produced no dtype".into()))?;
        // A true/false word column can only ever be bool; the import parser
        // will not read the words back as a number, so refuse to fold one into
        // a numeric wide track before any track is created. Checked on the
        // merged state ahead of finish() so the error can name the fields.
        if joint.saw_word_bool && !joint.bool_only {
            let word_bool_fields = selected
                .iter()
                .zip(&states)
                .filter(|(_, state)| state.saw_word_bool)
                .map(|(name, _)| format!("{name:?}"))
                .collect::<Vec<_>>();
            return Err(PbzError::Metadata(format!(
                "bed import: BED column(s) {} hold true/false and cannot join a numeric wide track; drop them with a `--field` selection or import them via a schema",
                word_bool_fields.join(", ")
            )));
        }
        let dtype = joint.finish(exact)?;
        return Ok(ResolvedImport::Wide {
            track: required_track(options, "when several BED columns form one wide track")?,
            dtype,
            columns,
            column_labels: selected,
        });
    }

    if options.track.is_some() {
        return Err(PbzError::Metadata(
            "bed import: a track name only applies to single-track imports; BED columns name their own tracks"
                .into(),
        ));
    }
    let (states, exact) = infer_states(sources, &columns, options.infer_rows)?;
    let fields = selected
        .into_iter()
        .zip(states)
        .map(|(name, state)| {
            let dtype = options
                .dtype_overrides
                .get(&name)
                .copied()
                .map(Ok)
                .unwrap_or_else(|| state.finish(exact))?;
            Ok(ResolvedField { name, dtype })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(ResolvedImport::PerField {
        fields,
        columns,
        labels,
    })
}

fn required_track(options: &BedImportOptions, context: &str) -> Result<String> {
    options.track.clone().ok_or_else(|| {
        PbzError::Metadata(format!("bed import: a track name is required {context}"))
    })
}

fn reject_overrides(options: &BedImportOptions) -> Result<()> {
    if options.dtype_overrides.is_empty() {
        Ok(())
    } else {
        Err(PbzError::Metadata(
            "bed import: per-BED-column dtype overrides only apply to headered per-BED-column imports"
                .into(),
        ))
    }
}

/// Per-field inference states plus whether every source was scanned to EOF.
/// Only an exhaustive scan knows the true min/max, so only it may min-width.
fn infer_states(
    sources: &[Source],
    columns: &[usize],
    limit: InferRows,
) -> Result<(Vec<InferenceState>, bool)> {
    let mut states = vec![InferenceState::default(); columns.len()];
    let mut exact = true;
    for source in sources {
        exact &= sample_columns(&source.path, columns, limit, &mut states)?;
    }
    Ok((states, exact))
}

/// Sample `bed_gz` and infer a dtype per file column (0-based, `>= 3`).
pub fn infer_bed_dtypes(
    bed_gz: &Path,
    columns: &[usize],
    infer_rows: InferRows,
) -> Result<Vec<Dtype>> {
    let mut states = vec![InferenceState::default(); columns.len()];
    let exact = sample_columns(bed_gz, columns, infer_rows, &mut states)?;
    let dtypes = states
        .into_iter()
        .map(|state| state.finish(exact))
        .collect::<Result<Vec<_>>>()?;
    log_inferred(&[bed_gz.to_path_buf()], columns, &dtypes, exact);
    Ok(dtypes)
}

/// Report what inference decided, and how it got there. A sampled scan only
/// sees `infer_rows` records, so it picks conservative classes; an exhaustive
/// scan min-widths instead. Which one ran explains any surprising dtype.
fn log_inferred(sources: &[std::path::PathBuf], columns: &[usize], dtypes: &[Dtype], exact: bool) {
    debug!(
        "bed dtype inference over {} source(s) ({}): columns {columns:?} -> {dtypes:?}",
        sources.len(),
        if exact {
            "exhaustive scan"
        } else {
            "sampled, conservative classes"
        }
    );
}

/// Sample every BED source into one inference state per file column.
///
/// Observations are combined before choosing each dtype, so the result covers
/// values from every source rather than promoting independently inferred
/// dtypes after the fact.
pub fn infer_bed_dtypes_for_sources(
    sources: &[Source],
    columns: &[usize],
    infer_rows: InferRows,
) -> Result<Vec<Dtype>> {
    let (states, exact) = infer_states(sources, columns, infer_rows)?;
    let dtypes = states
        .into_iter()
        .map(|state| state.finish(exact))
        .collect::<Result<Vec<_>>>()?;
    let paths = sources.iter().map(|s| s.path.clone()).collect::<Vec<_>>();
    log_inferred(&paths, columns, &dtypes, exact);
    Ok(dtypes)
}

/// Returns true when the scan reached EOF (saw every record) rather than
/// stopping at the sample limit.
fn sample_columns(
    path: &Path,
    columns: &[usize],
    limit: InferRows,
    inference: &mut [InferenceState],
) -> Result<bool> {
    let mut reader = open_bgzf(path).map_err(|error| PbzError::Store(error.to_string()))?;
    let mut line = Vec::new();
    let mut count = 0usize;
    while reader
        .read_until(b'\n', &mut line)
        .map_err(|error| PbzError::Store(error.to_string()))?
        != 0
    {
        if !line.starts_with(b"#") {
            let text = std::str::from_utf8(&line)
                .map_err(|error| PbzError::Metadata(error.to_string()))?;
            let cells = text.trim_end().split('\t').collect::<Vec<_>>();
            for (field, column) in columns.iter().enumerate() {
                let value = match cells.get(*column) {
                    Some(value) => *value,
                    None => {
                        let one_based = column.checked_add(1).ok_or_else(|| {
                            PbzError::Metadata(format!(
                                "BED column index {column} cannot be represented as a one-based column number"
                            ))
                        })?;
                        return Err(PbzError::Metadata(format!(
                            "BED {}: BED column {one_based} (one-based) is outside the {}-column input",
                            path.display(),
                            cells.len()
                        )));
                    }
                };
                inference[field].observe(value)?;
            }
            count += 1;
            if matches!(limit, InferRows::Sample(max) if count >= max) {
                return Ok(false);
            }
        }
        line.clear();
    }
    Ok(true)
}

#[cfg(test)]
fn infer_dtype<I, S>(cells: I) -> Result<Dtype>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut state = InferenceState::default();
    for cell in cells {
        state.observe(cell.as_ref())?;
    }
    state.finish(true)
}

#[derive(Clone)]
struct InferenceState {
    bool_only: bool,
    /// The column held a `true`/`false` word literal; such cells only parse
    /// back as bool at import time, never as a number.
    saw_word_bool: bool,
    saw_decimal: bool,
    f32_finite: bool,
    min: Option<i128>,
    max: Option<i128>,
}

impl Default for InferenceState {
    fn default() -> Self {
        Self {
            bool_only: true,
            saw_word_bool: false,
            saw_decimal: false,
            f32_finite: true,
            min: None,
            max: None,
        }
    }
}

impl InferenceState {
    fn observe(&mut self, raw: &str) -> Result<()> {
        let cell = raw.trim();
        if cell.is_empty() {
            return Err(PbzError::Metadata("bed type inference: empty value".into()));
        }
        let integer = match cell {
            "true" | "True" | "TRUE" => {
                self.saw_word_bool = true;
                Some(1)
            }
            "false" | "False" | "FALSE" => {
                self.saw_word_bool = true;
                Some(0)
            }
            _ => cell.parse::<i128>().ok(),
        };
        if let Some(value) = integer {
            if value != 0 && value != 1 {
                self.bool_only = false;
            }
            self.min = Some(self.min.map_or(value, |min| min.min(value)));
            self.max = Some(self.max.map_or(value, |max| max.max(value)));
            if cell.parse::<f32>().map(|value| value.is_finite()) != Ok(true)
                && !matches!(cell, "true" | "True" | "TRUE" | "false" | "False" | "FALSE")
            {
                self.f32_finite = false;
            }
            return Ok(());
        }
        self.bool_only = false;
        self.saw_decimal = true;
        let value = cell.parse::<f64>().map_err(|error| {
            PbzError::Metadata(format!(
                "bed type inference: parse {cell:?} as number: {error}"
            ))
        })?;
        if !value.is_finite() {
            return Err(PbzError::Metadata(format!(
                "bed type inference: non-finite value {cell:?}"
            )));
        }
        if cell.parse::<f32>().map(|value| value.is_finite()) != Ok(true) {
            self.f32_finite = false;
        }
        Ok(())
    }

    /// Combine two sampled states; joint (wide) inference merges the per-field
    /// states so one dtype covers every field.
    fn merge(&mut self, other: &Self) {
        self.bool_only &= other.bool_only;
        self.saw_word_bool |= other.saw_word_bool;
        self.saw_decimal |= other.saw_decimal;
        self.f32_finite &= other.f32_finite;
        self.min = match (self.min, other.min) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        self.max = match (self.max, other.max) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => a.or(b),
        };
    }

    /// Pick a dtype. `exact` means the scan saw every record: only then is the
    /// observed min/max the true range, so only then min-width. A sample bounds
    /// nothing (head-of-file coverage says nothing about peak coverage), so a
    /// sampled scan returns the conservative class instead: bool only for
    /// true/false words, int32 for integers, float32 for decimals.
    fn finish(self, exact: bool) -> Result<Dtype> {
        if self.min.is_none() && !self.saw_decimal {
            return Err(PbzError::Metadata("bed type inference: no values".into()));
        }
        if self.saw_word_bool && !self.bool_only {
            return Err(PbzError::Metadata(
                "bed type inference: column mixes true/false with numbers".into(),
            ));
        }
        if self.bool_only && (exact || self.saw_word_bool) {
            return Ok(Dtype::Bool);
        }
        if self.saw_decimal {
            return Ok(if self.f32_finite {
                Dtype::F32
            } else {
                Dtype::F64
            });
        }
        let min = self
            .min
            .ok_or_else(|| PbzError::Metadata("bed type inference: no integers".into()))?;
        let max = self
            .max
            .ok_or_else(|| PbzError::Metadata("bed type inference: no integers".into()))?;
        if !exact {
            return if min >= i128::from(i32::MIN) && max <= i128::from(i32::MAX) {
                Ok(Dtype::I32)
            } else if min >= 0 && max <= u32::MAX as i128 {
                Ok(Dtype::U32)
            } else {
                Err(PbzError::Metadata(format!(
                    "bed type inference: integer range {min}..{max} exceeds int32/uint32"
                )))
            };
        }
        if min >= 0 {
            return match max {
                0..=255 => Ok(Dtype::U8),
                256..=65_535 => Ok(Dtype::U16),
                65_536..=4_294_967_295 => Ok(Dtype::U32),
                _ => Err(PbzError::Metadata(format!(
                    "bed type inference: integer {max} exceeds uint32"
                ))),
            };
        }
        match (min, max) {
            (-128..=-1, ..=127) => Ok(Dtype::I8),
            (-32_768..=-1, ..=32_767) => Ok(Dtype::I16),
            (-2_147_483_648..=-1, ..=2_147_483_647) => Ok(Dtype::I32),
            _ => Err(PbzError::Metadata(format!(
                "bed type inference: signed range {min}..{max} exceeds int32"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infers_smallest_integer_dtype() {
        assert_eq!(infer_dtype(["0", "255"]).unwrap(), Dtype::U8);
        assert_eq!(infer_dtype(["-128", "127"]).unwrap(), Dtype::I8);
        assert_eq!(infer_dtype(["-129", "128"]).unwrap(), Dtype::I16);
        assert_eq!(infer_dtype(["65536"]).unwrap(), Dtype::U32);
    }

    #[test]
    fn infers_bool_only_for_boolean_cells() {
        assert_eq!(infer_dtype(["true", "0", "1"]).unwrap(), Dtype::Bool);
        assert_eq!(infer_dtype(["0", "2"]).unwrap(), Dtype::U8);
    }

    #[test]
    fn merge_widens_to_cover_both_ranges() {
        let mut a = InferenceState::default();
        a.observe("3").unwrap();
        let mut b = InferenceState::default();
        b.observe("70000").unwrap();
        a.merge(&b);
        assert_eq!(a.finish(true).unwrap(), Dtype::U32);
    }

    #[test]
    fn sampled_scan_is_class_only_and_conservative() {
        let mut ints = InferenceState::default();
        for cell in ["0", "1", "2"] {
            ints.observe(cell).unwrap();
        }
        assert_eq!(ints.clone().finish(false).unwrap(), Dtype::I32);
        assert_eq!(ints.finish(true).unwrap(), Dtype::U8);

        // Numeric 0/1 could be a truncated count; only true/false words prove bool.
        let mut numeric_flags = InferenceState::default();
        numeric_flags.observe("0").unwrap();
        numeric_flags.observe("1").unwrap();
        assert_eq!(numeric_flags.finish(false).unwrap(), Dtype::I32);

        let mut word_flags = InferenceState::default();
        word_flags.observe("false").unwrap();
        assert_eq!(word_flags.finish(false).unwrap(), Dtype::Bool);
    }

    #[test]
    fn words_mixed_with_numbers_are_rejected() {
        let mut state = InferenceState::default();
        state.observe("false").unwrap();
        state.observe("7").unwrap();
        assert!(state.clone().finish(true).is_err());
        assert!(state.finish(false).is_err());
    }
}
