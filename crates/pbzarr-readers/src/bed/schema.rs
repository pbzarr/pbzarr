use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::BufRead;

use pbzarr::io::Dtype;
use pbzarr::{PbzError, Result};

use super::import::{BedSource, column_label};
use super::reader::open_bgzf;

#[derive(Debug, Clone, Copy)]
pub enum InferRows {
    Sample(usize),
    All,
}

pub struct BedImportOptions {
    pub fields: Option<Vec<String>>,
    pub dtype_overrides: BTreeMap<String, Dtype>,
    pub infer_rows: InferRows,
}

impl Default for BedImportOptions {
    fn default() -> Self {
        Self {
            fields: None,
            dtype_overrides: BTreeMap::new(),
            infer_rows: InferRows::Sample(1_000),
        }
    }
}

pub(super) struct ResolvedField {
    pub name: String,
    pub dtype: Dtype,
}

pub(super) struct ResolvedSource {
    pub columns: Vec<usize>,
    pub label: String,
}

pub(super) fn resolve_sources(
    sources: &[BedSource],
    options: &BedImportOptions,
) -> Result<(Vec<ResolvedField>, Vec<ResolvedSource>)> {
    let first = sources
        .first()
        .ok_or_else(|| PbzError::Metadata("bed matrix import: no sources".into()))?;
    let first_header = read_header(&first.path)?;
    let selected = options
        .fields
        .clone()
        .unwrap_or_else(|| first_header[3..].to_vec());
    if selected.is_empty() {
        return Err(PbzError::Metadata(
            "bed matrix import: no value fields selected".into(),
        ));
    }
    let mut seen = HashSet::new();
    if selected.iter().any(|name| !seen.insert(name.clone())) {
        return Err(PbzError::Metadata(
            "bed matrix import: duplicate selected field".into(),
        ));
    }
    let mut inference = vec![InferenceState::default(); selected.len()];
    let mut resolved = Vec::with_capacity(sources.len());
    for source in sources {
        let header = read_header(&source.path)?;
        let index: HashMap<_, _> = header
            .into_iter()
            .enumerate()
            .map(|(i, name)| (name, i))
            .collect();
        let columns = selected
            .iter()
            .map(|name| {
                index
                    .get(name)
                    .copied()
                    .filter(|index| *index >= 3)
                    .ok_or_else(|| {
                        PbzError::Metadata(format!(
                            "bed matrix import: field {name:?} missing from {}",
                            source.path.display()
                        ))
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        sample_cells(source, &columns, options.infer_rows, &mut inference)?;
        resolved.push(ResolvedSource {
            columns,
            label: column_label(source),
        });
    }
    let fields = selected
        .into_iter()
        .zip(inference)
        .map(|(name, inference)| {
            let dtype = options
                .dtype_overrides
                .get(&name)
                .copied()
                .map(Ok)
                .unwrap_or_else(|| inference.finish())?;
            Ok(ResolvedField { name, dtype })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((fields, resolved))
}

fn read_header(path: &std::path::Path) -> Result<Vec<String>> {
    let mut reader = open_bgzf(path).map_err(|error| PbzError::Store(error.to_string()))?;
    let mut line = Vec::new();
    reader
        .read_until(b'\n', &mut line)
        .map_err(|error| PbzError::Store(format!("read BED header {}: {error}", path.display())))?;
    let line = std::str::from_utf8(&line).map_err(|error| {
        PbzError::Metadata(format!(
            "BED header {} is not UTF-8: {error}",
            path.display()
        ))
    })?;
    let Some(header) = line.strip_prefix('#') else {
        return Err(PbzError::Metadata(format!(
            "BED {} needs a #-prefixed header",
            path.display()
        )));
    };
    let fields = header
        .trim_end()
        .split('\t')
        .map(str::to_owned)
        .collect::<Vec<_>>();
    if fields.len() < 4 || fields[..3] != ["chrom", "start", "end"] {
        return Err(PbzError::Metadata(format!(
            "BED {} has invalid coordinate header",
            path.display()
        )));
    }
    Ok(fields)
}

fn sample_cells(
    source: &BedSource,
    columns: &[usize],
    limit: InferRows,
    inference: &mut [InferenceState],
) -> Result<()> {
    let mut reader = open_bgzf(&source.path).map_err(|error| PbzError::Store(error.to_string()))?;
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
                inference[field].observe(cells.get(*column).ok_or_else(|| {
                    PbzError::Metadata(format!(
                        "BED {} has no column {column}",
                        source.path.display()
                    ))
                })?)?;
            }
            count += 1;
            if matches!(limit, InferRows::Sample(max) if count >= max) {
                break;
            }
        }
        line.clear();
    }
    Ok(())
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
    state.finish()
}

#[derive(Clone)]
struct InferenceState {
    bool_only: bool,
    saw_decimal: bool,
    f32_finite: bool,
    min: Option<i128>,
    max: Option<i128>,
}

impl Default for InferenceState {
    fn default() -> Self {
        Self {
            bool_only: true,
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
            "true" | "True" | "TRUE" => Some(1),
            "false" | "False" | "FALSE" => Some(0),
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

    fn finish(self) -> Result<Dtype> {
        if self.min.is_none() && !self.saw_decimal {
            return Err(PbzError::Metadata("bed type inference: no values".into()));
        }
        if self.bool_only {
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
    fn inference_state_discards_cells_after_observing_them() {
        let mut state = InferenceState::default();
        for value in ["-129", "128", "127"] {
            state.observe(value).unwrap();
        }
        assert_eq!(state.finish().unwrap(), Dtype::I16);
    }
}
