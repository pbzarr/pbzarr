use std::path::PathBuf;

use crate::{PbzError, Result};

/// One input file feeding one column of an imported track.
#[derive(Debug, Clone)]
pub struct Source {
    /// Path to the source file (d4, bigWig, BED, BAM/CRAM, ...).
    pub path: PathBuf,
    /// Explicit column label; falls back to the file name without its format
    /// extension via [`Source::label`] when unset.
    pub column_label: Option<String>,
}

impl Source {
    /// A source with no explicit label; `label()` falls back to the file name.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            column_label: None,
        }
    }

    /// A source with an explicit column label, overriding the file-name fallback.
    pub fn labeled(path: impl Into<PathBuf>, label: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            column_label: Some(label.into()),
        }
    }

    /// `column_label` if set, else the file name without its format
    /// extension, else the full path string.
    ///
    /// A compression extension (`.gz`, `.bgz`) goes together with the format
    /// extension before it, so `s1.bed.gz` gives `s1`.
    pub fn label(&self) -> String {
        self.column_label.clone().unwrap_or_else(|| {
            let Some(name) = self.path.file_name() else {
                return self.path.to_string_lossy().into_owned();
            };
            let mut name = name.to_string_lossy().into_owned();
            for _ in 0..2 {
                let Some((stem, ext)) = name.rsplit_once('.') else {
                    break;
                };
                if stem.is_empty() {
                    break;
                }
                let compressed = matches!(ext.to_ascii_lowercase().as_str(), "gz" | "bgz");
                name.truncate(stem.len());
                if !compressed {
                    break;
                }
            }
            name
        })
    }

    /// Refuse a source list in which only some sources have a label.
    pub fn check_labels(sources: &[Source]) -> Result<()> {
        let labeled = sources.iter().filter(|s| s.column_label.is_some()).count();
        if labeled != 0 && labeled != sources.len() {
            return Err(PbzError::InvalidArgument(
                "sources must either all provide labels or none do".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_drops_the_format_extension() {
        assert_eq!(
            Source::new("/tmp/sampleA.final.bam").label(),
            "sampleA.final"
        );
        assert_eq!(Source::new("s1.bed.gz").label(), "s1");
        assert_eq!(Source::new("s1.bigwig").label(), "s1");
        assert_eq!(Source::labeled("/tmp/x.bam", "S1").label(), "S1");
    }
}
