use std::path::PathBuf;

/// One input file feeding one column of an imported track.
#[derive(Debug, Clone)]
pub struct Source {
    /// Path to the source file (d4, bigWig, BED, BAM/CRAM, ...).
    pub path: PathBuf,
    /// Explicit column label; falls back to the file stem via [`Source::label`]
    /// when unset.
    pub column_label: Option<String>,
}

impl Source {
    /// A source with no explicit label; `label()` falls back to the file stem.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            column_label: None,
        }
    }

    /// A source with an explicit column label, overriding the file-stem fallback.
    pub fn labeled(path: impl Into<PathBuf>, label: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            column_label: Some(label.into()),
        }
    }

    /// `column_label` if set, else the file stem, else the full path string.
    pub fn label(&self) -> String {
        self.column_label.clone().unwrap_or_else(|| {
            self.path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| self.path.to_string_lossy().into_owned())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_falls_back_to_file_stem() {
        let s = Source::new("/tmp/sampleA.final.bam");
        assert_eq!(s.label(), "sampleA.final");
        let l = Source::labeled("/tmp/x.bam", "S1");
        assert_eq!(l.label(), "S1");
    }
}
