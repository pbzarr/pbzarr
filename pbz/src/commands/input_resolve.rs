use color_eyre::Result;
use color_eyre::eyre::eyre;
use std::collections::HashSet;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputFormat {
    D4,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputSpec {
    pub path: PathBuf,
    pub column_name: String,
    pub format: InputFormat,
}

/// Parse a single input spec of the form `path` or `path:COLUMN_NAME`.
///
/// Splits on the last `:`. If no `:` is present, the column name defaults to
/// the file stem with `.d4` / `.d4.gz` stripped. Format is auto-detected by
/// extension.
pub fn parse_input_spec(s: &str) -> Result<InputSpec> {
    let (path_str, column_name) = match s.rfind(':') {
        Some(idx) => {
            let path_part = &s[..idx];
            let col_part = &s[idx + 1..];
            if path_part.is_empty() {
                return Err(eyre!("input spec has empty path: {s}"));
            }
            if col_part.is_empty() {
                return Err(eyre!("input spec has empty column name: {s}"));
            }
            (path_part.to_string(), Some(col_part.to_string()))
        }
        None => (s.to_string(), None),
    };

    let path = PathBuf::from(&path_str);
    let format = detect_format(&path_str)?;

    let column_name = match column_name {
        Some(c) => c,
        None => default_column_name(&path_str)?,
    };

    Ok(InputSpec {
        path,
        column_name,
        format,
    })
}

fn detect_format(path: &str) -> Result<InputFormat> {
    let lower = path.to_ascii_lowercase();
    if lower.ends_with(".d4") || lower.ends_with(".d4.gz") {
        Ok(InputFormat::D4)
    } else {
        Err(eyre!(
            "unsupported input extension for {path}: only .d4 and .d4.gz are recognized"
        ))
    }
}

fn default_column_name(path: &str) -> Result<String> {
    // Take just the file name component, then strip recognized suffixes.
    let file_name = Path::new(path)
        .file_name()
        .and_then(|os| os.to_str())
        .ok_or_else(|| eyre!("cannot derive column name from path: {path}"))?;

    let lower = file_name.to_ascii_lowercase();
    let stem = if lower.ends_with(".d4.gz") {
        &file_name[..file_name.len() - ".d4.gz".len()]
    } else if lower.ends_with(".d4") {
        &file_name[..file_name.len() - ".d4".len()]
    } else {
        return Err(eyre!(
            "cannot derive column name from {file_name}: unrecognized extension"
        ));
    };

    if stem.is_empty() {
        return Err(eyre!(
            "cannot derive column name from {file_name}: empty stem"
        ));
    }

    Ok(stem.to_string())
}

/// Parse a filelist where each line is an input spec (path or path:column).
/// Blank lines and lines starting with `#` are ignored. Relative paths are
/// resolved against the filelist's parent directory rather than the CWD.
pub fn parse_filelist(path: &Path) -> Result<Vec<InputSpec>> {
    let parent = path
        .parent()
        .map(|p| {
            if p.as_os_str().is_empty() {
                PathBuf::from(".")
            } else {
                p.to_path_buf()
            }
        })
        .unwrap_or_else(|| PathBuf::from("."));

    let file = std::fs::File::open(path)
        .map_err(|e| eyre!("failed to open filelist {}: {e}", path.display()))?;
    let reader = BufReader::new(file);

    let mut specs = Vec::new();
    for (lineno, line) in reader.lines().enumerate() {
        let line =
            line.map_err(|e| eyre!("failed to read filelist {}:{}: {e}", path.display(), lineno))?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let mut spec = parse_input_spec(trimmed).map_err(|e| {
            eyre!(
                "failed to parse filelist {}:{}: {e}",
                path.display(),
                lineno + 1
            )
        })?;

        if spec.path.is_relative() {
            spec.path = parent.join(&spec.path);
        }
        specs.push(spec);
    }

    Ok(specs)
}

/// Verify that every spec has a unique column name.
pub fn check_unique(specs: &[InputSpec]) -> Result<()> {
    let mut seen = HashSet::with_capacity(specs.len());
    for spec in specs {
        if !seen.insert(spec.column_name.as_str()) {
            return Err(eyre!(
                "duplicate column name '{}' in input specs",
                spec.column_name
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn parses_path_only() {
        let s = parse_input_spec("foo/bar.d4").unwrap();
        assert_eq!(s.path, PathBuf::from("foo/bar.d4"));
        assert_eq!(s.column_name, "bar");
        assert_eq!(s.format, InputFormat::D4);
    }

    #[test]
    fn parses_path_with_column() {
        let s = parse_input_spec("foo/bar.d4:NAME").unwrap();
        assert_eq!(s.path, PathBuf::from("foo/bar.d4"));
        assert_eq!(s.column_name, "NAME");
    }

    #[test]
    fn last_colon_is_separator() {
        let s = parse_input_spec("a:b:c.d4:NAME").unwrap();
        assert_eq!(s.path, PathBuf::from("a:b:c.d4"));
        assert_eq!(s.column_name, "NAME");
    }

    #[test]
    fn d4_gz_stem() {
        let s = parse_input_spec("/abs/sample.d4.gz").unwrap();
        assert_eq!(s.path, PathBuf::from("/abs/sample.d4.gz"));
        assert_eq!(s.column_name, "sample");
    }

    #[test]
    fn unknown_extension_errors() {
        let err = parse_input_spec("a.bw").unwrap_err();
        assert!(
            err.to_string().contains("unsupported") || err.to_string().contains("unrecognized")
        );
    }

    #[test]
    fn filelist_parses() {
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("list.txt");
        let mut f = std::fs::File::create(&p).unwrap();
        writeln!(f, "# header").unwrap();
        writeln!(f, "a.d4").unwrap();
        writeln!(f).unwrap();
        writeln!(f, "sub/b.d4:RENAMED").unwrap();
        drop(f);
        let v = parse_filelist(&p).unwrap();
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].path, p.parent().unwrap().join("a.d4"));
        assert_eq!(v[0].column_name, "a");
        assert_eq!(v[1].path, p.parent().unwrap().join("sub/b.d4"));
        assert_eq!(v[1].column_name, "RENAMED");
    }

    #[test]
    fn duplicates_error() {
        let specs = vec![
            InputSpec {
                path: PathBuf::from("a.d4"),
                column_name: "x".into(),
                format: InputFormat::D4,
            },
            InputSpec {
                path: PathBuf::from("b.d4"),
                column_name: "x".into(),
                format: InputFormat::D4,
            },
        ];
        assert!(check_unique(&specs).is_err());
    }

    #[test]
    fn unique_passes() {
        let specs = vec![
            InputSpec {
                path: PathBuf::from("a.d4"),
                column_name: "x".into(),
                format: InputFormat::D4,
            },
            InputSpec {
                path: PathBuf::from("b.d4"),
                column_name: "y".into(),
                format: InputFormat::D4,
            },
        ];
        assert!(check_unique(&specs).is_ok());
    }
}
