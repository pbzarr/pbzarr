//! D4-specific ingest glue. `import_d4` body is filled in by Task 11.

use std::path::PathBuf;

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct D4Source {
    pub path: PathBuf,
    pub sample_label: Option<String>,
}

// `import_d4` is intentionally absent at this task — the function depends on
// `PbzStore` which doesn't exist yet. Task 4 re-enables the `import_d4`
// re-export (still a stub at that point); Task 11 implements the body.
