//! Pipeline body is implemented in Task 10. This file currently provides
//! the public types only.

use std::sync::Arc;

pub trait ProgressSink: Send + Sync {
    fn tick(&self, _bytes: u64) {}
    fn done(&self) {}
}

pub struct ImportConfig {
    pub workers: usize,
    pub chunk_size: Option<usize>,
    pub column_chunk_size: Option<usize>,
    pub progress: Option<Arc<dyn ProgressSink>>,
}

impl Default for ImportConfig {
    fn default() -> Self {
        Self {
            workers: 4,
            chunk_size: None,
            column_chunk_size: None,
            progress: None,
        }
    }
}

pub struct ImportReport {
    pub contigs_written: usize,
    pub bytes_written: u64,
}
