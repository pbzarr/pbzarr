use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum BedError {
    #[error("file must be bgzipped (.bed.gz); plain BED not supported: {path}")]
    NotBgzipped { path: PathBuf },

    #[error("BED file is empty: {path}")]
    Empty { path: PathBuf },

    #[error(
        "unrecognized BED flavor at {path}:{line}: expected BED3 (3 cols) or bedGraph (4 cols, numeric col 4); got {got_cols} cols"
    )]
    FlavorAmbiguous {
        path: PathBuf,
        line: usize,
        got_cols: usize,
    },

    #[error("flavor changed mid-file at {path}:{line}: expected {expected} columns, got {got}")]
    FlavorMixed {
        path: PathBuf,
        line: usize,
        expected: usize,
        got: usize,
    },

    #[error(
        "BED must be sorted; {path}:{line}: {contig}:{this_start} follows {prev_contig}:{prev_start}"
    )]
    Unsorted {
        path: PathBuf,
        line: usize,
        contig: String,
        this_start: u64,
        prev_contig: String,
        prev_start: u64,
    },

    #[error(
        "BED has overlapping intervals on contig {contig} at {path}:{line}: interval [{this_start}, {this_end}) overlaps previous interval ending at {prev_end}\nhelp: pre-merge with `bedtools merge -i {path:?}`"
    )]
    Overlap {
        path: PathBuf,
        line: usize,
        contig: String,
        this_start: u64,
        this_end: u64,
        prev_end: u64,
    },

    #[error("BED references contig '{contig}' at {path}:{line} not present in --contigs / store")]
    UnknownContig {
        path: PathBuf,
        line: usize,
        contig: String,
    },

    #[error("{path}:{line} column 4: '{value}' is not a valid number")]
    ValueParse {
        path: PathBuf,
        line: usize,
        value: String,
    },

    #[error("{path}:{line}: value {value} cannot be represented as {dtype}")]
    ValueOutOfRange {
        path: PathBuf,
        line: usize,
        value: f64,
        dtype: &'static str,
    },

    #[error("{path}:{line}: zero-length interval [{start}, {start})")]
    ZeroLength {
        path: PathBuf,
        line: usize,
        start: u64,
    },

    #[error("{path}:{line}: reversed interval [{start}, {end})")]
    ReversedInterval {
        path: PathBuf,
        line: usize,
        start: u64,
        end: u64,
    },

    #[error(
        "{path}:{line}: interval extends past contig end ({contig} length={contig_length}, interval ends at {end})"
    )]
    PastContigEnd {
        path: PathBuf,
        line: usize,
        contig: String,
        contig_length: u64,
        end: u64,
    },

    #[error("{path}:{line}: malformed line (expected at least 3 tab-separated fields)")]
    Malformed { path: PathBuf, line: usize },

    #[error("I/O error reading {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}
