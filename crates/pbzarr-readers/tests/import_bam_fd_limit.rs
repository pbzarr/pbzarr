//! A cohort import must fit in the soft fd limit when the source count is
//! more than half of it. This binary lowers RLIMIT_NOFILE for its whole
//! process, so it holds this one test only.
#![cfg(unix)]

mod bam_common;

use pbzarr::import::{Config, Source};
use pbzarr_readers::{DepthFilter, ImportMode, from_bam};
use tempfile::TempDir;

const FD_LIMIT: u64 = 64;
const SOURCES: usize = 40;

fn set_soft_fd_limit(limit: u64) {
    let mut lim = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: getrlimit and setrlimit only read and write the given struct.
    unsafe {
        assert_eq!(libc::getrlimit(libc::RLIMIT_NOFILE, &mut lim), 0);
        lim.rlim_cur = limit as libc::rlim_t;
        assert_eq!(libc::setrlimit(libc::RLIMIT_NOFILE, &lim), 0);
    }
}

#[test]
fn cohort_import_fits_a_low_fd_limit() {
    let fx = bam_common::fixture();
    let dir = TempDir::new().unwrap();
    let store = dir.path().join("cohort.pbz");
    let sources: Vec<Source> = (0..SOURCES)
        .map(|i| Source::labeled(&fx.bam, format!("s{i}")))
        .collect();

    set_soft_fd_limit(FD_LIMIT);
    from_bam(
        store.as_path(),
        "depth",
        &sources,
        ImportMode::Depth,
        DepthFilter::default(),
        None,
        Config::default(),
    )
    .unwrap();
}
