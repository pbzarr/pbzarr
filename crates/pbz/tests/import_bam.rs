use std::path::Path;

use tempfile::TempDir;

mod common;
use common::{run_pbz, stdout_of};

#[test]
fn import_bam_then_view_a_region() {
    let dir = TempDir::new().unwrap();
    let bam = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/bam/reads.bam");
    let store = dir.path().join("depth.pbz");

    stdout_of(&run_pbz([
        "import".into(),
        "bam".into(),
        store.as_os_str().to_owned(),
        bam.into_os_string(),
        "--track".into(),
        "depth".into(),
        "--no-progress".into(),
    ]));

    // fixtures/bam/README.md derives this depth: r1a and r5 cover [10,30),
    // r7 adds [12,22), and r1b's overlap with r1a counts once.
    let view = run_pbz([
        "view".into(),
        store.into_os_string(),
        "-r".into(),
        "ref1:10-30".into(),
    ]);
    assert_eq!(
        stdout_of(&view),
        "#chrom\tstart\tend\tdepth\n\
         ref1\t10\t12\t2\n\
         ref1\t12\t22\t3\n\
         ref1\t22\t30\t2\n"
    );
}
