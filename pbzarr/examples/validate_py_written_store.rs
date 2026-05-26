//! Open a pbz store at the path passed on argv and validate the layout
//! matches what the Python `create_store` + `create_track` wrote.
//!
//! Used by `pbzarr-python/tests/test_python_to_rust_roundtrip.py` to confirm
//! the Python writer produces a layout the Rust library can read.

use ndarray::Ix2;
use pbzarr::{PbzStore, Region};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: validate_py_written_store <path>")?;

    let store = PbzStore::open(&path)?;

    let names: Vec<String> = store
        .genome()
        .contigs()
        .iter()
        .map(|c| c.name.clone())
        .collect();
    assert_eq!(names, vec!["chr1".to_string(), "chr2".to_string()]);
    assert_eq!(store.coordinate_space(), Some("GRCh38"));

    let mut tracks: Vec<String> = store.track_names().map(|s| s.to_owned()).collect();
    tracks.sort();
    assert!(tracks.contains(&"mask".to_string()), "expected mask track");
    assert!(
        tracks.contains(&"depth".to_string()),
        "expected depth track"
    );

    // read depth's chr1; Python wrote (i, i*2, i*3) for i in 0..100
    let depth = store.track("depth").unwrap();
    let chr1 = store.genome().id("chr1").unwrap();
    let region = Region {
        contig: chr1,
        start: 0,
        end: 100,
    };
    let got = depth.read_region::<u16>(&region)?;
    let arr = got.into_dimensionality::<Ix2>()?;
    for i in 0..100 {
        assert_eq!(arr[[i, 0]], i as u16, "depth[{i},0]");
        assert_eq!(arr[[i, 1]], (i * 2) as u16, "depth[{i},1]");
        assert_eq!(arr[[i, 2]], (i * 3) as u16, "depth[{i},2]");
    }

    println!("Python→Rust round-trip OK");
    Ok(())
}
