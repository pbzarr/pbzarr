// Smoke-test helper: write a v0.1 PBZ store. Used by cross-language round-trip checks.
use ndarray::Array2;
use pbzarr::{PbzStore, TrackConfig};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).unwrap_or_else(|| "/tmp/rt2.pbz.zarr".into());
    let p = Path::new(&path);
    if p.exists() {
        std::fs::remove_dir_all(p)?;
    }
    let store = PbzStore::builder(p)
        .contigs(&["chr1".to_string(), "chr2".to_string()], &[100_000, 50_000])
        .coordinate_space("GRCh38")
        .create()?;
    let cfg = TrackConfig {
        dtype: "uint16".into(),
        samples: Some(vec!["A".into(), "B".into()]),
        sample_chunk_size: 2,
        ..Default::default()
    };
    let track = store.create_track("depth", cfg)?;
    let data: Array2<u16> = Array2::from_shape_fn((100_000, 2), |(i, j)| (i * 2 + j) as u16);
    track.write_chunk::<u16>("chr1", 0, data)?;
    println!("rust write ok: {}", path);
    Ok(())
}
