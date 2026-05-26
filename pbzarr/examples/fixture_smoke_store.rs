//! Build a small pbz store for the Python round-trip script.
//!
//! Usage: cargo run --example fixture_smoke_store -- <out.pbz>

use ndarray::{Array1, Array2};
use pbzarr::io::Dtype;
use pbzarr::{Contig, Genome, PbzStore, Region, TrackConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: fixture_smoke_store <out.pbz>")?;

    let genome = Genome::new(vec![
        Contig { name: "chr1".into(), length: 2_000 },
        Contig { name: "chr2".into(), length: 1_000 },
    ])?;
    let mut store = PbzStore::create(&path, genome, Some("GRCh38".into()))?;

    // scalar track
    store.create_track("mask", TrackConfig::scalar(Dtype::Bool))?;
    let chr1 = store.genome().id("chr1").unwrap();
    let chr2 = store.genome().id("chr2").unwrap();

    let region = Region { contig: chr1, start: 0, end: 2_000 };
    let mut m = Array1::<bool>::from_elem(2_000, false);
    for i in 0..2_000 {
        if i % 7 == 0 {
            m[i] = true;
        }
    }
    store.track("mask").unwrap().write_region(&region, m.view().into_dyn())?;

    // cohort track
    store.create_track(
        "depth",
        TrackConfig::cohort(Dtype::U16, vec!["A".into(), "B".into(), "C".into()]),
    )?;
    for (cid, len) in [(chr1, 2_000u64), (chr2, 1_000u64)] {
        let mut d = Array2::<u16>::zeros((len as usize, 3));
        for i in 0..(len as usize) {
            d[[i, 0]] = i as u16;
            d[[i, 1]] = (i * 2) as u16;
            d[[i, 2]] = (i * 3) as u16;
        }
        let region = Region { contig: cid, start: 0, end: len };
        store.track("depth").unwrap().write_region(&region, d.view().into_dyn())?;
    }

    println!("wrote {}", path);
    Ok(())
}
