use color_eyre::Result;
use pbzarr::{PERBASE_ZARR_VERSION, PbzStore};
use serde::Serialize;

use crate::cli::InfoArgs;

#[derive(Serialize)]
struct StoreInfo {
    version: String,
    path: String,
    contigs: Vec<ContigEntry>,
    tracks: Vec<TrackEntry>,
}

#[derive(Serialize)]
struct ContigEntry {
    name: String,
    length: u64,
}

#[derive(Serialize)]
struct TrackEntry {
    name: String,
    dtype: String,
    chunk_size: u64,
    sample_chunk_size: Option<u64>,
    columns: Option<Vec<String>>,
    description: Option<String>,
    source: Option<String>,
}

pub fn run(args: InfoArgs) -> Result<()> {
    let store = PbzStore::open(&args.store)?;
    let mut contigs = Vec::new();
    for c in store.contigs() {
        contigs.push(ContigEntry {
            name: c.clone(),
            length: store.contig_length(c)?,
        });
    }
    let mut tracks = Vec::new();
    for name in store.tracks()? {
        let track = store.track(&name)?;
        let m = track.metadata();
        tracks.push(TrackEntry {
            name: name.clone(),
            dtype: m.dtype.clone(),
            chunk_size: m.chunk_size,
            sample_chunk_size: m.sample_chunk_size,
            columns: track.samples().map(|cs| cs.to_vec()),
            description: m.description.clone(),
            source: m.source.clone(),
        });
    }
    let info = StoreInfo {
        version: PERBASE_ZARR_VERSION.to_string(),
        path: args.store.display().to_string(),
        contigs,
        tracks,
    };
    if args.json {
        println!("{}", serde_json::to_string_pretty(&info)?);
    } else {
        print_human(&info);
    }
    Ok(())
}

fn print_human(info: &StoreInfo) {
    println!("perbase_zarr version: {}", info.version);
    println!("path: {}", info.path);
    println!("\ncontigs:");
    for c in &info.contigs {
        println!("  {}\t{}", c.name, c.length);
    }
    println!("\ntracks:");
    for t in &info.tracks {
        println!("  track: {}", t.name);
        println!("    dtype: {}", t.dtype);
        println!("    chunk_size: {}", t.chunk_size);
        if let Some(c) = t.sample_chunk_size {
            println!("    sample_chunk_size: {}", c);
        }
        if let Some(cols) = &t.columns {
            println!("    columns ({}): {}", cols.len(), cols.join(", "));
        }
        if let Some(d) = &t.description {
            println!("    description: {}", d);
        }
        if let Some(s) = &t.source {
            println!("    source: {}", s);
        }
    }
}
