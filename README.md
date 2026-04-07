# pbzarr

A Rust library for PBZ (Per-Base Zarr) — a Zarr v3 convention for storing per-base resolution genomic data such as read depths, methylation levels, and boolean masks.

PBZ is a modern alternative to D4 and bigWig. This crate handles store layout, metadata, region parsing, and chunk I/O, delegating array storage and compression to [`zarrs`](https://crates.io/crates/zarrs).

## Installation

```toml
[dependencies]
pbzarr = "0.1"
```

## Quick Start

```rust
use pbzarr::{PbzStore, TrackConfig};
use ndarray::Array2;

// Create a store
let contigs = vec!["chr1".into(), "chr2".into()];
let lengths = vec![248_956_422, 242_193_529];
let store = PbzStore::create("sample.pbz.zarr", &contigs, &lengths)?;

// Create a track
let config = TrackConfig::new("uint32")
    .columns(vec!["sample_A".into(), "sample_B".into()]);
let track = store.create_track("depths", config)?;

// Write a chunk
let data = Array2::<u32>::zeros((1_000_000, 2));
track.write_chunk::<u32>("chr1", 0, data)?;

// Read it back
let chunk: Array2<u32> = track.read_chunk("chr1", 0)?;
```

## Features

- **Zarr v3 only** with Blosc/Zstd compression and PackBits for booleans
- **Chunk-level I/O** with `ndarray::Array2<T>`
- **Region parsing**: `"chr1:1000-2000"` with comma-stripped numbers
- **Escape hatches** to raw `zarrs::Array` objects
- **All standard dtypes**: u8, u16, u32, i8, i16, i32, f32, f64, bool

## Links

- [PBZ Format Specification](https://github.com/pbzarr/pbzarr-spec)
- [Python Implementation](https://github.com/pbzarr/pbzarr-py)
- [API Documentation](https://docs.rs/pbzarr)
