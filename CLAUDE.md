# pbzarr-rs — Rust Implementation Plan

## What This Is

`pbzarr` is a Rust library crate implementing the PBZ (Per-Base Zarr) format — a Zarr v3 convention for storing per-base resolution genomic data (depths, methylation, masks, etc.). This is the Rust counterpart to the Python `pbzarr` library.

PBZ is a **convention and domain layer** on top of Zarr v3. This crate handles store layout, metadata, region parsing, and chunk I/O. It delegates all actual array storage and compression to the `zarrs` crate.

A separate CLI tool (`pbz`) will sit on top of this library for ingestion, inspection, and reshaping. That is NOT part of this crate.

## Reference Material

- **PBZ Spec**: The spec lives in the `pbzarr-spec` repo (`../pbzarr-spec/SPEC.md`). This implementation targets **spec version 0.1**. The canonical metadata keys are `"perbase_zarr"` / `"perbase_zarr_track"` — this crate already uses the correct names.
- **Proof of concept**: The `clam` project at ~/dev/clam has a working zarr implementation in `src/core/zarr.rs` using `zarrs` + `ndarray`. Study it for patterns but note the architectural differences below.

## Architecture: Differences from clam

clam's `ChromosomeArrays<T>` conflates "store" and "track" into one struct. PBZ separates them:

- **`PbzStore`** — wraps the zarr root group, owns contig registry, provides track access
- **`Track`** — wraps a zarr group under `/tracks/{name}`, owns its own dtype/columns/chunk config

Other differences from clam:
- Root attribute key is `"perbase_zarr"` (not `"clam_metadata"`)
- Contigs stored as zarr arrays (`/contigs`, `/contig_lengths`), not JSON in attributes
- Track metadata uses `"perbase_zarr_track"` attribute on the track group
- Column names stored as a zarr string array in each track
- No population/callable-loci fields — those are clam-domain concepts
- `Track` is NOT generic over element type — uses runtime dtype with typed read methods (`read_chunk::<u32>(...)`)
- Error handling via `thiserror` (not `color-eyre`)

## What carries over from clam

- `zarrs` crate + `FilesystemStore` (sync I/O, no async)
- `ndarray::Array2<T>` for chunk I/O
- Blosc(Zstd, level 5, byte shuffle) compression for numeric types
- PackBits codec for bool arrays
- Chunk boundary math (`position_to_chunk`, `overlapping_chunks`, `chunk_bounds`)
- `store_chunk_ndarray` / `retrieve_chunk_ndarray` for I/O



No `color-eyre`, no `rayon`, no `clap`. This is a library crate only.

## Project Structure

```
pbzarr-rs/
├── Cargo.toml
├── LICENSE (MIT)
├── src/
│   ├── lib.rs          # Public API exports
│   ├── error.rs        # PbzError enum (thiserror)
│   ├── region.rs       # Region struct, parse_region()
│   ├── store.rs        # PbzStore: create, open, contigs, track access
│   └── track.rs        # Track: metadata, read_chunk, write_chunk, escape hatch
└── tests/
    └── integration.rs  # Round-trip tests
```

## Implementation Phases

Execute these in order. Check in with the user after each phase.

### Phase 1: Scaffold + Error Types + Region Parsing

1. `Cargo.toml` with dependencies listed above. Edition 2024.
2. `src/error.rs` — `PbzError` enum with thiserror:
   - `ContigNotFound { contig: String, available: Vec<String> }`
   - `TrackNotFound { name: String, available: Vec<String> }`
   - `ColumnNotFound { name: String, available: Option<Vec<String>> }`
   - `InvalidRegion { message: String }`
   - `InvalidDtype { dtype: String }`
   - `Store(String)` — wraps zarrs errors as string (zarrs errors don't impl std Error consistently)
   - `Metadata(String)` — missing/malformed pbz metadata
   - Type alias: `pub type Result<T> = std::result::Result<T, PbzError>;`
3. `src/region.rs`:
   - `pub struct Region { pub contig: String, pub start: Option<u64>, pub end: Option<u64> }`
   - `pub fn parse_region(region: &str) -> Result<Region>` — handles `chr1`, `chr1:1000-2000`, `chr1:1000`, comma stripping in numbers, whitespace trimming
   - No `one_based` support (Rust callers handle their own coordinate conversion)
4. `src/lib.rs` — declare modules, re-export public types
5. Unit tests in each module.

### Phase 2: PbzStore — Create and Open

`src/store.rs`:

```rust
pub struct PbzStore {
    store: ReadableWritableListableStorage,
    path: PathBuf,
    contigs: Vec<String>,                  // cached from /contigs array
    contig_lengths: HashMap<String, u64>,   // cached from /contig_lengths array
}
```

**`PbzStore::create(path, contigs, contig_lengths)`:**
- Validates contigs.len() == contig_lengths.len(), non-empty
- Creates `FilesystemStore`, root group with `{"perbase_zarr": {"version": "0.1"}}` attribute
- Creates `/contigs` array — zarrs `DataType::String`, shape `(num_contigs,)`
- Creates `/contig_lengths` array — `DataType::Int64`, shape `(num_contigs,)`
- Creates empty `/tracks` group

**`PbzStore::open(path)`:**
- Opens root group, validates `pbz` attribute with `version` field
- Reads and caches `/contigs` and `/contig_lengths` arrays

**Methods:**
- `contigs() -> &[String]`
- `contig_lengths() -> &HashMap<String, u64>`
- `contig_length(name: &str) -> Result<u64>`
- `validate_contig(name: &str) -> Result<()>`
- `tracks() -> Result<Vec<String>>` — walk `/tracks/` for groups with `perbase_zarr_track` attr
- `track(name: &str) -> Result<Track>`
- `create_track(name: &str, config: TrackConfig) -> Result<Track>`

Tests: create + reopen round-trip, metadata validation, contig access, error cases.

### Phase 3: Track — Create and Metadata

`src/track.rs`:

```rust
pub struct TrackMetadata {
    pub dtype: String,
    pub chunk_size: u64,
    pub column_chunk_size: Option<u64>,
    pub has_columns: bool,
    pub description: Option<String>,
    pub source: Option<String>,
}

pub struct TrackConfig {
    pub dtype: String,
    pub columns: Option<Vec<String>>,
    pub chunk_size: u64,          // default 1_000_000
    pub column_chunk_size: u64,   // default 16
    pub description: Option<String>,
    pub source: Option<String>,
}

pub struct Track { ... }
```

**`Track::create()`:**
- Creates group at `/tracks/{name}`, sets `perbase_zarr_track` attribute
- Creates `/tracks/{name}/columns` string array if has_columns
- Creates per-contig data arrays: correct shape, chunks, dtype, compression (Blosc/Zstd/5/shuffle for numeric, PackBits for bool)
- Sets `_ARRAY_DIMENSIONS` on each data array

**Accessors:** `metadata()`, `dtype()`, `columns()`, `has_columns()`, `zarr_array(contig) -> zarrs::Array` (escape hatch)

Tests: create track, metadata round-trip, column array, per-contig shapes, all dtypes.

### Phase 4: Track Read/Write (Chunk-Level)

Port from clam's `read_chunk` / `write_chunk`:

- `read_chunk<T: ElementOwned>(contig: &str, chunk_idx: u64) -> Result<Array2<T>>`
- `write_chunk<T: ElementOwned>(contig: &str, chunk_idx: u64, data: Array2<T>) -> Result<()>`
  - Full chunk: `store_chunk_ndarray`
  - Partial last chunk: `store_chunk_subset_ndarray`
  - Validates column count matches

**Chunk math methods on Track:**
- `position_to_chunk(position: u64) -> u64`
- `chunk_bounds(chunk_idx: u64, contig_length: u64) -> (u64, u64)`
- `overlapping_chunks(start: u64, end: u64) -> Range<u64>`

Tests: write + read round-trip, partial last chunk, multi-contig, bool track, column mismatch error.

### Phase 5: Integration Test

`tests/integration.rs`:
1. Create store with 2 contigs (chr1: 5000bp, chr2: 3000bp)
2. Create u32 columnar track "depths" with 3 columns
3. Write data chunk-by-chunk (chunk_size=1000)
4. Reopen store
5. Read back, verify values match
6. Verify `tracks()`, metadata, columns
7. Test `zarr_array()` escape hatch
8. Create bool scalar track, write/read round-trip

## Coding Conventions

- Rust edition 2021
- `thiserror` for all error types — no `.unwrap()` in library code, no `panic!`
- Comments explain *why*, never *what*. No block-style section dividers.
- Public API gets doc comments (`///`). Internal helpers don't need them.
- Tests use `tempfile::TempDir` for write tests.
- All coordinates are 0-based, half-open.
- `PBZ_VERSION = "0.1"` constant.

## What NOT to Do

- Don't add a CLI — this is a library crate
- Don't add async support — sync only via zarrs FilesystemStore
- Don't add rayon/parallelism — that's the caller's responsibility
- Don't add ingestion logic (D4/GVCF/BED reading) — that belongs in the `pbz` CLI
- Don't make `Track` generic over element type — use runtime dtype with typed read methods
- Don't add population/callable-loci metadata — that's clam-domain, not PBZ-domain
- Don't wrap zarrs types unnecessarily — provide escape hatches instead
- Don't implement region-level reads yet — chunk-level only for v1
- Don't implement sharding yet — can be added to TrackConfig later
- Don't implement column-level slicing — chunk reads return full column width

## What's Deferred

- Region-level reads (`read_region("chr1:1000-2000")`) — adds later on top of chunk I/O
- CLI (`pbz` binary) — separate repo
- Ingestion (D4, GVCF, BED reading) — belongs in `pbz` CLI
- Async I/O
- Sharding — can be added to TrackConfig later
- Column-level slicing — chunk reads return full column width for now
- Cross-language interop tests (Rust ↔ Python)
