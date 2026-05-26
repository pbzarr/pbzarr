# pbzarr-rs

## What This Is

`pbzarr-rs` is the Rust implementation of PBZ — a Zarr v3 convention for storing per-base resolution genomic data (depths, signal, masks). The crate is a thin convention/domain layer on top of Zarr v3; array storage and compression are delegated to `zarrs`. A Python wheel (PyO3 + xarray accessor) lands once Plan 2 ships; for now the crate ships alone.

This repo is a single-crate workspace: `pbzarr/` (library). The `pbz` CLI was deferred from v0 and removed from the workspace.

## Status

Plan 1 of the v0 ship is implemented: library + d4 ingest + cross-language xarray round-trip. Plans 2–4 (PyO3 Python wheel, benchmarks, docs+release) follow.

An earlier "rank-N rewrite plan" was abandoned in favor of position×column tracks (rank ≤ 2 on disk, unified by `ArrayD<T>` at the I/O surface). See the design doc.

## Design (canonical)

`docs/superpowers/specs/2026-05-25-pbz-v0-ship-design.md` is the authoritative design for v0. When this file conflicts with the design doc, the design doc wins.

## Why It Exists

Three open issues motivate pbzarr:

- [d4-format#82](https://github.com/38/d4-format/issues/82) — accessibility-mask generation from per-sample d4 files is slow with Python loops.
- [d4-format#64](https://github.com/38/d4-format/issues/64) — d4 multi-track files don't compress across samples.
- [clam#25](https://github.com/cademirch/clam/issues/25) — per-position cross-sample sum/mean across samples.

Per-base cohort math is the use case. pbzarr is the cohort-shaped substrate. Read these issues before designing changes that touch ingest or the public API.

## On-disk layout

Extension: `.pbz`. Contig-major Zarr v3 store.

```
foo.pbz/
├── zarr.json                  # root attrs: perbase_zarr.{version, coordinate_space, tracks}
├── contigs                    # 1D string, dim=[contigs]
├── contig_lengths             # 1D int64, dim=[contigs]
├── chr1/
│   ├── sample                 # 1D string coord, dim=[sample] — present if any cohort track uses dim "sample"
│   ├── depth                  # 2D (len, n_samples), dims=[position, sample]
│   └── mask                   # 1D (len,),           dims=[position]
└── chr2/ …
```

- Tracks are zarr arrays (not groups). Multi-resolution is deferred indefinitely; if it lands it's a breaking layout change.
- 1D for scalar tracks, 2D for cohort tracks. Rank-faithful on disk.
- Per-contig coord arrays for the column dim (only when a cohort track uses it). Coord arrays for the same dim name must match across contigs (writer guarantees; validator deferred).
- Compression: Blosc(zstd-5, byte-shuffle) on every data array. Not configurable in v0.
- All coordinates are 0-based, half-open.

Root `perbase_zarr` attribute:

```json
{
  "version": "0.1",
  "coordinate_space": "GRCh38",
  "tracks": {
    "depth": {"dtype": "uint16", "chunk_size": 1000000,
              "column_dim": "sample", "column_chunk_size": 16},
    "mask":  {"dtype": "bool",   "chunk_size": 1000000}
  }
}
```

## Library public API

- **`PbzStore::create(path, genome, coordinate_space) -> Self`**, **`open(path) -> Self`**, accessors `genome()` / `coordinate_space()` / `track_names()` / `track(name)`, plus `create_track(name, config) -> &Track`.
- **`TrackConfig::new(dtype)` builder.** Chain `.columns(vec)` for 2D, `.column_dim("sample")` to override the default `"column"`, `.chunk_size(n)`, `.column_chunk_size(n)`, `.shard_size(n)`, `.fill_value(v)`, `.description(s)`, `.source(s)`. No public `scalar`/`cohort` constructors.
- **`Track::read_region<T>(region) -> ArrayD<T>`** and **`write_region<T>(region, ArrayViewD<T>)`** with runtime dtype check. Rank matches the track (1 or 2); callers downcast via `.into_dimensionality::<Ix1>()?` / `Ix2`.
- **`pbzarr::ingest`:** `D4Source`, `ImportConfig`, `ImportReport`, `run_pipeline<T, R: ValueReader<Item=T>>`, `import_d4(&store, track, sources, config)`.
- **`ValueReader::read_into(contig_name, start, end, dst: ArrayViewMut2)`** — name-based, no `ContigId` crossing.
- **`Numeric` trait** has `const DTYPE: Dtype` and `const ZERO: Self`; supertrait bounds include `zarrs::array::Element + ElementOwned`.

The library does not own coordinate-system conversion (callers handle 1-based input formats at the boundary). No `unwrap`/`expect`/`panic!` in library code; all errors flow through `PbzError`.

## Reader/writer architecture

Ingest lives in `pbzarr::ingest` (not in a CLI; PyO3 will bind `import_d4` directly when Plan 2 lands).

- **`ValueReader` trait** at `pbzarr::io::reader` (`Send + Sync`). Workers fork their reader per-thread via `ValueReader::fork`.
- **`D4Reader`** at `pbzarr::io::d4` — the only ingest format at v0.
- **Channel-based pipeline** in `pbzarr::ingest::pipeline`. `crossbeam-channel`, sync, bounded for backpressure. Within-file parallelism via worker forks; tasks are chunk-aligned.
- **`run_pipeline<T, R>` takes `&Track`.** No `&mut`; opens its own zarr arrays via the `Arc<FilesystemStore>` on `Track`.

## Operational notes

- **Cross-language gate:** `pixi run validate-roundtrip` writes a fixture pbz via `cargo run --example fixture_smoke_store`, then reads it back with `xr.open_datatree(...)`. Failing this is a release blocker.
- **`d4tools` available via pixi `dev` feature.** Used to synthesize fixture d4 files in `tests/import_d4.rs`. Tests skip silently if it's missing.
- **zarrs 0.23 quirks:**
  - `ArrayBuilder::new(shape, chunk_shape, data_type, fill_value)` — chunk_shape comes before data_type.
  - `Array::open` wants the concrete `Arc<FilesystemStore>`, not the trait-object alias.
  - `retrieve_array_subset_ndarray` / `store_array_subset_ndarray` are deprecated — use `retrieve_array_subset::<ArrayD<T>>` / `store_array_subset(&subset, data.to_owned())`.
  - `BytesToBytesCodecTraits` is at `zarrs::array::codec::api::` (not `zarrs::array::codec::`).
  - Zarr v3 variable-length strings (`vlen-utf8`) round-trip cleanly across zarrs / zarr-python / xarray.

## Commands

```bash
cargo test -p pbzarr
cargo clippy -p pbzarr -- -D warnings
cargo fmt --all -- --check
pixi run validate-roundtrip
```

CI runs the first three on push/PR to `main` (`.github/workflows/ci.yml`).

## Project structure

```
pbzarr-rs/                    # workspace root (single crate)
├── Cargo.toml                # [workspace] members = ["pbzarr"]
├── pixi.toml                 # dev (d4tools) + validate (xarray/zarr) features
├── scripts/
│   └── validate_xarray_read.py   # cross-language round-trip
├── docs/superpowers/specs/   # design docs (uncommitted by default)
└── pbzarr/
    ├── Cargo.toml
    ├── examples/
    │   └── fixture_smoke_store.rs
    ├── src/
    │   ├── lib.rs
    │   ├── error.rs
    │   ├── genome.rs         # Genome, Contig, ContigId, Region
    │   ├── region_query.rs   # RegionQuery + parser
    │   ├── store.rs          # PbzStore
    │   ├── track.rs          # TrackConfig, TrackMetadata, Track
    │   ├── ingest/
    │   │   ├── mod.rs
    │   │   ├── pipeline.rs   # run_pipeline + ImportConfig / ImportReport
    │   │   └── d4_import.rs  # D4Source + import_d4
    │   └── io/
    │       ├── mod.rs
    │       ├── reader.rs     # ValueReader trait
    │       ├── d4.rs         # D4Reader
    │       ├── dtype.rs      # Dtype + Numeric
    │       └── error.rs
    └── tests/
        ├── store_roundtrip.rs
        ├── track_io.rs
        ├── import_pipeline.rs
        ├── import_d4.rs
        └── d4_reader.rs
```

## Coding conventions

- Edition **2024**.
- `thiserror` for library errors; no `unwrap`/`expect`/`panic!` in library code.
- Comments explain *why*, never *what*. No block-style section dividers.
- Public API gets `///` doc comments; internal helpers don't need them.
- All coordinates are 0-based, half-open; the library does not convert 1-based formats.
- Tests use `tempfile::TempDir` for write paths.
- POC test discipline: cover round-trips, dtype/rank, and chunk-boundary cases. Skip attribute-pinning and exhaustive edge-case sweeps.
- **Adding a new key to root `perbase_zarr.tracks[name]` metadata?** Add it as a named field on `TrackMetadata` (with `#[serde(default, skip_serializing_if = "Option::is_none")]` if optional). Otherwise it lands in `extra` and round-trip silently double-stores it.

## Commit message format

Conventional Commits for every commit and PR title. Enforced on PR titles via `amannn/action-semantic-pull-request@v5`; `release-please` consumes the format.

- Type prefix: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`, `build`, `ci`, `style`, `revert`.
- Optional scope `pbzarr` for library-localized changes; omit for cross-cutting. Don't use `rs:` or `cli:`.
- Subject in lowercase, imperative mood, no trailing period.
- Subject-only by default; body when there's a real *why* to capture.
- Breaking: append `!` after type/scope and add a `BREAKING CHANGE:` footer.

Examples:

- `feat(pbzarr): Track read_region/write_region returning ArrayD`
- `refactor(pbzarr): ValueReader::read_into takes contig name + range`
- `chore(pbzarr): drop pbz crate; strip store/track for v0 rewrite`

## What NOT to do

In the library:

- Don't add async. Sync only via `zarrs::FilesystemStore`. For high-latency stores, raise worker counts in the caller.
- Don't add `rayon` / parallelism in the library — caller's responsibility. The ingest pipeline drives this via `crossbeam-channel`.
- Don't make `Track` generic over element type — runtime dtype + typed `read_region::<T>` / `write_region::<T>` is the chosen design.
- Don't wrap `zarrs` types unnecessarily — provide escape hatches.
- Don't promote tracks back to groups (multi-resolution is deferred). Tracks are zarr arrays.
- Don't put per-track metadata in per-group attributes — root `perbase_zarr.tracks` map only.
- Don't cross `ContigId` namespaces between readers and the store. `ValueReader::read_into` takes a contig name + range for exactly this reason.

In the format:

- Don't use `"position"` as a non-position dim name. Reserved.
- Don't add a `tracks/` subgroup at the root. Contig-major is the layout.

## Known limitations

- **Coord-array consistency across contigs is NOT validated on open.** Writers ensure consistency; readers trust it. A `pbzarr::validate` helper is deferred.
- **d4 concurrent reads:** within-file parallelism in `import_d4` depends on the `d4` crate supporting concurrent `read_range`. Not yet verified at scale; works correctly today against a `Mutex` inside `D4Reader`.
- **d4 ingest is currently restricted to `uint32` tracks** (d4's native dtype). Widening or narrowing during ingest is not supported in v0.

## Deferred

- PyO3 Python wheel (Plan 2): `create_store` + `create_track` in pure Python (zarr-python); `import_d4` via PyO3.
- xarray accessor for read-side ergonomics (Plan 2).
- Benchmarks: cohort compression, mask generation, cross-sample math, read throughput, sharding sweep (Plan 3).
- Docs + release: README, FORMAT.md, CHANGELOG, tag v0.1.0 (Plan 4).
- `pbz` CLI binary (post-v0).
- Additional input formats: bigWig, bedGraph, BED, BAM/CRAM, VCF.
- Sharding defaults (currently off; benchmark sweep will inform the default).
- Multi-resolution tracks (would require track-as-group, breaking change).
- Append samples / append contigs to an existing store.
- Remote stores (S3 / GCS).
- A formal spec document. `pbzarr-spec/SPEC.md` is set aside as draft; `docs/superpowers/specs/2026-05-25-pbz-v0-ship-design.md` is de-facto authority.

## Historical reference

A working zarr+ndarray POC lived in clam (`~/dev/clam/src/core/zarr.rs`). Diverged from current code; don't treat as authoritative. Git history captures the path from the old track-major layout through the abandoned rank-N rewrite plan to the current rank-faithful design with the builder API.
