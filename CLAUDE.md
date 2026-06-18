# pbzarr-rs

## What This Is

`pbzarr-rs` is the Rust implementation of PBZ — a Zarr v3 convention for storing per-base resolution genomic data (depths, signal, masks). The crate is a thin convention/domain layer on top of Zarr v3; array storage and compression are delegated to `zarrs`. A maturin-built Python wheel (PyO3 + xarray accessor) ships from this repo too.

This repo is a Cargo workspace with all Rust crates under `crates/` (Ruff-style layout): `crates/pbzarr/` (core library, the only crate published to crates.io), `crates/pbzarr-readers/` (input-format readers; owns the `d4` git dependency so the core stays publishable), and `crates/pbzarr-python/` (PyO3 bindings, the `_native` cdylib). Python source lives at top-level `python/`, with the root `pyproject.toml` driving maturin. The `pbz` CLI was deferred from v0 and removed from the workspace.

## Status

Plan 1 of the v0 ship is implemented: library + d4 import + cross-language xarray round-trip. Plans 2–4 (PyO3 Python wheel, benchmarks, docs+release) follow.

An earlier "rank-N rewrite plan" was abandoned in favor of position×column tracks (rank ≤ 2 on disk, unified by `ArrayD<T>` at the I/O surface). See the design doc.

## Design (canonical)

`docs/superpowers/specs/2026-05-25-pbz-v0-ship-design.md` is the authoritative design for v0. When this file conflicts with the design doc, the design doc wins.

## Why It Exists

Three open issues motivate pbzarr:

- [d4-format#82](https://github.com/38/d4-format/issues/82) — accessibility-mask generation from per-sample d4 files is slow with Python loops.
- [d4-format#64](https://github.com/38/d4-format/issues/64) — d4 multi-track files don't compress across samples.
- [clam#25](https://github.com/cademirch/clam/issues/25) — per-position cross-sample sum/mean across samples.

Per-base cohort math is the highest-value use case, but pbz is not cohort-only. It spans a spectrum keyed on a track's column-chunk width: a single-sample store is a first-class artifact competitive with D4/bigWig, many single-sample stores combine into a wide-column cohort array where cross-sample compression and vectorized math pay off, and (v0.2) samples append cheaply onto an existing cohort. The cohort is the high-value end, not the whole point. Read these issues before designing changes that touch import or the public API.

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
- **`Track::read_region<T>(region) -> ArrayD<T>`** and **`write_region<T>(region, ArrayD<T>)`** with runtime dtype check. `write_region` takes ownership of the buffer so zarrs can consume it without a clone. Rank matches the track (1 or 2); callers downcast via `.into_dimensionality::<Ix1>()?` / `Ix2`.
- **`pbzarr::import`** (core, format-agnostic): `Config`, `Report`, `run_pipeline<T, R: ValueReader<Item=T>>`. Format readers and their entry points live in the `pbzarr-readers` crate: **`pbzarr_readers::d4`** exposes `D4Source`, `D4Reader`, `from_d4(&store, track, sources, config)` (int32 tracks); **`pbzarr_readers::bigwig`** exposes `BigWigSource`, `BigWigReader`, `from_bigwig(...)` (float32 tracks). Format-specific entry points use the `from_<format>` naming pattern.
- **`ValueReader::read_into(contig_name, start, end, dst: ArrayViewMut2)`** — name-based, no `ContigId` crossing.
- **`Numeric` trait** has `const DTYPE: Dtype` and `const ZERO: Self`; supertrait bounds include `zarrs::array::Element + ElementOwned`.

The library does not own coordinate-system conversion (callers handle 1-based input formats at the boundary). No `unwrap`/`expect`/`panic!` in library code; all errors flow through `PbzError`.

## Reader/writer architecture

The generic pipeline lives in `pbzarr::import`; format readers live in the separate `pbzarr-readers` crate (not in a CLI; PyO3 binds `from_d4` and `from_bigwig` directly, exposed to Python as `PbzStore.import_d4` / `PbzStore.import_bigwig`).

- **`ValueReader` trait** at `pbzarr::io::reader` (`Send + Sync`). Workers fork their reader per-thread via `ValueReader::fork`.
- **`D4Reader`** at `pbzarr_readers::d4` (int32) and **`BigWigReader`** at `pbzarr_readers::bigwig` (float32) — the two import formats at v0.
- **Parallel-writer pipeline** in `pbzarr::import::pipeline`. `thread::scope` + a single bounded `crossbeam-channel` of tasks. Each worker forks readers, then does read+write itself — no writer thread. Shared `Arc<State>` with `AtomicU64` counters and a `Mutex<Option<PbzError>>` for first error. Zarrs is safe for concurrent writes to non-overlapping shard/chunk files. Tasks are partitioned by the on-disk write unit: one shard per task for sharded tracks (step = `shard_size`), one chunk otherwise. A sub-shard write RMWs the whole shard, so sharded tracks MUST be partitioned shard-aligned — see the zarrs sharding quirk below.
- **`run_pipeline<T, R>` takes `&Track`.** No `&mut`. `Track` caches an `Arc<Array<FilesystemStore>>` per contig (`RwLock<HashMap<...>>`) so per-chunk reads/writes don't re-open `zarr.json`.

## Operational notes

- **Cross-language gate:** `pixi run validate-roundtrip` writes a fixture pbz via `cargo run --example fixture_smoke_store`, then reads it back with `xr.open_datatree(...)`. Failing this is a release blocker.
- **`d4tools` available via pixi `dev` feature.** Used to synthesize fixture d4 files in `tests/import_d4.rs`. Run `pixi run -- cargo test` so it's on PATH; plain `cargo test` self-skips these tests.
- **zarrs 0.23 quirks:**
  - `ArrayBuilder::new(shape, chunk_shape, data_type, fill_value)` — chunk_shape comes before data_type.
  - `Array::open` wants the concrete `Arc<FilesystemStore>`, not the trait-object alias.
  - `retrieve_array_subset_ndarray` / `store_array_subset_ndarray` are deprecated — use `retrieve_array_subset::<ArrayD<T>>` / `store_array_subset(&subset, data)` (owned).
  - `BytesToBytesCodecTraits` is at `zarrs::array::codec::api::` (not `zarrs::array::codec::`).
  - **Sharded arrays: the shard IS the chunk at the `Array` level.** There is no inner-chunk write API. `store_array_subset` only hits the single-encode fast path when the subset exactly equals one shard's full extent; any sub-shard write read-modify-writes the entire shard. Write whole shards at a time.
  - Zarr v3 variable-length strings (`vlen-utf8`) round-trip cleanly across zarrs / zarr-python / xarray.
- **ndarray quirks:**
  - `ndarray::concatenate` is slow — backed by an iterator + `to_vec_mapped` allocator, not a flat memcpy. Cost can dominate when assembling per-reader column buffers. If you need fast assembly, write a manual loop using `Array2::uninit` + `ptr::copy_nonoverlapping` per column.

## Commands

```bash
cargo test -p pbzarr
cargo clippy -p pbzarr -- -D warnings
cargo fmt --all -- --check
pixi run validate-roundtrip
```

CI runs the first three on push/PR to `main` (`.github/workflows/ci.yml`).

## Profiling

- Build with `cargo build --profile profiling --example <name>` — release optimizations + DWARF in a packed `.dSYM` so samply / atos can symbolicate. Profile defined at workspace root.
- Install samply via `pixi global install samply`, then `samply setup` once on macOS to codesign.
- `samply record --save-only -o prof.json -- ./target/profiling/examples/<name> ...` captures; `samply load prof.json` opens Firefox profiler.
- Saved JSON keeps raw addresses. To symbolicate in CLI, use `atos -o <binary>.dSYM/Contents/Resources/DWARF/<...> -arch arm64 -l 0x100000000 <addr + 0x100000000>` (samply addresses are __TEXT-relative; add the base).
- `crates/pbzarr-readers/examples/profile_import.rs` is the throughput harness (synth or `--d4 <path>`). `crates/pbzarr-readers/examples/bench_d4_readers.rs` is a read-only mmap-vs-ssio microbench.

## Project structure

```
pbzarr-rs/                    # workspace root
├── Cargo.toml                # [workspace] members = crates/{pbzarr,pbzarr-readers,pbzarr-python}
├── pyproject.toml            # maturin entry: python-source = python, manifest = crates/pbzarr-python
├── pixi.toml                 # dev (d4tools) + validate (xarray/zarr) + wheel features
├── scripts/
│   └── validate_xarray_read.py   # cross-language round-trip
├── docs/superpowers/specs/   # design docs (uncommitted by default)
├── python/                   # Python package source (Ruff-style top-level dir)
│   ├── pbzarr/               # __init__, _open, _store, _track, _region, accessor, _native.pyi
│   └── tests/
└── crates/
    ├── pbzarr/               # core library (the only crate published to crates.io)
    │   ├── Cargo.toml        # README.md → symlink to repo-root README (for cargo publish)
    │   ├── examples/
    │   │   ├── fixture_smoke_store.rs
    │   │   └── validate_py_written_store.rs
    │   ├── src/
    │   │   ├── lib.rs
    │   │   ├── error.rs
    │   │   ├── genome.rs         # Genome, Contig, ContigId, Region
    │   │   ├── region_query.rs   # RegionQuery + parser
    │   │   ├── store.rs          # PbzStore
    │   │   ├── track.rs          # TrackConfig, TrackMetadata, Track
    │   │   ├── import/
    │   │   │   ├── mod.rs
    │   │   │   └── pipeline.rs   # run_pipeline + Config / Report (format-agnostic)
    │   │   └── io/
    │   │       ├── mod.rs
    │   │       ├── reader.rs     # ValueReader trait
    │   │       ├── dtype.rs      # Dtype + Numeric
    │   │       └── error.rs
    │   └── tests/
    │       ├── store_roundtrip.rs
    │       ├── track_io.rs
    │       └── import_pipeline.rs
    ├── pbzarr-readers/       # input-format readers; owns the d4 git dep (not published)
    │   ├── Cargo.toml
    │   ├── examples/
    │   │   ├── profile_import.rs    # throughput harness for samply
    │   │   └── bench_d4_readers.rs  # mmap vs ssio read-only microbench
    │   ├── src/
    │   │   ├── lib.rs
    │   │   ├── d4/
    │   │   │   ├── mod.rs
    │   │   │   ├── reader.rs        # D4Reader
    │   │   │   └── import.rs        # D4Source + from_d4
    │   │   └── bigwig/
    │   │       ├── mod.rs
    │   │       ├── reader.rs        # BigWigReader (bigtools)
    │   │       └── import.rs        # BigWigSource + from_bigwig
    │   └── tests/
    │       ├── common/mod.rs        # in-process .bw fixture writer (dev-only tokio)
    │       ├── d4_reader.rs
    │       ├── import_d4.rs
    │       ├── bigwig_reader.rs
    │       └── import_bigwig.rs
    └── pbzarr-python/        # PyO3 bindings, the _native cdylib (not published)
        ├── Cargo.toml
        └── src/lib.rs        # import_d4 / import_bigwig → exposed on PbzStore
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
- Don't add `rayon` / parallelism in the library — caller's responsibility. The import pipeline drives this via `crossbeam-channel`.
- Don't make `Track` generic over element type — runtime dtype + typed `read_region::<T>` / `write_region::<T>` is the chosen design.
- Don't wrap `zarrs` types unnecessarily — provide escape hatches.
- Don't promote tracks back to groups (multi-resolution is deferred). Tracks are zarr arrays.
- Don't put per-track metadata in per-group attributes — root `perbase_zarr.tracks` map only.
- Don't cross `ContigId` namespaces between readers and the store. `ValueReader::read_into` takes a contig name + range for exactly this reason.
- **Don't bake cohort framing into public API surface.** pbzarr is a per-base format; cohort analysis is one use case, not the only one (stranded signal, methylation contexts, mask categories — all are valid column-axis interpretations). Parameter, type, and dimension names exposed to users must be generic: use `column` / `column_dim` / `column_label`, never `sample` / `samples` / `n_samples`. The dim's *value* may be `"sample"` for a cohort track (set via `column_dim = "sample"`), but the *parameter name* in any function selecting on the column axis stays generic, and selection must resolve against the track's declared `column_dim` from metadata rather than hardcoding `"sample"` in `dims`.

In the format:

- Don't use `"position"` as a non-position dim name. Reserved.
- Don't add a `tracks/` subgroup at the root. Contig-major is the layout.

## Known limitations

- **Coord-array consistency across contigs is NOT validated on open.** Writers ensure consistency; readers trust it. A `pbzarr::validate` helper is deferred.
- **d4 concurrent reads:** within-file parallelism in `import_d4` depends on the `d4` crate supporting concurrent `read_range`. Not yet verified at scale; works correctly today against a `Mutex` inside `D4Reader`.
- **d4 import is restricted to `int32` tracks** (d4's actual native dtype; earlier code forced u32 and paid a per-position `try_from`). Widening or narrowing during import is not supported in v0.
- **d4 dep is pinned** to `cademirch/d4-format@f836299` with feature `local_reader` (mmap-backed; pulls in `mapped_io`, no htslib). `D4Reader` uses `D4TrackReader::split` + `to_codec().decode_block(...)`. This rev fixes two earlier bugs: `SparseArrayReader::split` now clips records for partitions `[0..K]` fully inside one record, and `bit_array.rs` `decode_block` now uses `read_unaligned` so debug-mode pointer-alignment checks don't trip. `crates/pbzarr-readers/examples/bench_d4_readers.rs` keeps the mmap-vs-ssio microbench around for perf regression checks. Bumping the rev requires verifying the `split` + `decode_block` APIs haven't shifted.
- **bigWig import is restricted to `float32` tracks** (bigWig's native value type). `BigWigReader` reads each region with `BigWigRead::values`, which returns a per-base `Vec<f32>` with `NaN` for uncovered positions. Those `NaN`s are copied through verbatim and match the f32 track's default `NaN` fill value, so all-gap chunks compare equal to the fill and zarrs elides them (`store_empty_chunks = false`). A track created with a non-`NaN` `fill_value` still imports correctly but stops eliding gap chunks.
- **bigtools is a crates.io dep, not git** — `bigtools = { default-features = false, features = ["read"] }` (drops the `remote` HTTP, `cli`, and `write` paths). bigtools hard-deps `tokio`, so it compiles into the graph regardless, but the read path is fully synchronous and `BigWigReader` never touches a runtime. The bigWig *writer* (used only to synthesize test fixtures, in `tests/common`) does need a tokio runtime, so it lives in `[dev-dependencies]` (`bigtools` with `write` + `tokio` rt, current-thread + `channel_size = 0`). Python import tests synthesize `.bw` fixtures with `pybigtools` (PyPI dep in the `wheel` pixi feature).

## Deferred

- PyO3 Python wheel (Plan 2): `create_store` + `create_track` in pure Python (zarr-python); `import_d4` via PyO3.
- xarray accessor for read-side ergonomics (Plan 2).
- Benchmarks: cohort compression, mask generation, cross-sample math, read throughput, sharding sweep (Plan 3).
- Docs + release: README, FORMAT.md, CHANGELOG, tag v0.1.0 (Plan 4).
- `pbz` CLI binary (post-v0).
- Additional input formats: bedGraph, BED, BAM/CRAM, VCF (d4 and bigWig are implemented).
- Sharding defaults (currently off; benchmark sweep will inform the default).
- Multi-resolution tracks (would require track-as-group, breaking change).
- Append samples / append contigs to an existing store. The v0.2 approach stages new columns as a size-1 tail on a rectilinear column grid, compacted into wide chunks later, so appends never rewrite existing chunks (RMW-free, remote-native). Prototyped on the **`recti-append-poc`** branch (isolated `recti` pixi env with tasks `recti-roundtrip` and `recti-lifecycle`), which proves the append/compact algorithm and its on-disk invariants against zarr-python plus the xarray fork. Gated on upstream: zarr-python 3.2+ supports rectilinear grids behind `zarr.config.set({'array.rectilinear_chunks': True})` (set on BOTH read and write; `dtype="str"` yields vlen-utf8); zarrs has it only on `main`/0.24-dev (0.23.x ships the older `rectangular` grid, NOT rectilinear); released xarray can't read it, needs PR #11279 (`maxrjones/xarray@poc/unified-zarr-chunk-grid`).
- Remote stores (S3 / GCS).
- A formal spec document. `pbzarr-spec/SPEC.md` is set aside as draft; `docs/superpowers/specs/2026-05-25-pbz-v0-ship-design.md` is de-facto authority.

## Historical reference

A working zarr+ndarray POC lived in clam (`~/dev/clam/src/core/zarr.rs`). Diverged from current code; don't treat as authoritative. Git history captures the path from the old track-major layout through the abandoned rank-N rewrite plan to the current rank-faithful design with the builder API.
