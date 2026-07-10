# pbzarr-rs

## What This Is

`pbzarr-rs` is the Rust implementation of PBZ — a Zarr v3 convention for storing per-base resolution genomic data (depths, signal, masks). The crate is a thin convention/domain layer on top of Zarr v3; array storage and compression are delegated to `zarrs`. A maturin-built Python wheel (PyO3 + xarray accessor) ships from this repo too.

This repo is a Cargo workspace with all Rust crates under `crates/` (Ruff-style layout): `crates/pbzarr/` (core library, the only crate published to crates.io), `crates/pbzarr-readers/` (input-format readers; owns the `d4` git dependency so the core stays publishable), and `crates/pbzarr-python/` (PyO3 bindings, the `_native` cdylib). Python source lives at top-level `python/`, with the root `pyproject.toml` driving maturin. The `pbz` CLI was deferred from v0 and removed from the workspace.

## Status

The flat self-describing track layout (pbz v0.4) is implemented for the Rust core, on branch `feat/flat-layout-rust-core`: `PbzStore`, per-track `Genome`, `Track::{read_region,write_region}`, and the import pipeline all target the flat layout. d4 and bigWig readers (`crates/pbzarr-readers`) and the PyO3/Python side are follow-on work to bring up to date with the rewrite.

An earlier contig-major layout (one array per contig per track, store-level union genome) and, before that, an abandoned "rank-N rewrite plan" both predate the current design. Git history has the path; don't treat either as current.

## Design (canonical)

`docs/adr/` (ADRs 0001–0005) holds the binding decisions for the flat layout; `CONTEXT.md` is the domain glossary (Store, Track, Genome, `genome_checksum`, `offsets`, write-unit, feature, coverage, fill value, position/column axis). Consult both first when touching layout, storage, import, genome checksum, or the read API. Both files are local (gitignored); keep them current as decisions land.

## Why It Exists

Three open issues motivate pbzarr:

- [d4-format#82](https://github.com/38/d4-format/issues/82) — accessibility-mask generation from per-sample d4 files is slow with Python loops.
- [d4-format#64](https://github.com/38/d4-format/issues/64) — d4 multi-track files don't compress across samples.
- [clam#25](https://github.com/cademirch/clam/issues/25) — per-position cross-sample sum/mean across samples.

Per-base cohort math is the highest-value use case, but pbz is not cohort-only. It spans a spectrum keyed on a track's column-chunk width: a single-sample store is a first-class artifact competitive with D4/bigWig, many single-sample stores combine into a wide-column cohort array where cross-sample compression and vectorized math pay off, and (v0.2) samples append cheaply onto an existing cohort. The cohort is the high-value end, not the whole point. Read these issues before designing changes that touch import or the public API.

## On-disk layout

Extension: `.pbz`. A flat, self-describing track layout: the store root is a bare `zarr_conventions` marker group; each track is its own group, sized to `ΣL` (the sum of that track's contig lengths), with no per-contig subdivision.

```
foo.pbz/
├── zarr.json                  # root attrs: zarr_conventions=[{uuid,name:"perbase"}]
├── depth/
│   ├── zarr.json               # perbase: version, genome_checksum, genome_name, ragged_index, ragged_contigs, coordinates
│   ├── values                  # 2D (ΣL, n_columns), dims=[position, sample] — cohort track
│   ├── offsets                 # int64[k+1] prefix-sum ragged index over contigs
│   ├── contigs                 # vlen-utf8[k], contig names in genome order
│   └── sample                  # vlen-utf8[n_columns] column-label coord array (name = column_dim)
└── mask/
    ├── zarr.json
    ├── values                  # 1D (ΣL,), dims=[position] — scalar track
    ├── offsets
    └── contigs
```

- Tracks are zarr groups holding arrays, not bare arrays. Multi-resolution is still deferred; landing it would mean nesting further under the track group, not a layout break.
- Rank-faithful `values`: 1D for scalar tracks, 2D (position × column) for cohort tracks. Cohort `values` is column-chunked (not just row-chunked) so cross-sample compression pays off; sharding is position-only at full column width and currently off (default deferred to a benchmark sweep).
- `offsets` is the prefix-sum flat-start index: `offsets[i]` is contig `i`'s flat start, `offsets[0] == 0`, `offsets[k] == ΣL`. A contig's length is recovered as `offsets[i+1] - offsets[i]`; there is no separate `contig_lengths` array.
- Genome is per-track, not store-level: each track group owns its own `contigs` + `offsets` and its own `perbase:genome_checksum`. The store itself holds no genome and no cross-track contig union; two tracks in one store may cover different genomes.
- No `contig_lengths` array and no store-level `contigs` array — both were contig-major-layout artifacts, replaced by the per-track `offsets`/`contigs` ragged index.
- Compression: Blosc(zstd-5, byte-shuffle) on every `values` array. Not configurable.
- All coordinates are 0-based, half-open.
- The `perbase:` block on a track group's `zarr.json` is written **last**, as the completion marker (ADR 0004): a group without it is either not a pbz track or a crashed/partial import, and store discovery (`PbzStore::open`) skips it.

Track `zarr.json` attributes (fields under `perbase:` are the interpretation block; structural facts like dtype, chunk shape, and dim names live on the `values` array itself and aren't duplicated here):

```json
{
  "zarr_conventions": [{"uuid": "b7e3c1a2-...", "name": "perbase"}],
  "perbase:version": "0.4",
  "perbase:genome_checksum": "md5:...",
  "perbase:genome_name": "hg38",
  "perbase:ragged_index": "offsets",
  "perbase:ragged_contigs": "contigs",
  "perbase:coordinates": "0-based-half-open"
}
```

`genome_checksum` is an md5 over the canonical `"{name}\t{length}\n"` join of the track's `(contig name, length)` pairs, sorted by name in byte order. It is the sole identity used for track mergeability; `genome_name` is decorative only and excluded from the checksum (see `Genome::checksum` / `Genome::checksum_payload` in `crates/pbzarr/src/genome.rs`, and ADR 0001 for why Python reimplements this independently rather than sharing the Rust code).

## Library public API

- **`PbzStore::create(path) -> Self`** / **`open(path) -> Self`** build a `FilesystemStore` and delegate to **`create_with_storage(storage)`** / **`open_with_storage(storage)`**, which take an arbitrary `ReadableWritableListableStorage` trait object (filesystem, memory, or a future async-to-sync remote adapter — see ADR 0005). No `genome` or `coordinate_space` argument at store level: the store holds no genome of its own. Accessors: `track_names()`, `track(name)`, `genome_for(track)`. `create_track(name, genome, config) -> &Track` takes the genome explicitly (the manual-authoring path; ADR 0003 covers the import path, which builds the `Genome` from source-file headers and creates the track itself).
- **`TrackConfig::new(dtype)` builder.** Chain `.columns(vec)` for 2D, `.column_dim("sample")` to override the default `"column"`, `.chunk_size(n)`, `.column_chunk_size(n)`, `.shard_size(n)`, `.shard_column_size(n)`, `.fill_value(v)`, `.description(s)`, `.source(s)`. No public `scalar`/`cohort` constructors.
- **`Genome::new(contigs) -> Result<Self>`**, **`from_fai(path)`** (hand-authoring path), **`offsets() -> Vec<i64>`** (the prefix-sum table, `k+1` entries), **`checksum() -> String`** (`"md5:" + hex`, over `checksum_payload()`), **`checksum_payload() -> String`** (the canonical `"{name}\t{length}\n"` join, sorted by name), **`with_name(name)`** (decorative, excluded from the checksum), plus `contigs()`, `len()`, `id(name)`, `get(id)`, `resolve(&RegionQuery)`. A `Genome` is owned by exactly one track, not the store.
- **`Track::read_region<T>(region) -> ArrayD<T>`** and **`write_region<T>(region, ArrayD<T>)`** with runtime dtype check. `write_region` takes ownership of the buffer so zarrs can consume it without a clone. Rank matches the track (1 or 2); callers downcast via `.into_dimensionality::<Ix1>()?` / `Ix2`. Other accessors: `name()`, `genome()`, `dtype()`, `rank()`, `column_dim()`, `total_len()` (ΣL), `chunk_size()`, `columns_count()`.
- **`pbzarr::import`** (core, format-agnostic): `Config`, `Report`, `run_pipeline<T, R: ValueReader<Item=T>>`. Format readers and their entry points live in the `pbzarr-readers` crate: **`pbzarr_readers::d4`** exposes `D4Source`, `D4Reader`, `from_d4(&store, track_name, sources, config)` (int32 tracks); **`pbzarr_readers::bigwig`** exposes `BigWigSource`, `BigWigReader`, `from_bigwig(...)` (float32 tracks). Per ADR 0003, `from_d4`/`from_bigwig` build the `Genome` from the source headers, take the column count from the file list, derive labels from filenames (overridable per-source), and create the track themselves — they no longer require `create_track` to run first.
- **`ValueReader::read_into(contig_name, start, end, dst: ArrayViewMut2)`** — name-based, no `ContigId` crossing.
- **`Numeric` trait** has `const DTYPE: Dtype` and `const ZERO: Self`; supertrait bounds include `zarrs::array::Element + ElementOwned`.

The library does not own coordinate-system conversion (callers handle 1-based input formats at the boundary). No `unwrap`/`expect`/`panic!` in library code; all errors flow through `PbzError`.

## Reader/writer architecture

The generic pipeline lives in `pbzarr::import`; format readers live in the separate `pbzarr-readers` crate (not in a CLI; PyO3 binds `from_d4` and `from_bigwig` directly, exposed to Python as `PbzStore.import_d4` / `PbzStore.import_bigwig`).

- **`ValueReader` trait** at `pbzarr::io::reader` (`Send + Sync`). Workers fork their reader per-thread via `ValueReader::fork`.
- **`D4Reader`** at `pbzarr_readers::d4` (int32) and **`BigWigReader`** at `pbzarr_readers::bigwig` (float32) — the two import formats.
- **Parallel-writer pipeline** in `pbzarr::import::pipeline`. `thread::scope` + a single bounded `crossbeam-channel` of tasks, one task per contig (a task's region is `[genome.offsets()[i], genome.offsets()[i+1])` on the flat position axis). Each worker forks readers, then does read+write itself — no writer thread. Shared `Arc<State>` with `AtomicU64` counters and a `Mutex<Option<PbzError>>` for first error.
- **Import tasks must partition by physical write-unit (chunk, or shard once sharding lands), not by contig** — this is the target design, not yet fully in place. In the flat layout, contig starts are almost never chunk-size multiples, so two tasks over disjoint contigs can still land in the same physical chunk; a sub-chunk write is a zarrs read-modify-write, and two concurrent RMWs on the same chunk race. Until the task partitioner is redesigned to align to physical chunk (or shard) boundaries, `pipeline.rs`'s `State` carries an interim global `write_lock: Mutex<()>` that serializes the write half of every task, trading away write-side parallelism for correctness. Repartitioning to chunk-aligned tasks is a follow-on plan; don't build on the per-contig task shape as if it were final.
- **`run_pipeline<T, R>` takes `&Track`.** No `&mut`. `Track` caches its `values` array handle once (`RwLock<Option<Arc<Array<dyn ReadableWritableListableStorageTraits>>>>>`, one array now that a track is a single flat array rather than one array per contig) so per-chunk reads/writes don't re-open `zarr.json`. `Genome::offsets()` gives the cumulative-offset table `run_pipeline` uses to place each contig's task on the flat axis.

## Operational notes

- **Cross-language gate:** `pixi run validate-roundtrip` writes a fixture pbz via `cargo run --example fixture_smoke_store`, then reads it back with `xr.open_datatree(...)`. Failing this is a release blocker.
- **`d4tools` available via pixi `dev` feature.** Used to synthesize fixture d4 files in `tests/import_d4.rs`. Run `pixi run -- cargo test` so it's on PATH; plain `cargo test` self-skips these tests.
- **zarrs 0.23 quirks:**
  - `ArrayBuilder::new(shape, chunk_shape, data_type, fill_value)` — chunk_shape comes before data_type.
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
├── docs/adr/                 # ADRs 0001–0005, binding decisions for the flat layout
├── CONTEXT.md                 # domain glossary (Store, Track, Genome, genome_checksum, ...)
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
    │   │   ├── store.rs          # PbzStore (flat, storage-agnostic)
    │   │   ├── track.rs          # TrackConfig, PerbaseTrackAttrs, Track
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
- **Adding a new key to a track's `perbase:` metadata?** Add it as a named field on `PerbaseTrackAttrs` (with `#[serde(rename = "perbase:...")]`, plus `#[serde(default, skip_serializing_if = "Option::is_none")]` if optional). There is no root-level track map to update; each track group carries its own block.

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

- Don't add async. Sync only, over a trait-object `ReadableWritableListableStorage` (filesystem/memory today; remote via a future feature-gated async-to-sync adapter, see ADR 0005). For high-latency stores, raise worker counts in the caller.
- Don't add `rayon` / parallelism in the library — caller's responsibility. The import pipeline drives this via `crossbeam-channel`.
- Don't make `Track` generic over element type — runtime dtype + typed `read_region::<T>` / `write_region::<T>` is the chosen design.
- Don't wrap `zarrs` types unnecessarily — provide escape hatches.
- Don't put per-track metadata in a store-level map. Each track group carries its own `perbase:` block on its own `zarr.json`, written last as the completion marker (ADR 0004).
- Don't cross `ContigId` namespaces between readers and the store. `ValueReader::read_into` takes a contig name + range for exactly this reason.
- **Don't bake cohort framing into public API surface.** pbzarr is a per-base format; cohort analysis is one use case, not the only one (stranded signal, methylation contexts, mask categories — all are valid column-axis interpretations). Parameter, type, and dimension names exposed to users must be generic: use `column` / `column_dim` / `column_label`, never `sample` / `samples` / `n_samples`. The dim's *value* may be `"sample"` for a cohort track (set via `column_dim = "sample"`), but the *parameter name* in any function selecting on the column axis stays generic, and selection must resolve against the track's declared `column_dim` from metadata rather than hardcoding `"sample"` in `dims`.

In the format:

- Don't use `"position"` as a non-position dim name. Reserved.
- Tracks ARE groups (each holding `values` + `offsets` + `contigs` + optional column-label array), not bare arrays. The flat layout is the layout; don't reintroduce per-contig arrays or a store-level union genome.
- Don't move genome ownership to the store. A `Genome` belongs to exactly one track; two tracks in the same store may cover different genomes.

## Known limitations

- **Import tasks are per-contig, not chunk-aligned; a global write lock is the interim stopgap.** See "Reader/writer architecture" above. Correctness holds; write-side parallelism is reduced until the task partitioner repartitions by physical write-unit.
- **d4 concurrent reads:** within-file parallelism in `import_d4` depends on the `d4` crate supporting concurrent `read_range`. Not yet verified at scale; works correctly today against a `Mutex` inside `D4Reader`.
- **d4 import is restricted to `int32` tracks** (d4's actual native dtype; earlier code forced u32 and paid a per-position `try_from`). Widening or narrowing during import is not supported in v0.
- **d4 dep is pinned** to `cademirch/d4-format@f836299` with feature `local_reader` (mmap-backed; pulls in `mapped_io`, no htslib). `D4Reader` uses `D4TrackReader::split` + `to_codec().decode_block(...)`. This rev fixes two earlier bugs: `SparseArrayReader::split` now clips records for partitions `[0..K]` fully inside one record, and `bit_array.rs` `decode_block` now uses `read_unaligned` so debug-mode pointer-alignment checks don't trip. `crates/pbzarr-readers/examples/bench_d4_readers.rs` keeps the mmap-vs-ssio microbench around for perf regression checks. Bumping the rev requires verifying the `split` + `decode_block` APIs haven't shifted.
- **bigWig import is restricted to `float32` tracks** (bigWig's native value type). `BigWigReader` reads each region with `BigWigRead::values`, which returns a per-base `Vec<f32>` with `NaN` for uncovered positions. Those `NaN`s are copied through verbatim and match the f32 track's default `NaN` fill value, so all-gap chunks compare equal to the fill and zarrs elides them (`store_empty_chunks = false`). A track created with a non-`NaN` `fill_value` still imports correctly but stops eliding gap chunks.
- **bigtools is a crates.io dep, not git** — `bigtools = { default-features = false, features = ["read"] }` (drops the `remote` HTTP, `cli`, and `write` paths). bigtools hard-deps `tokio`, so it compiles into the graph regardless, but the read path is fully synchronous and `BigWigReader` never touches a runtime. The bigWig *writer* (used only to synthesize test fixtures, in `tests/common`) does need a tokio runtime, so it lives in `[dev-dependencies]` (`bigtools` with `write` + `tokio` rt, current-thread + `channel_size = 0`). Python import tests synthesize `.bw` fixtures with `pybigtools` (PyPI dep in the `wheel` pixi feature).

## Deferred

- Bringing d4/bigWig readers (`pbzarr-readers`) and the PyO3 Python bindings up to date with the flat layout rewrite.
- Import task repartitioning to physical write-unit (chunk/shard) boundaries, replacing the interim global write lock; see "Reader/writer architecture" above.
- xarray accessor for read-side ergonomics.
- Benchmarks: cohort compression, mask generation, cross-sample math, read throughput, sharding sweep.
- Docs + release: README, FORMAT.md, CHANGELOG, tag.
- `pbz` CLI binary.
- Additional input formats: bedGraph, BED, BAM/CRAM, VCF (d4 and bigWig are implemented, pending the flat-layout update above).
- Sharding defaults (currently off; benchmark sweep will inform the default).
- Multi-resolution tracks (would nest further under the track group; not yet designed).
- Append samples / append contigs to an existing store. The v0.2 approach stages new columns as a size-1 tail on a rectilinear column grid, compacted into wide chunks later, so appends never rewrite existing chunks (RMW-free, remote-native). Prototyped on the **`recti-append-poc`** branch (isolated `recti` pixi env with tasks `recti-roundtrip` and `recti-lifecycle`), which proves the append/compact algorithm and its on-disk invariants against zarr-python plus the xarray fork. Gated on upstream: zarr-python 3.2+ supports rectilinear grids behind `zarr.config.set({'array.rectilinear_chunks': True})` (set on BOTH read and write; `dtype="str"` yields vlen-utf8); zarrs has it only on `main`/0.24-dev (0.23.x ships the older `rectangular` grid, NOT rectilinear); released xarray can't read it, needs PR #11279 (`maxrjones/xarray@poc/unified-zarr-chunk-grid`).
- Cloud writes: `object_store`/`opendal` + `AsyncToSyncStorageAdapter` behind a feature gate (ADR 0005 option A). Remote read over `HTTPStore` is available to wire up now; remote read/write via the trait-object storage handle is otherwise unstarted.
- A formal spec document. `pbzarr-spec/SPEC.md` is set aside as draft; `docs/adr/` plus `CONTEXT.md` are the de-facto authority until one is written.

## Historical reference

A working zarr+ndarray POC lived in clam (`~/dev/clam/src/core/zarr.rs`). Diverged from current code; don't treat as authoritative. Git history captures the path from the old track-major layout, through the abandoned rank-N rewrite plan, to the contig-major rank-faithful design with the builder API, to the current flat self-describing layout (ADRs 0001–0005).
