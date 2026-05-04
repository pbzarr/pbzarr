# pbzarr-rs

## What This Is

`pbzarr` is the Rust implementation of the PBZ (Per-Base Zarr) format — a Zarr v3 convention for storing per-base resolution genomic data (depths, methylation, masks, etc.). Counterpart to the Python `pbzarr` library.

This repo is a Cargo workspace with two members:

1. **The `pbzarr` library crate** (`pbzarr/src/lib.rs`) — store layout, metadata, region parsing, chunk I/O. Delegates array storage and compression to `zarrs`.
2. **The `pbz` CLI binary crate** (`pbz/src/main.rs`) — ingestion, inspection, and metadata editing. v1 implements `import` (D4 → PBZ), `export`/`cat` (bedGraph/BED/TSV), `info`/`list`/`validate` (read-only inspection), and `set`/`rename`/`drop` (metadata edits) plus `completions`. Lives here for now; may split into its own repo once the surface stabilizes.

PBZ is a **convention and domain layer** on top of Zarr v3 — the library doesn't reimplement what `zarrs` already does; it provides escape hatches (`Track::zarr_array(...)`) for callers that need raw access.

## Spec Version

Targets PBZ spec **v0.1**. Spec lives in `../pbzarr-spec/SPEC.md`. Canonical metadata keys (already used by this crate): `"perbase_zarr"` (root), `"perbase_zarr_track"` (per track group). Version constant exported as `pbzarr::PERBASE_ZARR_VERSION`.

## Commands

```bash
cargo test --workspace
cargo clippy --workspace -- -D warnings
cargo fmt --all -- --check
cargo doc --open
```

CI runs all three on push/PR to `main` (`.github/workflows/ci.yml`).

## Project Structure

```
pbzarr-rs/                  # workspace root
├── Cargo.toml              # [workspace] members = ["pbzarr", "pbz"]
├── pbzarr/                 # library crate
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs
│   │   ├── error.rs
│   │   ├── region.rs
│   │   ├── store.rs
│   │   └── track.rs
│   └── tests/
│       └── integration.rs
└── pbz/                    # CLI binary crate
    ├── Cargo.toml
    ├── src/
    │   ├── main.rs
    │   ├── cli.rs                # clap derive structs
    │   ├── error.rs              # color-eyre install
    │   ├── limits.rs             # fd budget pre-flight
    │   ├── pipeline.rs           # crossbeam-channel import pipeline
    │   ├── progress.rs           # indicatif wrapper
    │   ├── commands/             # one module per subcommand
    │   │   ├── import.rs, export.rs, cat.rs, info.rs, list.rs,
    │   │   ├── set.rs, rename.rs, drop_cmd.rs, validate.rs,
    │   │   ├── completions.rs, input_resolve.rs, export_engine.rs
    │   └── io/
    │       ├── value_reader.rs   # ValueReader trait + ValueDtype/ValueChunk
    │       ├── d4_reader.rs      # D4 input
    │       └── tsv_writer.rs, bedgraph_writer.rs, bed_writer.rs
    └── tests/
        ├── fixtures/             # shared StoreFixture builders
        └── *.rs                  # one e2e test binary per subcommand
```

## Library Public API (high-level)

- **`PbzStore`** — `create(path, contigs, lengths)`, `open(path)`, `contigs()`, `contig_length()`, `tracks()`, `track(name)`, `create_track(name, config)`, `rename_track(old, new)`, `drop_track(name)`.
- **`Track`** — `metadata()`, `columns()`, `has_columns()`, `zarr_array(contig)` (escape hatch), `set_description(Option<&str>)`, `set_source(Option<&str>)`, and chunk I/O:
  - 2D (columnar): `read_chunk::<T>(contig, idx)`, `write_chunk::<T>(...)`
  - 1D (scalar tracks): `read_chunk_1d::<T>(...)`, `write_chunk_1d::<T>(...)`
  - Chunk math: `position_to_chunk`, `chunk_bounds`, `overlapping_chunks`
- **`TrackConfig`** — has an `extra: serde_json::Map<String, Value>` field for tool-specific namespaced metadata (use a namespaced key like `"clam"` to avoid collisions).

## CLI Surface (`pbz`)

- `pbz import STORE --track NAME (-i PATH[:COL] ... | -f FILE)` — D4 → PBZ. Channel-based pipeline (crossbeam, sync; no async).
- `pbz export STORE TRACK -o OUT [--format auto|bedgraph|bed|tsv] [--region R] [--column C] [--include-zero]`
- `pbz cat STORE TRACK [--region R] [--format tsv|bedgraph|bed]` — same engine, stdout.
- `pbz info STORE [--json]`, `pbz list STORE (--tracks|--contigs|--columns T) [--json]`, `pbz validate STORE [--json]`.
- `pbz set STORE TRACK [--description S] [--source S]`, `pbz rename STORE OLD NEW`, `pbz drop STORE TRACK [--yes]`.
- `pbz completions {bash|zsh|fish|powershell|elvish}`.

CLI uses `color-eyre` for errors, `tracing`/`tracing-subscriber` for logs (env-filter; `-v`/`-vv`/`-vvv`), `indicatif` for progress (auto-disabled on non-TTY), `rlimit` for fd budget pre-flight.

## Coding Conventions

- Edition **2024**.
- `thiserror` for all error types — no `.unwrap()` / `panic!` in library code.
- Comments explain *why*, never *what*. No block-style section dividers. No SPEC section numbers in code.
- Public API gets `///` doc comments; internal helpers don't need them.
- All coordinates are 0-based, half-open. No 1-based conversion in the library — callers handle their own coordinate systems (the CLI is where 1-based input formats get converted at the boundary).
- Tests use `tempfile::TempDir` for write paths.
- **Adding a new attribute to `perbase_zarr_track`?** Update the `STANDARD_FIELDS` array in `pbzarr/src/track.rs` so `Track::open` doesn't shove the new key into `extra` (round-trip would silently double-store it). Mirror the on-disk default behavior used by `column_dim_name`: omit the key from JSON when the user passes `None`; reader falls back to a default.
- **`zarrs` `DimensionName` is `Option<String>`**, not a wrapper struct. Extract with `.clone().unwrap_or_default()` or `.expect("dim name set")` in tests.
- **Cohort tracks** should set `TrackConfig::column_dim_name = Some("sample".into())`, gated on `has_columns` (don't set it on 1D tracks). Generic tracks leave it `None` (default `"column"`).

## Lint Caveat

The `feat/pbz-cli-v1` branch carries pre-existing clippy errors in `pbz/src/io/bed_error.rs` and `pbz/src/io/bed_reader.rs` (unused variant, unused methods, collapsible `if`). They are unrelated to any current task; do not "fix" them as part of unrelated work.

## Historical Reference

A working zarr+ndarray implementation pattern lived in clam (`~/dev/clam/src/core/zarr.rs`). It was the original POC; the current code has diverged considerably (separate Store/Track, runtime dtype, `perbase_zarr*` keys instead of `clam_metadata`, `thiserror` instead of `color-eyre`). Don't treat clam as authoritative — read the current source.

## What NOT to Do

In the library:
- Don't add async — sync only via `zarrs::FilesystemStore`.
- Don't add `rayon`/parallelism — caller's responsibility.
- Don't put ingestion logic (D4/GVCF/BED reading) in the library — that belongs in the `pbz` CLI binary.
- Don't make `Track` generic over element type — runtime dtype + typed `read_chunk::<T>` is the chosen design.
- Don't add population/callable-loci metadata — that's clam-domain, use `TrackConfig::extra` with a namespaced key instead.
- Don't wrap `zarrs` types unnecessarily — provide escape hatches.

In the CLI:
- Don't introduce `tokio`/async. The pipeline is sync + `crossbeam-channel` by design; for high-latency stores raise worker counts, don't switch runtimes.
- Don't add new input formats inside the library — extend `ValueReader` impls under `pbz/src/io/` and dispatch via the `InputFormat` enum in `commands/input_resolve.rs`.

## Deferred

- Region-level reads (`read_region("chr1:1000-2000")`) — would build on chunk I/O.
- Sharding — can be added to `TrackConfig` later. Pipeline implications are noted in the design spec.
- Column-level slicing — chunk reads currently return full column width.
- Cross-language interop tests (Rust ↔ Python).
- Splitting the `pbz` CLI into its own repo (intentional for now; revisit when the CLI surface stabilizes).
- Additional CLI input formats: bigWig, bedGraph (interval), BED, BAM/CRAM, GVCF/VCF.
- Provenance metadata in tracks (full command line + timestamps + resolved input list under a `pbz.provenance` namespaced `extra` key).
- `pbz stats`, `pbz merge`.
- Auto-raising `RLIMIT_NOFILE` (currently we just check + error with a clear fix).
- Loud-fail in CI when `d4tools` is missing (today the import e2e tests skip silently).
- S3 / remote stores (sync model designed to scale via worker counts; no architectural rewrite required).
