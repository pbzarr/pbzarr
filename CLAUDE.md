# pbzarr-rs

## What This Is

`pbzarr` is the Rust implementation of the PBZ (Per-Base Zarr) format — a Zarr v3 convention for storing per-base resolution genomic data (depths, methylation, masks, etc.). Counterpart to the Python `pbzarr` library.

This repo currently holds two things:

1. **The `pbzarr` library crate** (`src/lib.rs`) — store layout, metadata, region parsing, chunk I/O. Delegates array storage and compression to `zarrs`.
2. **The `pbz` CLI binary** (planned, not yet present) — ingestion, inspection, reshaping. Will live here for now and eventually be split into its own repo. When adding CLI code, put it under `src/bin/pbz/` (or a `[[bin]]` target) so the library/binary boundary stays clean for the future split.

PBZ is a **convention and domain layer** on top of Zarr v3 — the library doesn't reimplement what `zarrs` already does; it provides escape hatches (`Track::zarr_array(...)`) for callers that need raw access.

## Spec Version

Targets PBZ spec **v0.1**. Spec lives in `../pbzarr-spec/SPEC.md`. Canonical metadata keys (already used by this crate): `"perbase_zarr"` (root), `"perbase_zarr_track"` (per track group). Version constant exported as `pbzarr::PERBASE_ZARR_VERSION`.

## Commands

```bash
cargo test                       # all tests (unit + tests/integration.rs)
cargo clippy -- -D warnings      # lint (CI fails on any warning)
cargo fmt -- --check             # formatting check (CI enforces)
cargo doc --open                 # render API docs locally
```

CI runs all three on push/PR to `main` (`.github/workflows/ci.yml`).

## Project Structure

```
src/
├── lib.rs       # Public API re-exports + PERBASE_ZARR_VERSION constant
├── error.rs     # PbzError (thiserror), Result alias
├── region.rs    # Region, parse_region()
├── store.rs     # PbzStore: create/open, contigs, track listing/access
└── track.rs     # Track, TrackConfig, TrackMetadata, chunk I/O
tests/
└── integration.rs  # End-to-end round-trip
```

## Public API Surface (high-level)

- **`PbzStore`** — `create(path, contigs, lengths)`, `open(path)`, `contigs()`, `contig_length()`, `tracks()`, `track(name)`, `create_track(name, config)`.
- **`Track`** — `metadata()`, `columns()`, `has_columns()`, `zarr_array(contig)` (escape hatch), and chunk I/O:
  - 2D (columnar): `read_chunk::<T>(contig, idx)`, `write_chunk::<T>(...)`
  - 1D (scalar tracks): `read_chunk_1d::<T>(...)`, `write_chunk_1d::<T>(...)`
  - Chunk math: `position_to_chunk`, `chunk_bounds`, `overlapping_chunks`
- **`TrackConfig`** — has an `extra: serde_json::Map<String, Value>` field for tool-specific namespaced metadata (use a namespaced key like `"clam"` to avoid collisions).

## Coding Conventions

- Edition **2024**.
- `thiserror` for all error types — no `.unwrap()` / `panic!` in library code.
- Comments explain *why*, never *what*. No block-style section dividers. No SPEC section numbers in code.
- Public API gets `///` doc comments; internal helpers don't need them.
- All coordinates are 0-based, half-open. No 1-based conversion in the library — callers handle their own coordinate systems (the CLI is where 1-based input formats get converted at the boundary).
- Tests use `tempfile::TempDir` for write paths.

## Historical Reference

A working zarr+ndarray implementation pattern lived in clam (`~/dev/clam/src/core/zarr.rs`). It was the original POC; the current code has diverged considerably (separate Store/Track, runtime dtype, `perbase_zarr*` keys instead of `clam_metadata`, `thiserror` instead of `color-eyre`). Don't treat clam as authoritative — read the current source.

## What NOT to Do (in the library)

- Don't add async — sync only via `zarrs::FilesystemStore`.
- Don't add `rayon`/parallelism — caller's responsibility.
- Don't put ingestion logic (D4/GVCF/BED reading) in the library — that belongs in the `pbz` CLI binary.
- Don't make `Track` generic over element type — runtime dtype + typed `read_chunk::<T>` is the chosen design.
- Don't add population/callable-loci metadata — that's clam-domain, use `TrackConfig::extra` with a namespaced key instead.
- Don't wrap `zarrs` types unnecessarily — provide escape hatches.

## Deferred

- Region-level reads (`read_region("chr1:1000-2000")`) — would build on chunk I/O.
- Sharding — can be added to `TrackConfig` later.
- Column-level slicing — chunk reads currently return full column width.
- Cross-language interop tests (Rust ↔ Python).
- Splitting the `pbz` CLI into its own repo (intentional for now; revisit when the CLI surface stabilizes).
