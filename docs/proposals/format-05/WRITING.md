# Rust imports for the PBZ core draft

Status: implemented additive Rust import path, 2026-10-01. This page concerns
`0.5`; format 0.4 remains the current default. The
[core convention](README.md) owns the schema.

## Objective

Make existing Rust imports produce the new root-Dataset layout. Reuse
`TrackConfig`, the `Import` builder, existing source readers, chunk scheduling,
sharding, codecs, and Zarr writes. There is no separate public writer object
or general-purpose Rust array-authoring framework in this milestone.

The additive Dataset reader already handles the convention's broader shapes
and primitive widths. Native Xarray/Zarr can author those arrays. A valid
arbitrary-rank, binned, scalar, or 64-bit variable does not need a matching
source importer to be valid or readable. Initial Rust imports keep their
current source dtypes and 1D/2D output.

## Public API

Create the Dataset, then use the existing import builder and routing methods:

```rust
let mut ds = Dataset::create(path, reference)?;
Import::from_readers(readers)?
    .into_dataset(&mut ds, vec![(
        "depth".into(),
        TrackConfig::new(Dtype::U32)
            .columns(labels).column_dim("sample"),
    )])
    .readers_as_columns()
    .run()?;
```

`Dataset::create_with_storage` accepts an existing Zarr storage handle.
`into_dataset` only configures the operation. `estimate()` is read-only;
a successful `run()` refreshes the supplied Dataset's catalog and caches.
The [executable fixture](../../../crates/pbzarr/examples/write_dataset.rs)
shows creation, sharding, explicit codecs, metadata notes, and later additions.

## Required format changes

Create the root convention attributes and shared reference arrays. Store
measurements directly at `/<track>` instead of `/<track>/values`.
Use native dimension names and shared root dimension coordinates. Preserve
user descriptions and notes as ordinary array attributes.

Reference order and lengths are fixed. Imports using an existing dimension
must match its labels exactly in order; do not union, reorder, or append.
Explicit singleton column dimensions remain present. `column_dim` remains
an import convenience naming an ordinary dimension, not stored role metadata.

The initial path creates string column coordinates from existing import
labels. It can reuse supported string coordinates with the same decoded
labels. Unsupported coordinate encodings fail explicitly before mutation;
do not cast numeric coordinates to strings to make an import fit.

Preflight the whole import before changing the destination: ordered source
references, track names, dimensions, label identity, dtype, fill, codecs,
and engine layout constraints. Reject occupied keys/prefixes, including
metadata-free leftovers. Do not read unrelated measurement chunks.

## Publication and failure

The existing multi-target engine interleaves writes and has one population
completion result. Populate all targets, then publish array metadata in
request order. A population failure publishes no tracks from that run.
A later publication failure retains the tracks already confirmed complete.
Tracks completed by earlier calls remain unchanged.

Complete shared dimension coordinates before publishing measurements.
Retain completed coordinates if a later import fails; they remain valid
Dataset coordinates. Missing all-fill chunks are legitimate completed output.

Before publishing additions, remove stale consolidated metadata while
preserving other root fields. Leave output unconsolidated for this milestone
so native Xarray can discover every completed array without a final
batch-wide consolidation step. Create a new store's reference arrays before
publishing its root PBZ metadata; creation requires an empty namespace.

Task 3's failure-hardening work is skipped at the user's request. Errors propagate
normally; this milestone does not implement cleanup, uncertain-write readback,
root restoration, structured partial-completion reporting, or fault-injection
tests. Failed writes may leave incomplete output. Reopen after an import error.
The broader failure policy remains design intent, not an implemented guarantee.

Concurrent writers/readers, append, replacement, pyramids, automatic provenance,
new source dtypes, and arbitrary-rank Rust authoring are outside this step.
The [implementation plan](../../STORE_SPEC.md)
covers these minimal changes and their checks.
