# Rust writing and import contract for the PBZ core draft

Status: accepted failure policy and proposed milestone-B implementation scope,
updated 2026-10-01. This page concerns `core-draft-1`; format 0.4 and its default
entry points remain unchanged. The [core convention](README.md) owns the schema.

## Supported operations

Create an empty Dataset with its complete reference, or add new variables to
an existing Dataset. The generic writer accepts the convention's scalar,
non-genomic, binned, and multidimensional shapes. It writes physical Boolean,
signed/unsigned 8-, 16-, 32-, and 64-bit integer, and float32/float64 values
without routing them through the legacy import dtype enum.

Reference coordinates and ordinary dimension coordinates are shared. New
coordinates may also be auxiliary coordinates, classified using native Xarray
metadata. Coordinate inputs are materialized for validation in this first Rust
interface; measurement arrays are populated through bounded typed writes.
This is not a requirement for future Python/Dask writers to materialize data.

The import adapter accepts existing `ValueReader` implementations and retains
their current source dtypes and one- or two-dimensional output. It requires a
leading `position` dimension. An explicitly requested second dimension survives
even when its extent is one. No implicit dtype conversion, label reordering,
or reference reconciliation occurs. Generic 64-bit writing does not promise
64-bit source decoding by readers whose current output types exclude it.

No replacement, append, mutation of existing measurement chunks, pyramid
creation, or automatic provenance is part of this milestone. The low-level
writer accepts explicit rectangular positions in a full-sized new variable;
the later Python write boundary owns regional view mapping and warnings.

## Preflight

Validate the whole request before mutating the destination:

* Check the convention/version, ordered reference names and lengths, dimension
  names and extents, decoded labels, and coordinate classification.
* Check every proposed name against existing arrays, groups, bare keys, and
  metadata-free prefixes. Leftovers from another operation are occupied.
* Resolve every requested chunk grid, shard layout, dtype, fill, and codec.
  Explicit layout choices remain authoritative.
* Check every source header and import route before writing coordinates or
  chunks. A common inner position chunk grid remains an import-engine limit.
* Require supplied dimension labels for every non-genomic dimension used by
  a new measurement. Reusing existing labels checks their order and values;
  it does not replace that coordinate.

For this first writer, coordinate reuse also requires matching physical dtype
and supported decoding. This conservative capability boundary avoids an
implicit coordinate cast; it does not make differently encoded stores invalid.
Descriptions and notes on existing coordinates are preserved.

Read the reference and required coordinates, not existing measurement chunks.
An unrelated unsupported measurement encoding does not prevent adding a
compatible supported variable. Unknown semantics on the root or a required
coordinate do prevent writing.

## Completion and failure

Completed tracks remain when a later addition fails. Clean up incomplete
additions where possible. Never remove or overwrite existing tracks.

A successful population callback means all intended writes finished. Unwritten
regions retain the declared storage fill; missing all-fill chunks do not mean
that a track is incomplete. Publish a track by writing its final array metadata
only after population succeeds.

The current multi-target import engine interleaves writes and reports completion
for the whole run. Therefore, populate the whole run first, then publish each
track's metadata in request order. A decoding/chunk-write failure publishes none
of that run's tracks. A later publication failure can leave a completed subset.
Tracks completed by earlier calls remain in either case. This contract does not
require independent completion detection for interleaved import targets.

Complete new coordinate prerequisites before publishing tracks. Retain completed
coordinates after a later failure and include them in the write report.
An auxiliary coordinate is complete only after its native coordinate
classification is persisted. Do not leave a failed auxiliary addition silently
classified as a measurement.

Before adding arrays to an existing store, remove any stale consolidated
metadata while preserving the other root metadata. Leave the result
unconsolidated for this milestone. Fresh native Xarray/Zarr discovery must see
every completed track without a final batch-wide consolidation step.
Readers explicitly requiring consolidated metadata must reconsolidate first.

For new stores, write the reference coordinates before the root PBZ metadata.
Creation requires an empty storage namespace; it never overwrites another store.

## Error reporting and limits

Use a single writer. Before cleanup, stop scheduling and join all outstanding
writes. Cleanup is limited to paths first created by this operation. Remove
incomplete array metadata before its chunks so readers do not discover an array
whose payload is being removed. Retain completed tracks and their coordinates.

Storage can persist bytes and still return an error. After a failed metadata
write, read back the key. Exact intended metadata, with successful prior data
writes, establishes completion. Absent or malformed new metadata permits
cleanup of the owned addition. Failed readback leaves an uncertain outcome:
preserve the affected paths and report them, rather than claiming rollback.

A failed root update requires separate handling. Preserve the previous root
bytes, inspect the result, and attempt restoration if the outcome is partial.
Do not restore a stale consolidated catalog after tracks have completed.
If root restoration fails, report that the store may need repair and identify
`zarr.json`; do not publish further tracks.

Errors retain the original cause and list completed tracks, retained completed
coordinates, and unresolved paths/cleanup failures. Successful cleanup is not
an atomic rollback guarantee. Existing track arrays are never intentionally
modified, but a failed root-metadata overwrite can affect opening the store.

There is no guarantee for concurrent readers/writers, process termination,
power loss, or a storage backend that cannot establish a read-after-write
outcome. No transaction journal or automatic recovery service is introduced.

The [writer implementation plan](../../superpowers/plans/2026-10-01-pbz-core-writer.md)
defines the Rust interfaces and failure-injection tests. These behaviors are
planned, not yet verified implementation results.
