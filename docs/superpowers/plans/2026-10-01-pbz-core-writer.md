# PBZ Core Import Migration Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans with the user's selected inline execution and one fresh final reviewer. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Make existing Rust imports write the new PBZ root-Dataset layout with the smallest necessary changes.

**Architecture:** Add root creation to Dataset and a Dataset target to the existing Import builder. Reuse TrackConfig, source readers, routing, codecs, chunk buffers, shard handling, and Zarr I/O. Keep new-format preparation/publication private.

**Tech Stack:** Existing Rust workspace, zarrs, ndarray, project-local Pixi, and the existing native Xarray conformance harness. No new dependencies.

**Spec:** [Core convention](../../proposals/pbz-core-draft/README.md),
[Rust scope](../../proposals/pbz-core-draft/RUST_CLI.md), and the narrowed
[import/write contract](../../proposals/pbz-core-draft/WRITING.md).

This replaces the broader writer proposal at the user's request on 2026-10-01.
Milestone A is complete. No separate DatasetWriter, WriteRequest, CoordinateInput,
public pending-write context, or generic streaming callback framework is needed.

## Global Constraints

* "The token `core-draft-1` remains provisional."
* "Explicit chunk, shard, and codec choices remain authoritative."
* "The reference is fixed within a store."
* "Conversion from format 0.4 is outside the initial work."
* Keep existing format-0.4 entry points working during this additive step.
* Initial imports retain the nine current io::Dtype variants and 1D/2D outputs.
  The reader's support for broader valid stores remains intact.
* Append, replacement, pyramids, new source types, CLI/Python migration and
  generic Rust array authoring are not prerequisites.
* Use the existing isolated worktree and Pixi environment; preserve unrelated work.

## Review Focus

1. Ordered references or labels differ: reject before destination writes.
2. A prefix contains chunks without metadata: reject without deleting those bytes.
3. Successful imports preserve fill values, codecs, sharding and explicit singleton dimensions.
4. Adding to a consolidated store succeeds: fresh Xarray must discover the new arrays.
5. Existing source readers and legacy import paths retain their numerical and scheduling behavior.

## Public changes

Only root creation and another target for the existing builder are required:

```rust
impl Dataset {
    pub fn create(path: impl AsRef<std::path::Path>,
        reference: Reference) -> crate::Result<Self>;
    pub fn create_with_storage(
        storage: zarrs::storage::ReadableWritableListableStorage,
        reference: Reference) -> crate::Result<Self>;
}
impl<'a, R: crate::io::ValueReader> ImportBuilder<'a, R> {
    pub fn into_dataset(self, dataset: &'a mut Dataset,
        tracks: Vec<(String, TrackConfig)>) -> Self;
}
```

The builder's existing routing, options, estimate, cost-model and tap methods
remain available. into_dataset only configures the operation; estimate is
read-only; run performs preflight, population, and publication. Existing
into_track(s) and run's successful Report remain unchanged.

```rust
let mut ds = Dataset::create(path, reference)?;
Import::from_readers(readers)?
    .into_dataset(&mut ds, vec![(
        "depth".into(),
        TrackConfig::new(dtype).columns(labels).column_dim("sample"),
    )])
    .readers_as_columns()
    .run()?;
```

Use private prepared state and a small destination adapter. Reuse ordinary
PbzError propagation; Task 3's recovery and reporting machinery is skipped.

## Files and ownership

| Files | Necessary change |
| --- | --- |
| crates/pbzarr/src/dataset/write.rs | Root creation, root-array preparation, coordinate compatibility and publication. |
| crates/pbzarr/src/dataset/mod.rs | Register write module and refresh catalog/caches after mutation. |
| crates/pbzarr/src/import/builder.rs | Store either legacy destinations or a Dataset plus existing TrackConfigs. |
| crates/pbzarr/src/import/destination.rs | Private adapter exposing the destination operations the engine already uses. |
| crates/pbzarr/src/import/{routing,engine,mod}.rs | Consume/register the adapter, preserving scheduling and buffers. |
| crates/pbzarr/src/store.rs or codec_spec.rs | Expose only existing dtype/fill/codec helpers needed for reuse. |
| crates/pbzarr/tests/core_import.rs | New-layout creation, import, preflight and regression checks. |
| crates/pbzarr-readers/tests/core_import.rs | Existing file readers targeting the new layout. |
| crates/pbzarr/examples/write_core_dataset.rs | Small Rust-generated import fixture. |
| scripts/validate_core_dataset.py | Native Xarray check of Rust-imported stores. |

Do not redesign Track, reorganize unrelated modules, or duplicate the engine.
Generic scalars, bins, higher ranks and 64-bit results already have reader
conformance cases; retain them without building new authoring APIs.

## Task 1: Create a root Dataset

**Interfaces:** Produce the two Dataset creation methods above. Reference is
the already-implemented checked type; use canonical contig/contig_length names
for creation while continuing to read spec-valid alternate reference names.

- [x] Add a creation test to core_import.rs.

```rust
#[test]
fn creates_shared_reference() {
    let storage: zarrs::storage::ReadableWritableListableStorage =
        std::sync::Arc::new(zarrs::storage::store::MemoryStore::new());
    let reference = Reference::new(vec![
        Contig { name: "chr1".into(), length: 5 },
        Contig { name: "chr2".into(), length: 7 },
    ]).unwrap();
    let ds = Dataset::create_with_storage(storage, reference.clone()).unwrap();
    assert_eq!(ds.reference().unwrap().contigs(), reference.contigs());
    assert_eq!(ds.variables()["contig_length"].role, Role::Coordinate);
    assert_eq!(ds.validate().unwrap().status, ValidationStatus::Valid);
}
```

- [x] Run `pixi run cargo test -p pbzarr --test core_import`; expect the missing creation methods.
- [x] Use existing Zarr builders to write string contig names and int64 lengths,
  their native dimension metadata, then root convention/reference attributes.
  Write reference data before its metadata and publish the root last. Require
  an empty namespace, use positive chunk extents for empty references, and
  do checked reference arithmetic through Reference.
- [x] Add empty-reference/zero-length and occupied-namespace tests, including
  metadata-free keys. Assert occupied bytes remain unchanged.
- [x] Run the focused test plus core_dataset/core_reference; expect success.
- [x] Commit `feat: create core datasets with shared reference coordinates`.

## Task 2: Target the existing import engine at root arrays

**Interfaces:** Add ImportBuilder::into_dataset and a private legacy/root
destination adapter. The adapter supplies the Track operations currently used
by routing, estimates and writes; legacy calls delegate to Track.

- [x] Reuse the ConstReader test fixture from import_pipeline.rs and add a
  single-source, explicit-singleton test using the public builder.

```rust
Import::from_readers(vec![reader])?
    .into_dataset(&mut ds, vec![(
        "depth".into(),
        TrackConfig::new(Dtype::U32)
            .columns(vec!["sample_1".into()]).column_dim("sample"),
    )])
    .readers_as_columns()
    .expect_column_labels(vec!["sample_1".into()])
    .run()?;
assert_eq!(ds.variables()["depth"].dims, ["position", "sample"]);
assert_eq!(ds.variables()["depth"].shape, [12, 1]);
assert_eq!(ds.read_raw::<u32>("depth", &[0..12, 0..1])?,
           ndarray::ArrayD::from_elem(ndarray::IxDyn(&[12,1]), 7));
```

- [x] Run the focused test; expect the absent Dataset target.
- [x] Build unpublished arrays at /<track> from existing TrackConfigs. Reuse
  dtype/fill/codec resolution and explicit layout settings. Prepare root
  string dimension coordinates; use existing coordinates only after decoding
  and comparing labels exactly. Do not write legacy perbase group metadata.
- [x] Feed existing routing and engine code through the private destination
  adapter. Retain source reads, buffers, shard locks, scheduling and taps.
  Reuse held Array handles before publication; do not reopen unpublished nodes.
  Preserve Track sealing on the legacy path.
- [x] Preflight all names/prefixes, dimensions, labels, ordered source references,
  types, codecs and shared inner position grids before any destination write.
  Keep estimate read-only. Reject unsupported coordinate encodings explicitly.
- [x] Add multi-source, multi-field, unsharded/sharded and all-fill cases.
  Compare with the legacy destination for the same source and layout.
  Add reordered-reference/label, collision, dtype and layout rejection tests
  using an instrumented store to assert zero mutations.
- [x] Run `pixi run cargo test -p pbzarr`; expect all new and legacy tests to pass.
- [x] Commit `feat: target core datasets with the existing import builder`.

## Task 3: Skipped at the user's request

Cleanup, uncertain-write reconciliation, root restoration, fault injection, and
structured partial-completion reporting are excluded from this implementation.
Retain ordinary preflight and successful-import metadata publication from Task 2.
Errors propagate normally; a failed write may leave incomplete output.

## Task 4: Verify the new layout through native Xarray

**Interfaces:** Existing source readers implement ValueReader unchanged.
The example writes a fresh destination using Dataset::create and Import's
new target. Native Xarray must open it directly as one Dataset.

- [x] Add backend tests using existing BigWig/BAM fixtures and reader constructors.
  Exercise one source, shared sample labels and multiple fields without
  duplicating legacy from_* convenience APIs.
- [x] Extend the conformance harness to invoke the Rust example and check
  actual data, coordinate classification, dimensions and storage layout.

```python
with xr.open_zarr(output, chunks=None, consolidated=False) as ds:
    assert ds["depth"].dims == ("position", "sample")
    assert "contig_length" in ds.coords
    assert "sample" in ds.coords
    assert "position" not in ds.coords
    np.testing.assert_array_equal(ds["depth"].values[:, 0], expected_depth)
```

- [x] Run the new backend/harness tests before the fixture implementation;
  expect the missing new-layout output.
- [x] Implement the small example and harness invocation using existing fixture
  generation/Cargo target discovery. Check additions to consolidated input through
  native Xarray default discovery too. Retain all previous reader conformance.
- [x] Run `pixi run validate-core-dataset` and `pixi run check`, then
  `pixi run install-wheel` and `pixi run validate-roundtrip`.
- [x] Record actual results and supported limits in VALIDATION.md/RUST_CLI.md;
  update the proposal example to show the existing builder targeting Dataset.
- [x] Commit `test: verify core imports with native Xarray`.
- [ ] Run one fresh final review, fix material findings with regression tests,
  and rerun affected checks. No merge/push/release is part of this task.

## Self-review

The plan covers required root metadata/coordinates (Task 1), compatible existing
imports and explicit storage choices (Task 2), successful metadata publication (Task 2), and cross-language evidence (Task 4). It adds no Rust authoring
framework. Wider valid shapes remain covered by the completed reader and
native Xarray/Zarr authoring. All five Review Focus cases have owning tests.

## Execution record

Tasks 1 and 2 were completed in cbd323f and b83afcf. Task 3 was explicitly
skipped. Task 4 verifies existing backend readers and native Xarray imports.
The [validation record](../../proposals/pbz-core-draft/VALIDATION.md) records
executed checks and limits.

Implementation rulings:

* Root destinations privately reuse Track with a populated Array handle and
  separately held labels, avoiding duplicated bounded writes. A future Track
  method that reads track-group metadata would require an adapter update.
* The initial success fixture used shard `[8,3]` with subchunk `[4,2]`, which
  native Zarr rejects. The valid fixture was corrected and the non-divisible
  case became a preflight rejection test. This follows native codec validation.
