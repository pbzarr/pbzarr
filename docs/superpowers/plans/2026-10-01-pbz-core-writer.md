# PBZ Core Writer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create core-draft-1 Datasets and add compatible root variables through
typed Rust writes and the existing streaming import engine.

**Architecture:** Add a Dataset writer beside the completed additive reader.
Preflight the whole request, populate unpublished arrays with existing zarrs
handles, then publish metadata after successful population. A private import
destination adapter shares the current engine with format 0.4.

**Tech Stack:** Existing Rust workspace, pinned zarrs 0.23.13, ndarray 0.17,
serde_json, project-local Pixi, Xarray/Zarr/Dask conformance environment.
No new storage engine, transaction dependency, or package installation.

**Spec:** [Core convention](../../proposals/pbz-core-draft/README.md),
[writing contract](../../proposals/pbz-core-draft/WRITING.md),
[Rust/CLI scope](../../proposals/pbz-core-draft/RUST_CLI.md), and
[conformance rules](../../proposals/pbz-core-draft/CONFORMANCE.md).

## Global Constraints

* "The token `core-draft-1` remains provisional."
* "Conversion from format 0.4 is outside the initial work."
* "Explicit chunk, shard, and codec choices remain authoritative."
* "Core stores omit explicit coordinate variables for position and bin_N, so the implicit mapping has a single authority."
* "The reference is fixed within a store."
* "Completed tracks remain when a later addition fails."
* "Clean up incomplete additions where possible."
* "Never remove or overwrite existing tracks."
* "A storage fill alone does not exclude values from a reduction."
* Preserve `PBZ_FORMAT_VERSION = "0.4"`, its public importer entry points, the
  default Python opener, and the CLI during this additive milestone.
* Source files and temporary output stores must be different paths.
* Use the isolated worktree and its project-local Pixi environment. Do not
  modify the dirty primary checkout or install missing tools.
* No append, replacement, pyramids, expression provenance, GPU implementation,
  native index changes, or release-version assignment.
* This is milestone B in the [implementation sequence](2026-09-30-pbz-core-reader.md#overall-implementation-sequence).
  CLI routing is C; Python/Dask persistence is D. This plan does not change those entry points.

## Review Focus

1. A destination contains abandoned chunks but no metadata: reject without writes or cleanup of those bytes (Tasks 1–2).
2. A storage call writes bytes and then errors: distinguish completed, incomplete, and uncertain outcomes; report failed cleanup without deleting completed tracks (Task 3).
3. A consolidated store receives multiple tracks and the second publication fails: native Xarray must still discover the first and its coordinates (Tasks 3 and 6).
4. Labels include integers above 2**53, missing sentinels, or reordered identities: exact preflight must reject mismatches before any destination mutation (Tasks 1–2).
5. Import workers fail while other writes remain active: join before cleanup; preserve explicit singleton dimensions, shard layout, fill elision, and the legacy path (Tasks 4–5).

## Scope and execution

Milestone A is complete at `7bac38e`; do not replay its tasks. Use the user's
selected native execution method: implement inline, then one fresh final
reviewer. This document is ready for review; no milestone-B implementation or
verification is claimed by its checkboxes.

Generic storage writes all agreed primitive widths, including i64/u64, at any
supported rank. The adapter accepts the nine current `io::Dtype` variants.
Do not expand the reader sinks, statistics, scale machinery, or CLI dtype
dispatch just to store a generic 64-bit result.

Existing file reader types already implement `ValueReader`. Expose the new
Dataset import operation over those readers and verify real backend inputs.
Do not duplicate every legacy `from_bam/from_bigwig/from_d4/from_bed_matrix`
convenience function in this milestone. Their public behavior stays intact;
CLI integration can assemble the new operation in milestone C.

The multi-target engine has one population barrier. On population failure,
publish no track from that run. On success, publish metadata in request order,
retaining the confirmed subset if a subsequent publication fails. All-fill
tracks complete normally without stored chunk objects.

## Files and ownership

| Path | Responsibility |
| --- | --- |
| `crates/pbzarr/src/dataset/write.rs` | Public writer/types, request validation, create/open/write orchestration. |
| `crates/pbzarr/src/dataset/publish.rs` | Owned-prefix ledger, metadata publication/readback, root updates, cleanup, write reports. |
| `crates/pbzarr/src/dataset/import.rs` | Dataset import preflight and reuse of the shared engine. |
| `crates/pbzarr/src/dataset/mod.rs` | Re-export the additive writer interface. |
| `crates/pbzarr/src/dataset/metadata.rs`, `read.rs` | Share existing checked geometry, encoding and label helpers within the module; preserve reader behavior. |
| `crates/pbzarr/src/import/destination.rs` | Private legacy/prepared destination adapter. |
| `crates/pbzarr/src/import/{mod,builder,routing,engine}.rs` | Route both destination forms through existing buffers/scheduling. |
| `crates/pbzarr/tests/core_writer.rs` | Creation, exact typed writes, preflight, coordinate semantics. |
| `crates/pbzarr/tests/core_write_failures.rs` | Storage faults, completion, cleanup and consolidation. |
| `crates/pbzarr/tests/core_import.rs` | Synthetic readers, scheduling and legacy equivalence. |
| `crates/pbzarr/tests/core_support/mod.rs` | Small test fixtures and instrumented/faulting MemoryStore adapter. |
| `crates/pbzarr-readers/tests/core_import.rs` | Real file-reader inputs through the additive writer. |
| `crates/pbzarr/examples/write_core_dataset.rs` | Rust-written conformance fixtures, including partial-success case. |
| `scripts/validate_core_dataset.py` | Compare Rust-produced stores against native Xarray, in addition to existing reader checks. |
| `docs/proposals/pbz-core-draft/{WRITING,RUST_CLI,VALIDATION,README}.md` | Contract, supported capability, measured evidence, and discovery links. |

Keep the storage lifecycle out of the engine. Keep legacy `Track` sealing
checks intact. Do not split other modules or tune scheduling in this change.

## Shared interfaces

This interface listing omits private fields; it is not a literal implementation.
The types below are Rust API choices, not new stored PBZ metadata. Re-export
public types from `pbzarr::dataset`. `ArrayMetadataV3` is
`zarrs::metadata::v3::ArrayMetadataV3`; `Storage` below is a local alias for
`zarrs::storage::ReadableWritableListableStorage`.

```rust
pub struct CreateOptions {
    pub names: String,   // default "contig"
    pub lengths: String, // default "contig_length"
    pub attributes: serde_json::Map<String, serde_json::Value>,
}
impl Default for CreateOptions {
    fn default() -> Self {
        Self { names: "contig".into(), lengths: "contig_length".into(),
               attributes: serde_json::Map::new() }
    }
}

pub struct VariableSpec {
    pub name: String,
    pub metadata: ArrayMetadataV3,
}
pub struct CoordinateInput; // Private representation omitted from this interface listing.
impl CoordinateInput {
    pub fn new<T: zarrs::array::Element>(
        spec: VariableSpec, values: ndarray::ArrayD<T>,
    ) -> crate::Result<Self>;
}
pub struct WriteRequest {
    pub reference: Reference,
    pub coordinates: Vec<CoordinateInput>,
    pub variables: Vec<VariableSpec>,
}
#[derive(Debug, Default)]
pub struct WriteReport {
    pub tracks: Vec<String>,
    pub coordinates: Vec<String>,
}
#[derive(Debug)]
pub struct WriteFailure {
    pub cause: crate::PbzError,
    pub completed: WriteReport,
    pub unresolved: Vec<(String, String)>, // storage path, explanation
}
pub type WriteResult<T> = std::result::Result<T, WriteFailure>;

pub struct DatasetWriter; // Private representation omitted from this interface listing.
impl DatasetWriter {
    pub fn create(path: impl AsRef<std::path::Path>,
        reference: Reference, options: CreateOptions) -> WriteResult<Self>;
    pub fn create_with_storage(storage: Storage,
        reference: Reference, options: CreateOptions) -> WriteResult<Self>;
    pub fn open(path: impl AsRef<std::path::Path>) -> crate::Result<Self>;
    pub fn open_with_storage(storage: Storage) -> crate::Result<Self>;
    pub fn write_with<T>(
        &mut self, request: WriteRequest,
        populate: impl FnOnce(&PendingVariables) -> crate::Result<T>,
    ) -> WriteResult<(T, WriteReport)>;
}
pub struct PendingVariables; // Private representation omitted from this interface listing.
impl PendingVariables {
    pub fn get(&self, name: &str) -> crate::Result<&PendingVariable>;
}
pub struct PendingVariable; // Private representation omitted from this interface listing.
impl PendingVariable {
    pub fn write_raw<T: zarrs::array::Element>(
        &self, ranges: &[std::ops::Range<u64>], values: ndarray::ArrayD<T>,
    ) -> crate::Result<()>;
    pub fn write_bool(
        &self, ranges: &[std::ops::Range<u64>], values: ndarray::ArrayD<bool>,
    ) -> crate::Result<()>;
}
pub struct DatasetImportOptions {
    pub tracks_from: Option<crate::import::SourceAxis>,
    pub columns_from: Option<crate::import::SourceAxis>,
    pub expected_labels: Option<Vec<String>>,
    pub pipeline: crate::import::PipelineOptions,
}
impl Default for DatasetImportOptions {
    fn default() -> Self {
        Self { tracks_from: None, columns_from: None, expected_labels: None,
               pipeline: crate::import::PipelineOptions::default() }
    }
}
impl DatasetWriter {
    pub fn import<R: crate::io::ValueReader>(
        &mut self, readers: Vec<R>, request: WriteRequest,
        options: DatasetImportOptions,
    ) -> WriteResult<(crate::import::Report, WriteReport)>;
}
```

Define `Display`/`Error` for `WriteFailure`; the original cause is its source.
Preflight failures have an empty report and no unresolved paths. After a
mutation failure, require reopening the writer before another operation;
never continue from a stale catalog or uncertain root.

`CoordinateInput::new` stores the supplied typed array in a private MemoryStore
using its exact metadata, then decodes it with the reader's coordinate helpers.
It does not touch the destination. This avoids a second runtime enum for every
numeric width. Validate dimensions and uniqueness for dimension coordinates;
ordinary auxiliary values may repeat. Memory use is proportional to supplied
coordinate data, not measurement data.

The request supplies all non-genomic dimension coordinates used by its
measurements, including existing ones, except the fixed reference dimension
whose coordinate is already supplied by the ordered reference. Match ordered decoded values and
physical dtype; preserve existing coordinate metadata. Reference coordinates
come from the writer and request's ordered reference, not duplicate user inputs.
Reject variables whose attributes would reclassify a measurement as a coordinate
or introduce dangling `coordinates` references.

`write_raw` uses the physical dtype. `write_bool` supports physical Boolean
and unmasked Xarray int8+`dtype="bool"`, explicitly converting only the latter.
Unsupported logical encodings fail in preflight. Raw writes must validate
`T::validate_data_type` before chunk I/O. A failed write poisons its pending
variable even if the callback catches the error and returns success.

## Task 1: Create a valid root Dataset and validate coordinate inputs

**Files:** Create `dataset/write.rs`, `dataset/publish.rs`,
`tests/core_writer.rs`, `tests/core_support/mod.rs`; modify `dataset/mod.rs`
and the internal coordinate helpers in `dataset/{metadata,read}.rs`.

**Interfaces:** Produce `CreateOptions`, `VariableSpec`, `CoordinateInput::new`,
`WriteReport`, `WriteFailure`, and `DatasetWriter::{create,create_with_storage,open,open_with_storage}`
with the shared signatures. Opening remains read-only.

- [ ] **Write creation and occupied-storage tests.** Add these shared fixture
  helpers and representative test; use the same module in subsequent tasks.

```rust
pub fn reference() -> Reference {
    Reference::new(vec![
        pbzarr::Contig { name: "chr1".into(), length: 5 },
        pbzarr::Contig { name: "chr2".into(), length: 7 },
    ]).unwrap()
}
pub fn spec(name: &str, dims: &[&str], shape: &[u64], dtype: &str) -> VariableSpec {
    let chunks: Vec<_> = shape.iter().map(|&n| n.max(1)).collect();
    VariableSpec {
        name: name.into(),
        metadata: serde_json::from_value(serde_json::json!({
            "zarr_format": 3, "node_type": "array", "shape": shape,
            "dimension_names": dims, "data_type": dtype,
            "chunk_grid": {"name":"regular","configuration":{"chunk_shape":chunks}},
            "chunk_key_encoding":{"name":"default","configuration":{"separator":"/"}},
            "fill_value": if dtype == "string" { serde_json::json!("") }
                          else if dtype == "bool" { serde_json::json!(false) }
                          else { serde_json::json!(0) },
            "codecs": if dtype == "string" { serde_json::json!([{"name":"vlen-utf8"}]) }
                      else { serde_json::json!([{"name":"bytes","configuration":{"endian":"little"}}]) },
            "attributes": {}
        })).unwrap(),
    }
}
#[test]
fn creates_native_reference_coordinates() {
    let s: Storage = std::sync::Arc::new(zarrs::storage::store::MemoryStore::new());
    DatasetWriter::create_with_storage(s.clone(), reference(), CreateOptions::default()).unwrap();
    let ds = Dataset::open_with_storage(s).unwrap();
    assert_eq!(ds.reference().unwrap().contigs(), reference().contigs());
    assert_eq!(ds.variables()["contig_length"].role, Role::Coordinate);
    assert_eq!(ds.validate().unwrap().status, ValidationStatus::Valid);
}
```

  Add executable cases for alternate reference names, empty reference,
  zero-length contigs, maximum valid i64 length, root attribute preservation,
  reserved-attribute override rejection, invalid direct-child names, and
  `orphan/c/0` in an otherwise rootless store. For occupied stores, snapshot
  every key/value and assert both an empty mutation log and identical bytes.

- [ ] **Run red:** `pixi run cargo test -p pbzarr --test core_writer`.
  Expect missing writer types/methods before implementation.
- [ ] **Implement creation with chunk-first reference arrays and root-last publication.**
  Require an empty namespace, including bare keys. Validate configurable names
  as distinct direct children; the auxiliary length name must be representable
  in Xarray's whitespace-delimited coordinate declaration. Reject user
  `pbz:*`, `zarr_conventions`, and `coordinates` overrides before writes.
  Preserve other user root attributes. Use string names, int64 lengths,
  positive coordinate chunks even for empty arrays, and no missing sentinel.

```rust
let lengths: Vec<i64> = reference.contigs().iter()
    .map(|c| i64::try_from(c.length))
    .collect::<std::result::Result<_, _>>()
    .map_err(|e| PbzError::Metadata(e.to_string()))?;
let mut attrs = options.attributes;
attrs.insert("zarr_conventions".into(), serde_json::json!([{
    "uuid": crate::PERBASE_CONVENTION_UUID, "name": "pbz:"
}]));
attrs.insert("pbz:version".into(), serde_json::json!(FORMAT_VERSION));
attrs.insert("pbz:reference".into(), serde_json::json!({
    "names": options.names, "lengths": options.lengths
}));
attrs.insert("coordinates".into(), serde_json::json!(options.lengths));
```

  Build arrays with `ArrayBuilder::build` (no storage mutation), write their
  data, then array metadata, then root metadata. Track owned prefixes before
  the first attempted write. Do not call legacy `Genome::offsets` for arithmetic.
  Use the publication/cleanup ledger now; Task 3 expands its storage-fault tests.
  Check declared coordinate decoding, missingness and exact labels using the
  current reader helpers. Permit ordinary repeated auxiliary values.
- [ ] **Run green:** `pixi run cargo test -p pbzarr --test core_writer --test core_dataset --test core_reference`.
- [ ] **Commit:** `feat: create core datasets with native reference coordinates`.

## Task 2: Preflight and populate generic new variables

**Files:** Modify `dataset/write.rs`, `dataset/publish.rs`,
`dataset/{metadata,read,mod}.rs`; expand `tests/core_writer.rs`.

**Interfaces:** Produce `WriteRequest`, `PendingVariables::get`,
`PendingVariable::{write_raw,write_bool}`, and `DatasetWriter::write_with`.
Internal `DatasetWriter::prepare(&self, request: WriteRequest) -> crate::Result<PreparedWrite>`
performs all checks and builds handles without destination writes.
`DatasetWriter::populate_and_publish<T>(&mut self, prepared: PreparedWrite,
populate: impl FnOnce(&PendingVariables) -> crate::Result<T>)
-> WriteResult<(T, WriteReport)>` owns mutation. `PreparedWrite` is private
to the dataset module and holds validated coordinates, pending variables,
and the prior root metadata bytes.

- [ ] **Write exact-storage and preflight tests.**

```rust
#[test]
fn preserves_uint64_scalar_without_float_conversion() {
    let s: Storage = std::sync::Arc::new(zarrs::storage::store::MemoryStore::new());
    let mut w = DatasetWriter::create_with_storage(
        s.clone(), reference(), CreateOptions::default()).unwrap();
    let request = WriteRequest {
        reference: reference(), coordinates: vec![],
        variables: vec![spec("total", &[], &[], "uint64")],
    };
    let n = (1_u64 << 53) + 1;
    let (_, report) = w.write_with(request, |pending| {
        pending.get("total")?.write_raw(&[], ndarray::arr0(n).into_dyn())
    }).unwrap();
    assert_eq!(report.tracks, ["total"]);
    let ds = Dataset::open_with_storage(s).unwrap();
    assert_eq!(ds.read_raw::<u64>("total", &[]).unwrap()[ndarray::IxDyn(&[])], n);
}
```

  Add data-bearing cases for `(position,sample,threshold)`, independent
  `bin_4`/`bin_1000`, a non-genomic result, partial rectangles with untouched
  storage fill, zero/false with no missing declaration, and logical bool int8
  encoding. Test i64/u64 extrema and every other agreed primitive dtype.
  Preflight failures must record zero mutations: reordered references/labels,
  label dtype mismatch, missing/duplicate labels (including decoded sentinels),
  missing required coordinates, wrong bin extent, duplicate names, implicit
  genomic coordinates, unsupported codec/grid, and an occupied metadata-free
  destination. Reuse compatible coordinates without rewriting their bytes.
  An unrelated unsupported measurement must not block the request.

- [ ] **Run red:** `pixi run cargo test -p pbzarr --test core_writer`.
  Expect missing write interfaces or failing new behavior assertions.
- [ ] **Implement preflight and bounded writes.** Validate the prospective
  catalog's shapes and coordinate identities before any root update. Build
  arrays with `Array::new_with_metadata`; do not serialize measurement data to
  a staging store or materialize full measurement arrays. Check direct-child
  names with the existing node-name validator plus PBZ namespace rules.
  Check all keys under each new prefix, not only its `zarr.json`.
  Reuse the selected-reader validation rules where possible without requiring
  all unrelated measurement codecs to be supported.

```rust
// Inside PendingVariable::write_raw, after recording errors in its sticky state:
T::validate_data_type(self.array.data_type())
    .map_err(|e| PbzError::InvalidDtype { dtype: e.to_string() })?;
let shape = self.array.shape();
if ranges.len() != shape.len()
    || ranges.iter().zip(shape).any(|(r, &n)| r.start > r.end || r.end > n)
    || values.shape().iter().zip(ranges).any(|(&n, r)| n as u64 != r.end-r.start)
    || values.ndim() != ranges.len()
{
    return Err(PbzError::Metadata("write selection and value shape differ".into()));
}
let subset = zarrs::array::ArraySubset::new_with_ranges(ranges);
self.array.store_array_subset(&subset, values)
    .map_err(|e| PbzError::Store(e.to_string()))
```

  Check addressable element/byte sizes before any temporary conversion.
  An empty rectangle is a checked no-op; a scalar uses no ranges.
  Implement `write_bool` with explicit bool-to-i8 conversion only for the
  supported Xarray encoding. Do not accept a bool buffer as arbitrary int8
  bytes. Reject masked logical Booleans as unsupported.
  The callback receives no public Array handle or metadata mutator; a recorded
  write error prevents publication even if the callback returns Ok.
- [ ] **Run green:** `pixi run cargo test -p pbzarr --test core_writer --test core_dataset`.
- [ ] **Commit:** `feat: preflight and write generic core variables`.

## Task 3: Pin partial completion and uncertain storage failures

**Files:** Modify `dataset/publish.rs`, `dataset/write.rs`; create
`tests/core_write_failures.rs`; expand `tests/core_support/mod.rs`.

**Interfaces:** Consume `PreparedWrite` and `WriteFailure`. Keep publication
helpers internal. The test adapter exposes `FaultStore::new()`,
`storage() -> Storage`, `fail_next_set(key: &str, mode: SetFault)`,
`fail_next_get(key: &str)`, `fail_next_erase(key: &str)`,
`mutation_log() -> Vec<String>`, and `snapshot() -> BTreeMap<String, Vec<u8>>`.
`SetFault` is `Before | After | Partial`; faults fire once and then delegate.

- [ ] **Add fault tests against the public operation.** Implement the adapter
  by delegating the pinned zarrs storage traits to MemoryStore. Set-after
  persists all bytes before Err; partial persists half the bytes before Err.

```rust
#[test]
fn retains_first_track_when_second_metadata_write_fails() {
    let fault = FaultStore::new();
    let s = fault.storage();
    let mut w = DatasetWriter::create_with_storage(
        s.clone(), reference(), CreateOptions::default()).unwrap();
    fault.fail_next_set("second/zarr.json", SetFault::Before);
    let err = w.write_with(WriteRequest {
        reference: reference(), coordinates: vec![],
        variables: vec![spec("first", &[], &[], "uint64"),
                        spec("second", &[], &[], "uint64")],
    }, |pending| {
        pending.get("first")?.write_raw(&[], ndarray::arr0(7_u64).into_dyn())?;
        pending.get("second")?.write_raw(&[], ndarray::arr0(9_u64).into_dyn())
    }).unwrap_err();
    assert_eq!(err.completed.tracks, ["first"]);
    assert!(err.unresolved.is_empty());
    let ds = Dataset::open_with_storage(s).unwrap();
    assert!(ds.variables().contains_key("first"));
    assert!(!ds.variables().contains_key("second"));
}
```

  Repeat with a preexisting consolidated root and retained auxiliary coordinate.
  Check that the resulting root is unconsolidated and preserves unrelated attrs.
  Test chunk failure before/after persistence, metadata full-write-then-error,
  partial metadata, failed readback, failed erase, auxiliary classification
  failure, root invalidation/restoration failure, and reference creation failure.
  Snapshot all preexisting array keys to prove they are unchanged.
  Test an ignored callback write error still blocks publication.
  Failed readback must retain and name uncertain paths; never assert rollback.

- [ ] **Run red:** `pixi run cargo test -p pbzarr --test core_write_failures`.
  Expect assertions exposing unhandled fault outcomes; each injected fault
  must be observed in the adapter log.
- [ ] **Implement the publication sequence and outcome reconciliation.**

```text
Existing store:
  complete preflight
  remove consolidated_metadata from root if present, preserving other fields
  write new coordinate payloads, then coordinate metadata
  add new auxiliary names to root.coordinates; confirm root persistence
  retain the now-completed coordinate prerequisites
  run population callback; require successful return and no sticky write errors
  publish measurement zarr.json documents in request order
  return completed names

Any ordinary failure:
  stop scheduling; await/join all operation-owned writes
  read back the failed metadata key if publication was attempted
  exact intended bytes => record completion, retain, return original error
  confirmed incomplete owned metadata => erase metadata, then owned chunks
  failed readback => retain affected paths and report uncertainty
  cleanup errors => preserve original cause and append unresolved paths
```

  Retain dimension coordinates as soon as complete. Auxiliary coordinates
  become complete after confirmed root classification; if classification
  fails and the old root is restored, remove owned unclassified auxiliaries.
  Do not clean up any coordinate whose published state or root references are
  uncertain. Existing arrays and previously occupied prefixes are never owned.
  For root writes, preserve prior bytes; if the attempted bytes are neither
  confirmed current nor confirmed absent/old, attempt restoration and report
  failure. Abort further publication in all error cases.
  Do not restore a stale consolidated root after completing measurements.
  Do not add consolidation on success.
- [ ] **Run green:** `pixi run cargo test -p pbzarr --test core_write_failures --test core_writer`.
- [ ] **Commit:** `fix: retain completed core writes and report uncertain failures`.

## Task 4: Share the import engine through a private destination adapter

**Files:** Create `import/destination.rs`; modify
`import/{mod,builder,routing,engine}.rs`; create `tests/core_import.rs`.

**Interfaces:** Keep existing public `ImportBuilder::into_track(s)` and
`ImportRouting::resolve`. Internally use:
```rust
pub(crate) enum Destination<'a> {
    Legacy(&'a crate::Track),
    Prepared(PreparedDestination),
}
pub(crate) struct PreparedDestination {
    pub name: String,
    pub genome: std::sync::Arc<crate::Genome>,
    pub dtype: crate::io::Dtype,
    pub array: std::sync::Arc<zarrs::array::Array<
        dyn zarrs::storage::ReadableWritableListableStorageTraits>>,
    pub column_labels: Option<Vec<String>>,
}
```
  Add `ImportRouting::resolve_destinations` with the existing resolver signature
  but replace `tracks: &[&Track]` with `destinations: &[Destination<'_>]`.
  Change the private engine `run_import` destination argument likewise,
  preserving its readers, routing, options, cost-model and tap arguments.

- [ ] **Add an internal adapter test that writes to an unpublished root handle
  and compares it to a legacy track using the existing constant-reader fixture.**
  Assert absence of root-array metadata after engine completion and exact
  values read directly from its in-memory array handle. A focused test in
  `destination.rs` can access private adapter constructors; public integration
  tests in Task 5 test the full lifecycle.

```rust
// Preserve the existing public resolver as a thin compatibility boundary.
let destinations: Vec<_> = tracks.iter()
    .map(|track| Destination::Legacy(*track)).collect();
Self::resolve_destinations(
    schemas, genomes, &destinations, tracks_from, columns_from, expected_labels,
)
```

  Pin the adapter's metadata-free write behavior directly as well:

```rust
#[test]
fn prepared_destination_writes_without_publishing_metadata() {
    let storage: Storage = std::sync::Arc::new(zarrs::storage::store::MemoryStore::new());
    let array = std::sync::Arc::new(zarrs::array::ArrayBuilder::new(
        vec![2], vec![2], zarrs::array::data_type::uint32(), 0_u32,
    ).dimension_names(["position"].into()).build(storage.clone(), "/depth").unwrap());
    let destination = Destination::Prepared(PreparedDestination {
        name: "depth".into(),
        genome: std::sync::Arc::new(crate::Genome::new(vec![crate::Contig {
            name: "chr1".into(), length: 2,
        }]).unwrap()),
        dtype: crate::io::Dtype::U32, array: array.clone(), column_labels: None,
    });
    destination.write_flat(0, 2, ndarray::arr1(&[7_u32, 9]).into_dyn()).unwrap();
    assert!(storage.get(&zarrs::storage::StoreKey::new("depth/zarr.json").unwrap()).unwrap().is_none());
    let got: ndarray::ArrayD<u32> = array.retrieve_array_subset(
        &zarrs::array::ArraySubset::new_with_shape(vec![2])).unwrap();
    assert_eq!(got, ndarray::arr1(&[7_u32, 9]).into_dyn());
}
```

  Exercise unsharded, whole-shard and partial-shard output, column tiling,
  cross-contig spans, empty spans, nonzero fill and out-of-order completion.
  Existing import tests remain intact; compare emitted tap units and source
  read counts between legacy/prepared destinations for identical layouts.
- [ ] **Run red:** `pixi run cargo test -p pbzarr destination`.
  Expect missing adapter support.
- [ ] **Implement the adapter methods needed by existing routing and buffers.**
  Match the existing Track signatures for name/genome/dtype/rank/total_len,
  columns_count/column_labels, values_array/write_unit_shape, and typed
  write_flat/write_flat_columns. Legacy delegates to Track, preserving sealed
  writes and all current checks. Prepared validates rank 1/2, leading position,
  dtype, labels and full-reference extent, then writes the held Array directly.
  No Array::open call is possible before publication.
  Update TargetGeom and TileBuffer's unsharded path to accept Destination;
  leave existing shard buffer/store code and worker scheduling intact.
  Preserve ImportBuilder::estimate using the same resolved destination geometry.
  Prepared Boolean import targets use physical Boolean storage, matching the
  existing Boolean buffers. Reject an int8 target for a Boolean source before
  writes; generic write_bool remains the portable int8 conversion interface.
- [ ] **Run green:** `pixi run cargo test -p pbzarr`.
  Include the unchanged import_pipeline, explicit_codecs and sealed-track tests.
- [ ] **Commit:** `refactor: share import engine across legacy and core destinations`.

## Task 5: Import existing readers into root variables

**Files:** Create `dataset/import.rs`; modify `dataset/{write,mod}.rs`;
expand `tests/core_import.rs`; create
`crates/pbzarr-readers/tests/core_import.rs`.

**Interfaces:** Produce `DatasetImportOptions` and `DatasetWriter::import`
with the shared signatures. Consume `prepare`, `populate_and_publish`,
`Destination::Prepared` and `ImportRouting::resolve_destinations`.
Expose a crate-private accessor from PreparedWrite to its ordered pending
handles/specs; this is not a public storage escape hatch.

- [ ] **Write a singleton import test and a source-error cleanup test.**
  Reuse the existing ConstReader implementation from import_pipeline.rs in the
  new test module. Put the simple Reference/spec fixtures in core_support.

```rust
let labels = CoordinateInput::new(
    spec("sample", &["sample"], &[1], "string"),
    ndarray::arr1(&["sample_1".to_owned()]).into_dyn(),
).unwrap();
let request = WriteRequest {
    reference: reference(), coordinates: vec![labels],
    variables: vec![spec("depth", &["position","sample"], &[12,1], "uint32")],
};
let options = DatasetImportOptions {
    columns_from: Some(pbzarr::import::SourceAxis::Readers),
    expected_labels: Some(vec!["sample_1".into()]),
    ..Default::default()
};
writer.import(vec![reader], request, options).unwrap();
let ds = Dataset::open_with_storage(storage).unwrap();
assert_eq!(ds.variables()["depth"].shape, [12, 1]);
assert_eq!(ds.variables()["depth"].dims, ["position", "sample"]);
```

  Add zero-mutation tests for ordered-reference mismatch despite equal legacy
  checksum, label mismatch, wrong source dtype, rank 3, scalar/bin target,
  unsupported routing, and inconsistent inner position grids.
  Use a multi-field synthetic reader for interleaved targets; failure before
  the common barrier publishes none. Delay one writer while another fails and
  assert no writes occur after cleanup begins. An all-fill source still
  publishes a valid track without payload objects.

  In pbzarr-readers, construct BigWigReader and BamReader from existing generated
  test fixtures and import through DatasetWriter::import. Compare values,
  source field mapping and gaps against existing backend tests. Include
  composition fields and multiple sources where current routing supports them.
  Reuse current fixture builders and reader constructors; no new external data.
- [ ] **Run red:** `pixi run cargo test -p pbzarr --test core_import` and
  `pixi run cargo test -p pbzarr-readers --test core_import`.
  Expect absent Dataset import API or failed lifecycle assertions.
- [ ] **Implement orchestration with source preflight before mutation.**

```text
open/inspect readers
prepare the WriteRequest (read-only destination checks)
construct checked Genome from request.reference's ordered contigs
construct prepared engine destinations from held handles
resolve routing against every reader schema, ordered reference and labels
validate shared inner position grid and all engine allocation/layout limits
populate_and_publish(prepared, callback that runs the shared engine)
return engine Report plus WriteReport
```

  Engine setup checks currently performed inside run_import must run before
  coordinate publication when they can reject headers/layout without I/O.
  Share those checks between estimate/preflight/run instead of duplicating
  inconsistent validation. A storage/read/decode failure during execution uses
  the publication ledger; do not hide it in an otherwise successful Report.
  Derive reference from validated source headers explicitly before creating a
  new Dataset. A successful empty Dataset may remain if a later import fails.
- [ ] **Run green:** the two red commands, then `pixi run cargo test -p pbzarr --test import_pipeline`.
- [ ] **Commit:** `feat: import existing value readers into core datasets`.

## Task 6: Verify native Xarray interoperability and document the boundary

**Files:** Create `examples/write_core_dataset.rs`; modify
`scripts/validate_core_dataset.py` and proposal
`{README,RUST_CLI,WRITING,VALIDATION}.md`.

**Interfaces:** The example accepts one output-directory argument and creates
fresh `generic.pbz`, `imported.pbz`, and `partial.pbz` children, rejecting an
occupied output. It uses only the new public writer/import interface to write
the valid data. A fault-injected storage wrapper creates the partial-success
case; its construction is test support, not a production API.

- [ ] **Add Rust-produced fixture expectations to the existing Python harness.**
  Include alternate reference names, ordinary auxiliary coordinates,
  multidimensional data, scalar/large integers, short terminal bins, native
  and Xarray-encoded Booleans, declared missingness versus zero storage fill,
  and explicit unsharded/sharded codecs. The partial fixture begins with a
  consolidated root, then fails the second metadata publication.

```python
with xr.open_zarr(output / "generic.pbz", chunks=None, consolidated=False) as ds:
    assert ds["total"].item() == (1 << 53) + 1
    assert ds["depth"].dims == ("position", "sample", "threshold")
    assert "contig_length" in ds.coords
    assert "position" not in ds.coords
with xr.open_zarr(output / "partial.pbz", chunks=None) as ds:
    assert "first" in ds.data_vars
    assert "second" not in ds.data_vars
    assert ds["first"].item() == 7
```

  Compare complete arrays to explicit NumPy expectations, including actual
  terminal-bin shape and labels. Also open imported data lazily, construct the
  callable-count expression and confirm zero Dask tasks during construction;
  compute only the small fixture for value comparison. Inspect Zarr metadata
  to verify the supplied chunks/shards/codecs survived exactly in meaning.
- [ ] **Run red:** `pixi run validate-core-dataset`.
  Expect the missing Rust writer example/fixtures before this task's implementation.
- [ ] **Implement the example and harness dispatch.** Reuse the current harness's
  Cargo target-directory discovery and isolated temporary output handling.
  Require both Rust validation and Xarray comparisons for every generated store.
  Keep all previous valid/invalid/unknown-version reader fixtures.
- [ ] **Run final checks:**
  `pixi run validate-core-dataset`,
  `pixi run check`,
  `pixi run install-wheel`, then
  `pixi run validate-roundtrip`.
  Run Python tests if binding-facing behavior changes; this plan should not
  change bindings. Record actual checks, versions and limits, not predictions.
- [ ] **Update proposal status and examples.** Mark writer/import support only
  after the preceding commands pass. Link the new public Rust example and
  keep legacy/default entry points and future CLI/Python work distinct.
  Record coordinate materialization, import dtype/rank limits, physical bool
  import encoding, unconsolidated output, and partial completion behavior.
- [ ] **Commit:** `test: verify Rust core writes through native Xarray`.
- [ ] **Run one fresh final review of the milestone-B commits.** Fix actionable
  correctness findings and rerun only affected checks. Preserve the user's
  selected native execution method. Do not merge, push, or publish a PR
  without a corresponding user request.

## Plan self-review

Schema coverage maps to Tasks 1–2 (reference, coordinates, generic shapes,
encodings, raw values), Task 3 (publication), Tasks 4–5 (existing import engine),
and Task 6 (cross-language evidence). All five Review Focus cases have owning
tests. Unsupported encodings on unrelated measurements remain inspectable and
do not become a writer-wide ban.

Deliberate boundaries: coordinate payloads are in memory in the initial Rust
interface; measurements are bounded writes. Source import keeps its existing
dtype/rank set. Explicit range writes are not a Python view-mapping inference.
No CLI or Python migration, append, replacement, transaction, crash recovery,
or performance claim is included.
