# PBZ Core Reader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Rust inspect, validate, and read the new root-Dataset layout,
including Python-written multidimensional, binned, non-genomic, and scalar data.

**Architecture:** Add an explicit new-format Dataset reader around existing Zarr
array handles. Keep raw storage access separate from logical encoding support
and genomic interpretation. Integrate importers, writers, and user-facing CLI
commands after this reader passes the shared conformance corpus.

**Tech Stack:** Existing Rust workspace, zarrs 0.23, ndarray 0.17, serde_json,
project-local Pixi, native Xarray/Zarr/Dask for cross-language fixtures. No new
array engine or oxiphysics dependency.

**Spec:** [Core convention](../../proposals/pbz-core-draft/README.md),
[conformance cases](../../proposals/pbz-core-draft/CONFORMANCE.md), and
[accepted Rust/CLI decisions](../../proposals/pbz-core-draft/RUST_CLI.md).

## Global constraints

* "Conversion from format 0.4 is outside the initial work."
* "The token `core-draft-1` remains provisional."
* "An unsupported encoding on an unrelated variable does not block supported variables."
* "A storage fill alone does not exclude values from a reduction."
* "Core stores omit explicit coordinate variables for position and bin_N, so the implicit mapping has a single authority."
* "Explicit chunk, shard, and codec choices remain authoritative."
* Source data and fixture output must use distinct paths. Tests use temporary stores.
* Use project-local Pixi. Do not install tools or alter the user's existing work.
* This plan does not change the existing `PBZ_FORMAT_VERSION = "0.4"`, default
  opener, current CLI, or current-format fixtures. This temporary integration
  sequence makes no promise about legacy support in the eventual release.
* Use the new module's `FORMAT_VERSION = "core-draft-1"` for development.
  Assigning a published version belongs to the release milestone.

## Review focus

1. Dataset inspection must not fetch measurement chunks, including unsupported variables. Task 2 supplies an instrumented storage test.
2. Large integer reference lengths, numeric labels, and counts must not pass through `f64`. Tasks 1, 3, and 4 test overflow and values above `2**53`.
3. Coordinate classification must follow native Xarray metadata, including alternate reference names and ordinary auxiliary coordinates. Tasks 2 and 3 exercise these cases.
4. A short terminal bin must keep its original interval and have no inferred aggregation meaning. Task 1 pins bin geometry; CLI weighting is a later milestone.
5. Masked logical Booleans and unavailable coordinate codecs must produce explicit capability limits, rather than wrong values or a false full-validation success. Tasks 3 and 4 test both.

## Overall implementation sequence

This is the detailed plan for **milestone A**. Each subsequent milestone receives
its own implementation plan against the preceding milestone's working interface.

| Milestone | Independently testable result |
| --- | --- |
| A. Rust reader and conformance | Python's valid example stores can be cataloged, checked, and selectively read in Rust. Invalid examples produce the expected error category. |
| B. Rust creation and import | Create root reference/coordinates; import existing 1D/2D inputs into root variables; add compatible tracks with preflight before mutation. Rust-written output opens directly in Xarray. Preserve explicit storage configuration. |
| C. CLI | Inspect arbitrary variables; select named dimensions for `view`/`stat`; handle bins as observations; honor missingness; reject unsupported output shapes clearly. |
| D. Python API and derived writing | Promote Dataset/Track composition, genomic indexing, wrapper-owned windows, native escape/re-entry, resource lifetime, and supported new-track writes. Keep Dask lazy and avoid unnecessary NumPy conversion. |
| E. Integrated workflow and release | Import → select → callable counts → windows → write → reopen across Rust/Python. Record reads/tasks/memory/write behavior, publish the format version, and reconcile the maintained spec, roadmap, examples, and release docs. |

Append is a separate milestone after ordinary new-track writing. Its existing
contract extends all arrays using the appended dimension and fills unspecified
values. Pyramids, automatic provenance, GPU implementation, conversion, and the
separate region-reduction redesign are not prerequisites for A–E.

The existing roadmap describes the old release scope. This new plan is a
separate, approved design direction; it must not be confused with restoring the
withdrawn track-group root-reference plan. Current-behavior documentation changes
only when the corresponding implementation exists.

## Files and ownership for milestone A

| Path | Responsibility |
| --- | --- |
| `crates/pbzarr/src/dataset/mod.rs` | Public new-format Dataset reader and orchestration. |
| `crates/pbzarr/src/dataset/metadata.rs` | Root/array catalog, coordinate classification, validation report and capability metadata. |
| `crates/pbzarr/src/dataset/reference.rs` | Checked reference geometry and genomic row/interval conversion. |
| `crates/pbzarr/src/dataset/read.rs` | Typed rectangular reads, Boolean decoding, and reference/label value loading. |
| `crates/pbzarr/src/lib.rs` | Export `pub mod dataset`; preserve old exports/constants. |
| `crates/pbzarr/src/error.rs` | Explicit unsupported-format/encoding errors, retaining existing variants. |
| `crates/pbzarr/tests/core_dataset.rs` | Public-interface behavior, validation categories, selected reads, storage-access assertions. |
| `crates/pbzarr/tests/core_reference.rs` | Public reference geometry and integer-bound tests. |
| `crates/pbzarr/examples/check_core_dataset.rs` | Machine-readable conformance probe invoked by Python. |
| `scripts/validate_core_dataset.py` | Generate proposal fixtures, invoke Rust, compare classifications and actual values. |
| `pixi.toml` | Add `validate-core-dataset`; preserve existing tasks. |
| `docs/proposals/pbz-core-draft/VALIDATION.md` | Record measured results and limits after execution. |

The existing `io::Dtype` is also used by import routing, buffers, statistics,
scales, CLI dispatch, and bindings. Do not extend that enum merely to read a
Zarr `uint64` array in this milestone. Use zarrs `DataType` and `ElementOwned`
for typed storage reads. Importer dtype integration belongs to milestone B.

## Interfaces shared by the tasks

These are proposed implementation interfaces, not additions to the on-disk
schema. Define all types in the new module and re-export them there.

```rust
pub const FORMAT_VERSION: &str = "core-draft-1";

pub enum GenomicAxis { Position, Bin(std::num::NonZeroU64) }
pub struct BinInterval { pub contig: String, pub start: u64, pub stop: u64 }

impl Reference {
    pub fn new(contigs: Vec<crate::Contig>) -> crate::Result<Self>;
    pub fn contigs(&self) -> &[crate::Contig];
    pub fn extent(&self, axis: &GenomicAxis) -> crate::Result<u64>;
    pub fn rows(&self, axis: &GenomicAxis, query: &crate::RegionQuery)
        -> crate::Result<std::ops::Range<u64>>;
    pub fn interval(&self, axis: &GenomicAxis, row: u64)
        -> crate::Result<BinInterval>;
}

pub enum Role { Coordinate, Measurement }
pub enum Support { Supported, Unsupported { reason: String } }
pub struct VariableInfo {
    pub name: String,
    pub dims: Vec<String>,
    pub shape: Vec<u64>,
    pub role: Role,
    pub metadata: serde_json::Value,
    pub logical_support: Support,
}
pub enum ValidationStatus { Valid, Invalid, Incomplete }
pub struct ValidationIssue {
    pub array: Option<String>,
    pub rule: String,
    pub message: String,
}
pub struct ValidationReport {
    pub status: ValidationStatus,
    pub issues: Vec<ValidationIssue>,
}

impl Dataset {
    pub fn open(path: impl AsRef<std::path::Path>) -> crate::Result<Self>;
    pub fn open_with_storage(storage: zarrs::storage::ReadableWritableListableStorage)
        -> crate::Result<Self>;
    pub fn variables(&self) -> &std::collections::BTreeMap<String, VariableInfo>;
    pub fn reference(&self) -> crate::Result<&Reference>;
    pub fn validate(&self) -> crate::Result<ValidationReport>;
    pub fn read_raw<T: zarrs::array::ElementOwned>(
        &self, name: &str, ranges: &[std::ops::Range<u64>],
    ) -> crate::Result<ndarray::ArrayD<T>>;
    pub fn read_bool(&self, name: &str, ranges: &[std::ops::Range<u64>])
        -> crate::Result<ndarray::ArrayD<bool>>;
}
```

`open` builds a catalog and checks the root identity/version. Full `validate`
checks the PBZ rules and reports unsupported dependencies as incomplete.
Selection/read operations must check their own required invariants even if a
caller has not requested full validation. `variables()` exposes metadata;
it is not evidence that every variable can be interpreted.

`read_raw` returns physical values, including sentinels. It neither applies
`_FillValue` nor converts integers to floats. This naming is intentional. Logical
missingness is retained in metadata for the later CLI/analysis layer.
`read_bool` handles physical Boolean arrays and native Xarray's unmasked
`int8`/`dtype="bool"` encoding; masked logical Booleans return an unsupported
encoding error. Do not present raw integer values as decoded Booleans.

Named label selection is a later consumer of the dimension/coordinate catalog.
The first storage reader accepts one positional range per dimension. It must
support arbitrary rank, empty ranges, and `[]` for a scalar without flattening
or dropping axes.

## Task 1: Checked reference and bin geometry

**Files:** Create `dataset/mod.rs`, `dataset/reference.rs`, and
`tests/core_reference.rs`; modify `lib.rs` to expose the module.

**Interfaces:** Produces `Reference`, `GenomicAxis`, and `BinInterval` above;
consumes existing `Contig` and `RegionQuery` only. Do not use the order-insensitive
genome checksum as reference identity or call unchecked prefix-offset arithmetic.

- [ ] Add this public test and the corresponding overflow/empty cases:

```rust
use pbzarr::{Contig, RegionQuery};
use pbzarr::dataset::{GenomicAxis, Reference};
use std::num::NonZeroU64;

#[test]
fn bins_restart_at_each_contig() {
    let reference = Reference::new(vec![
        Contig { name: "chr1".into(), length: 5 },
        Contig { name: "chr2".into(), length: 7 },
    ]).unwrap();
    let axis = GenomicAxis::Bin(NonZeroU64::new(4).unwrap());
    assert_eq!(reference.extent(&axis).unwrap(), 4);
    let query: RegionQuery = "chr2:5-6".parse().unwrap();
    assert_eq!(reference.rows(&axis, &query).unwrap(), 3..4);
    let interval = reference.interval(&axis, 3).unwrap();
    assert_eq!((interval.contig.as_str(), interval.start, interval.stop), ("chr2", 4, 7));
}

#[test]
fn rejects_reference_extent_overflow() {
    assert!(Reference::new(vec![
        Contig { name: "a".into(), length: i64::MAX as u64 },
        Contig { name: "b".into(), length: 1 },
    ]).is_err());
    assert!(Reference::new(vec![Contig { name: "a".into(), length: u64::MAX }]).is_err());
}
```

- [ ] Run `pixi run cargo test -p pbzarr --test core_reference`; record the
  expected missing-interface failure before implementation.
- [ ] Implement prefix offsets with `checked_add`; reject empty/duplicate names
  and lengths/totals above `i64::MAX`. Calculate ceil division as
  `length / width + u64::from(length % width != 0)` without adding `width - 1`.
  Preserve reference order. Check query bounds before mapping intersecting bins.
- [ ] Add checks for `[]`, `[0,5,0]`, widths `1` and larger than a contig,
  a query ending exactly at a bin boundary, unknown contigs, and out-of-bounds
  rows. Empty genomic axes accept no nonempty region. Reject empty genomic
  queries explicitly; raw positional selections in Task 4 can be empty. No
  empty query may select a bin merely by rounding.
- [ ] Re-run the focused test; format the new files. Commit only this task's
  files as `feat: add checked core reference geometry`.

## Task 2: Metadata-only root catalog

**Files:** Create `dataset/metadata.rs` and `tests/core_dataset.rs`; modify
`dataset/mod.rs` and `error.rs`.

**Interfaces:** Produces `Dataset::open`, `open_with_storage`, `variables`,
`VariableInfo`, `Role`, and `Support`. Add `PbzError::UnsupportedFormat(String)`
and `PbzError::UnsupportedEncoding(String)` so callers can distinguish failure
categories without parsing prose.

- [ ] Use a local temporary store with hand-authored root/array `zarr.json`
  files, no measurement chunks, and the accepted three root attributes. Add an
  assertion that the resulting catalog has `depth` as Measurement and `contig`
  plus `contig_length` as Coordinate. Repeat with `sequence`/`sequence_length`.
  The probe in Task 5 will supply full Python-generated arrays.

```rust
let dataset = Dataset::open(&path).unwrap();
assert!(matches!(dataset.variables()["depth"].role, Role::Measurement));
assert!(matches!(dataset.variables()["contig_length"].role, Role::Coordinate));
assert_eq!(dataset.variables()["depth"].dims, ["position", "sample"]);
```

- [ ] Run `pixi run cargo test -p pbzarr --test core_dataset` and confirm these
  new assertions fail before the reader exists.
- [ ] Read direct root array metadata without constructing a decoded Zarr
  handle for every measurement. Detect unsupported codecs/dtypes per variable;
  retain raw metadata and the variable's name instead of omitting it. Validate
  the root JSON envelope and convention UUID before interpreting references.
  Recognize dimension coordinates plus root/variable `coordinates` declarations
  according to native Xarray encoding. Do not assume names such as `values`,
  `contigs`, or `offsets` identify arrays in this new format.
- [ ] Add a `ReadableStorageTraits` delegating test wrapper that records requested
  keys using `Arc<Mutex<Vec<String>>>`; forward listing/size operations to its
  MemoryStore. Assert catalog opening reads metadata keys only, including when
  an unrelated measurement has an unknown codec name. Test a scalar without
  `dimension_names`, nested groups outside the root catalog, and unknown versions.
- [ ] Re-run focused tests and `pixi run cargo test -p pbzarr --test store_roundtrip`
  to verify current-format opening still behaves as before. Commit as
  `feat: catalog root dataset variables without reading measurements`.

## Task 3: Executable PBZ validation and coordinate interpretation

**Files:** Modify `dataset/metadata.rs`, `dataset/read.rs`, `dataset/mod.rs`,
and `tests/core_dataset.rs`; add `ValidationStatus`, `ValidationIssue`, and
`ValidationReport` with the exact fields above.

**Interfaces:** Produces `Dataset::validate` and `reference`; consumes Task 1's
checked geometry and Task 2's catalog. Reference/label buffers may be loaded;
measurement and per-position auxiliary buffers may not be materialized merely
to validate metadata/geometry. Report uncheckable content as incomplete.

- [ ] Add table-driven public validation assertions for conformance rules R1–R6,
  D1–D4 and G1–G4 using small temporary metadata/coordinate stores. Include the
  case where `contig_length` has correct values but is classified as a track.

```rust
let report = dataset.validate().unwrap();
assert!(matches!(report.status, ValidationStatus::Invalid));
assert!(report.issues.iter().any(|issue| issue.rule == "R3"));
```

- [ ] Run `pixi run cargo test -p pbzarr --test core_dataset` and confirm a real
  missing validation behavior before adding it.
- [ ] Load reference names as strings and reference lengths as signed/unsigned
  integers. Check missing declarations before conversion; never use `f64` as an
  intermediate for lengths or integer labels. Reuse checked `Reference::new`.
  Validate numeric labels in their stored domain, handling signed/unsigned
  values exactly and floating `-0.0`/`0.0` as the same label; NaN labels are missing.
  Repeated values are permitted in ordinary auxiliary coordinates.
- [ ] Validate all dimension extents from metadata, canonical bin spelling,
  contig-local bin totals, one leading genomic axis per measurement, and absence
  of stored genomic dimension coordinates. Treat absent scalar dimension names
  as `[]`; reject missing names on nonscalars. Ordinary genomic auxiliary
  coordinates must have compatible geometry but need not become data variables.
  Keep the original bin dimension name. For arithmetic, a width greater than
  `i64::MAX` can use that bound as its effective width because no valid contig
  is longer; compare decimal strings before parsing an oversized integer.
  Thus a very large canonical width is not incorrectly rejected or conflated
  with a different dimension name. Add a one-bin-per-nonempty-contig case with
  a width greater than `u64::MAX`.
- [ ] Add cases with labels `2**53` and `2**53 + 1`, missing-sentinel labels,
  two same-length but differently ordered references, a `bin_1` variable,
  repeated auxiliary labels, and an unsupported required coordinate codec.
  Unsupported dependencies yield `Incomplete`, not `Valid`; a confirmed violated
  invariant yields `Invalid` even when some other encoding is unsupported.
- [ ] Run the focused tests and confirm measurement-chunk read counts remain
  zero. Commit as `feat: validate core dataset coordinates and geometry`.

## Task 4: Typed reads with bounded selections

**Files:** Modify `dataset/read.rs`, `dataset/mod.rs`, and
`tests/core_dataset.rs`.

**Interfaces:** Produces `read_raw<T: ElementOwned>` and `read_bool`; consumes
validated variable metadata and cached Zarr handles. The existing zarrs
`Element::validate_data_type` checks the requested physical type, and
`retrieve_array_subset::<ArrayD<T>>` performs the actual storage access.

- [ ] Add tests reading a small `(position, sample, threshold)` subset, an
  independent bin array, a non-genomic vector, and a scalar. Write fixture
  values with existing `zarrs::ArrayBuilder` in the test; this is not a new
  production writer interface. Pin the raw-value contract:

```rust
let scalar = dataset.read_raw::<u64>("large_unsigned", &[]).unwrap();
assert_eq!(scalar.shape(), &[]);
assert_eq!(scalar[ndarray::IxDyn(&[])], (1_u64 << 53) + 1);
let counts = dataset.read_raw::<i16>("missing_count", &[0..3]).unwrap();
assert_eq!(counts.iter().copied().collect::<Vec<_>>(), vec![0, 1, -1]);
let part = dataset.read_raw::<u16>("depth", &[2..4, 1..3]).unwrap();
assert_eq!(part.shape(), &[2, 2]);
```

- [ ] Run `pixi run cargo test -p pbzarr --test core_dataset` and record the
  expected failure before implementing read methods.
- [ ] Construct one Zarr `ArraySubset` range per named axis, check range count
  and bounds before I/O, and preserve requested rank. Use the zero-axis subset
  for scalars. Return correctly shaped empty arrays for empty ranges without
  reading measurement chunks. Cache opened handles without caching all values.
- [ ] Implement `read_bool` for physical bool and unmasked Xarray `int8` Boolean
  encoding; do not mask then cast a NaN to bool. Reject masked logical Booleans
  explicitly. Raw physical reads remain possible by their explicit contract.
  An unsupported measurement must not prevent reading a supported neighbor.
- [ ] Check storage keys for a small selection against a multi-chunk array:
  only intersecting chunks are read in an unsharded fixture; a sharded fixture
  may fetch its containing shard, which must be reported separately. Test a
  wrong requested Rust element type and out-of-bounds selection before access.
- [ ] Re-run tests, then `pixi run cargo clippy -p pbzarr --all-targets -- -D warnings`.
  Commit as `feat: read core arrays without rank or integer precision loss`.

## Task 5: Cross-language conformance command and completion evidence

**Files:** Create `examples/check_core_dataset.rs` and
`scripts/validate_core_dataset.py`; modify `pixi.toml` and the proposal validation
record. Preserve `scripts/validate_xarray_read.py` and its current-format task.

**Interfaces:** The Rust example takes a store path and prints a JSON report
with `status`, `issues`, variable names/dimensions/shapes, and selected sample
values from recognized fixture variables. Status is one of `valid`, `invalid`,
`incomplete`, or `unsupported_version`. Expected fixture errors produce a report;
unexpected I/O errors exit nonzero. JSON integers must remain integers.

- [ ] Add a Python integration script that imports the proposal's `examples.py`
  by its file path, runs `generate` inside a TemporaryDirectory, and invokes
  the Rust probe for each manifest entry. Compare decoded results with NumPy,
  rather than merely checking that Rust exited successfully.

```python
completed = subprocess.run(
    [str(probe), str(store)], check=True, text=True, capture_output=True,
)
report = json.loads(completed.stdout)
assert report["status"] == expected_status
if store.name == "values-and-missingness.pbz":
    assert report["values"]["large_unsigned"] == 2**53 + 1
    assert report["values"]["missing_count_first_three"] == [0, 1, -1]
```

- [ ] Build the probe once with
  `pixi run cargo build -p pbzarr --example check_core_dataset`. Resolve its
  directory from `cargo metadata --no-deps --format-version 1` so an existing
  `CARGO_TARGET_DIR` is respected. Run the script and verify it reports a real
  disagreement until all fixture outcomes and values are implemented.
- [ ] Include all seven valid stores, sixteen malformed stores, and the unknown
  version. Compare 3D masks through `read_bool`, raw signed/unsigned values,
  sample means, selected reference/bin intervals, coordinate classification,
  and the unchanged zero-fill values. Add the extra validation cases from
  Tasks 1–4 to the shared corpus where useful to future Python validators.
- [ ] Add `validate-core-dataset = "python scripts/validate_core_dataset.py"`
  under the existing validation tasks. Run `pixi run validate-core-dataset`,
  `pixi run check`, and the current `pixi run validate-roundtrip`. If the existing
  Python extension is unavailable, use the documented `pixi run install-wheel`
  before that current-format check; do not install a second dependency stack.
- [ ] Update the proposal validation record with actual dependency versions,
  commands, pass/fail results, and storage read counts. Keep native construction
  laziness distinct from performance. Do not describe this read-only milestone
  as validating Rust-created new stores, append, or production Python wrappers.
- [ ] Commit only the task files as `test: check core dataset fixtures across Rust and Xarray`.

## Completion and handoff

Milestone A is complete when the fixture expectations are executable, selected
values retain shape/type/coordinate meaning, measurements stay unread during
catalog/geometry validation, and current-format regression checks pass. Any
unsupported required encoding must be visible in the report.

Milestone B then plans the new writer and importer integration. It must settle
root-array publication/failure behavior explicitly: the old track-group marker
written after population cannot simply be copied to root data variables. This
plan does not add a transaction protocol or infer a concurrency guarantee.

The eventual release also needs an explicit legacy-reading decision and a
published format version. Neither is necessary to execute this separate draft
reader milestone, and neither is silently decided by keeping current commands
working while development proceeds.

## Plan review record

The plan was checked against the core proposal and accepted Rust/CLI decisions.
Its detailed tasks cover the reader/conformance portion only; the milestone
table explicitly assigns writing, imports, CLI, Python, and release work.
Review-focus cases each have an owning task. No production code or test execution
is claimed by writing this plan.
