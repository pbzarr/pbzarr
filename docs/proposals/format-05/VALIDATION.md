# Proposal validation

## Rust import milestone — 2026-10-01

Dataset creation and the existing Import builder now support the provisional
root-Dataset convention. Format-0.4 entry points remain the default.

Executed checks:

* `pixi run check` passed formatting, workspace Clippy, and 256 Rust tests
  (one doctest ignored). Strict Clippy for all core targets also passed.
* `pixi run validate-dataset` passed the Rust-created fixture through native
  Xarray, then all seven valid, sixteen invalid, and one unknown-version reader
  cases. The new fixture checked shared reference/sample coordinates, ordinary
  zero/false values, all-fill chunk omission, explicit Zstd settings, and shards
  `[8,2]` with inner chunks `[4,1]` on a `[12,2]` depth array.
* `(depth >= 5).sum("sample")` built without executing Dask tasks and computed
  the expected counts. This checks lazy construction, not GPU support or peak memory.
* Adding a Rust track after native Zarr consolidation preserved existing data,
  sample metadata bytes, and root notes. Stale consolidation was removed, and
  native Xarray's default discovery found the new track. Xarray warned that it
  fell back to unconsolidated metadata, as expected for this milestone.
* Three backend tests passed using existing BigWig/BAM fixtures: gap fills,
  singleton sample dimensions, uncovered contigs, and BAM composition fields
  sharing a two-sample coordinate.
* Core import tests checked metadata publication after chunks, read-only estimates,
  ordered label/reference rejection, orphan-name rejection, multi-field routing,
  chunk/shard choices, explicit codecs, and unrelated unsupported neighbors.
* `pixi run install-wheel` rebuilt the local extension, and
  `pixi run validate-roundtrip` passed the existing format-0.4 round trip.

An additional `cargo clippy -p pbzarr -p pbzarr-readers --all-targets -- -D warnings`
check failed on three pre-existing `disallowed_types` uses in
`crates/pbzarr-readers/tests/common/mod.rs` (lines 15, 109, 110), shared by
several existing test binaries and the new backend test. This helper was not
changed; these are not failures of the project's required `pixi run check`.

Environment: macOS arm64, Rust 1.93.0, pinned zarrs 0.23.13 (`cac400e`),
Python 3.12.13, NumPy 2.4.6, Xarray 2025.12.0, Zarr 3.2.1, Dask 2026.8.0.
Fixtures are small, local synthetic stores. No performance claim is made.

A fresh independent review of the import changes found no actionable issues.
No additional fixes were required.

User-requested scope: Task 3 is skipped. No fault-injection, cleanup, recovery,
or structured partial-completion reporting was added. Failed writes may leave
incomplete output; callers must reopen after an error. CLI/Python integration,
append, replacement, and arbitrary-rank Rust authoring remain outside this step.

## Rust reader milestone — 2026-09-30

The additive `pbzarr::dataset::Dataset` uses `0.5`.
Rust verification passed after independent review and fixes:

* `pixi run check`: formatting, workspace Clippy, and 243 Rust tests passed
  (including doctests; one example ignored).
* `cargo clippy -p pbzarr --all-targets -- -D warnings` passed.
* Four reference tests and 23 catalog/validation/read tests passed.
* All six review findings were reproduced by failing regression tests and fixed:
  native base64 floating missing sentinels; infinite coordinate labels;
  mandatory Zarr extensions; chunk-grid/key/transformer/fill metadata;
  masked physical Boolean capability reporting; and allocation byte bounds.
* The rebuilt Rust probe matched all 47 arrays from the seven previously
  generated valid stores against native Xarray/NumPy. Shapes, coordinate roles,
  references, bin intervals, and selected depth matched. All sixteen invalid
  stores and the unknown version returned the expected outcomes. The generator's
  informal `root schema` label was corrected to executable rule `R2`.

Environment: macOS arm64, Rust 1.93.0, pinned zarrs 0.23.13 (`cac400e`),
Python 3.12.13, NumPy 2.4.6, Xarray 2025.12.0, Zarr 3.2.1.
The comparison used `chunks=None` and the existing synthetic corpus at
`/private/tmp/pbz-core-examples-20260930-9305`; that path is evidence from this
run, not a test dependency.

**Complete:** after approval, added `dask-core>=2024.12` to the Pixi
validation feature and installed the locked dependencies in an independent
worktree environment. `pixi run validate-dataset` generated fresh stores
and passed all seven valid, sixteen invalid, and one unsupported-version case.
It compared values, Boolean masks, dimensions, coordinate roles, references,
bins, and selections; expression construction executed zero Dask tasks.

`pixi run install-wheel` built and installed this worktree's Python extension.
`pixi run validate-roundtrip` then passed the unchanged format-0.4 check.
`pixi run check` passed again with 243 Rust tests and one ignored doctest.
Dask is 2026.8.0; the other recorded versions are unchanged. Milestone A is
complete. These results do not validate a new-format production writer.

Instrumented in-memory storage observed zero measurement chunk reads during
opening and validation. A selection from a `[12,3]` uint16 array with `[4,3]`
chunks requested only `depth/c/0/0`, with no requests for the other two chunks.
An empty selection requested no depth chunks. These are request-key assertions
on unsharded fixtures, not memory, throughput, or sharded-I/O benchmarks.

Physical integer values and sentinels remain exact; logical Boolean masks use
the separate decoder. Unsupported required encodings yield an incomplete report.
Invalid known metadata yields an invalid report. Masked logical Booleans,
CF scaling/calendar decoding, production writers, append, GPU execution, and
CLI/Python wrapper integration remain outside this reader milestone.

Validation loads reference and dimension labels, and reads can revalidate
labels; large-cohort coordinate costs have not been benchmarked. Measurement
payload corruption is detected when reading those payloads, not by metadata
validation. Raw reads, including scalars, require a valid root reference.
Callers must reopen after external mutation because handles and geometry are
cached; this is not a concurrent snapshot guarantee.


## Concrete conformance examples — 2026-09-30

Ran [examples.py](examples.py) with the existing project interpreter:

```sh
.pixi/envs/default/bin/python docs/proposals/format-05/examples.py --output /private/tmp/pbz-core-examples-20260930-9305
```

Environment: Python 3.12.13, NumPy 2.4.6, Xarray 2025.12.0, Dask 2026.8.0,
Zarr 3.2.1. The fixtures use tiny synthetic arrays on the local filesystem.
No dependencies were installed and no production code was changed.

Results:

* Seven valid examples wrote and reopened through native Xarray with the
  expected decoded values, coordinates, attributes, and shapes.
* Callable counts matched an independent NumPy calculation. Logical Boolean
  masks reopened as Boolean, and count arrays retained unsigned 64-bit dtype.
* Signed and unsigned scalar values `2**53 + 1` retained exact integer values
  and their original dtypes, without conversion through floating point.
* Storage fill zero remained an ordinary value. A separate integer `_FillValue`
  sentinel decoded as missing.
* Reference names used string storage, including the empty-reference example.
  An empty NumPy object array otherwise serialized as `float64`; the example
  explicitly supplies `StringDType` and checks the physical encoding. This
  Xarray version reopens those strings as object arrays, so the expected decoded
  representation accounts for that difference.
* Native scalar metadata omitted `dimension_names`. The draft now explicitly
  permits absence or `[]` for scalar arrays.
* A Dask callback observed zero tasks while building the canonical expressions.
  This is a construction-laziness check, not a memory or performance benchmark.
* Sixteen malformed stores and one unknown-version store were generated with
  documented expected outcomes. No production PBZ validator was run against
  them; rejection coverage remains implementation work.

The [conformance page](CONFORMANCE.md) records those fixtures and the limits of
the initial encoding profile. Proposal JSON parsing, Python syntax, and local
Markdown file-link checks also passed. `git diff --check` passed for tracked
changes. No Rust tests, production Python suite, or cross-language round trip was
run as part of this documentation/fixture work.

An external JSON Schema validator is still unavailable in the project
environment. The root schema was parsed and its rules reviewed, not executed
through such a validator.

## Earlier native example — 2026-09-26

Checked 2026-09-26 using the existing project environment: Xarray 2025.12.0,
Zarr 3.2.1, Python 3.12. The example used Dask-backed measurements on the local
filesystem. No production reader or writer was changed.

## Native example

Reference: chr1 length 1500 and chr2 length 2200. Labels: sample_1, sample_2,
sample_3; thresholds 5 and 10.

The Dataset contained depth, a three-dimensional callable mask, per-threshold
callable counts, a sample-level mean, and independent tracks on bin_1000 and
bin_1500. Their genomic extents were respectively 3700, 5, and 3.
The reference coordinate arrays were resolved in memory; measurement arrays
remained lazy until writing.

The following checks passed:

* Native Xarray Zarr v3 write followed by open_zarr and assert_identical.
* Reference arrays and dimension labels reopened as coordinates; measurement
  arrays reopened as data variables.
* Per-threshold callable counts matched an independent NumPy calculation.
* The callable variable retained dimensions position, sample, threshold.
* The emitted root attributes exactly matched example-root.zarr.json.
* A newly allocated, full-reference regional_count array had shape 3700, chunk
  length 100, and storage fill zero. Writing positions [100, 200) stored one
  chunk out of 37 logical chunks. Native reopening recovered the selected
  values and zeros elsewhere.
* An extracted depth DataArray did not contain the independent reference arrays,
  confirming that the proposed Track wrapper must retain reference context.

This demonstrates the proposed layout and a small regional write. It is not
a performance benchmark, append-transaction test, or claim of production PBZ
support.

## Implementation details exposed by the example

Native broadcasting initially put threshold before position. The example
explicitly transposed to position, sample, threshold before serialization.
This is straightforward normalization for the proposed writer.

Xarray encoded the logical Boolean mask as int8 with its ordinary Boolean
encoding attribute. Native Xarray reopened it as Boolean. Physical Zarr dtype
alone is therefore insufficient to infer every logical Xarray dtype.

## JSON and documentation checks

Both JSON artifacts parsed, their draft versions matched, and the root example
matched Xarray's actual emitted metadata. The schema was reviewed as JSON
Schema draft 2020-12, but no external JSON Schema validator was available in
the checked project or bundled runtimes. No packages were installed.

Cross-array geometry, coordinate classification, and uniqueness still require
a future PBZ store validator; the root-attribute JSON Schema cannot check them.
