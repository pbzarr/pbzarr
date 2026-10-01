# Proposal validation

## Rust reader milestone — 2026-09-30

The new `pbzarr::dataset::Dataset` is additive and uses `core-draft-1`.
The targeted Rust tests currently pass: four reference tests and seventeen
catalog/validation/read tests. Strict `cargo clippy -p pbzarr --all-targets -- -D warnings` passed.
The probe accepted all seven previously generated valid stores, rejected all
sixteen malformed stores, and distinguished the unknown version.

Fresh fixture generation and the integrated Python harness are pending the
Dask dependency in the declared Pixi environment. `pixi run check` passed: formatting, workspace Clippy, and 237 Rust tests
(including doctests; one example ignored). The fresh Python check is not claimed
as passing.

Instrumented in-memory storage observed zero measurement chunk reads during
opening and validation. A selection from a `[12,3]` uint16 array with `[4,3]`
chunks requested only `depth/c/0/0`, with no requests for the other two chunks.
An empty selection requested no depth chunks. These are request-key assertions
on unsharded fixtures, not memory, throughput, or sharded-I/O benchmarks.

Physical integer values and sentinels remain exact; logical Boolean masks use
the separate decoder. Unsupported required coordinate encodings yield an
incomplete report. Masked logical Booleans, CF scaling/calendar decoding,
production writers, append, GPU execution, and CLI/Python wrapper integration
remain outside this reader milestone.


## Concrete conformance examples — 2026-09-30

Ran [examples.py](examples.py) with the existing project interpreter:

```sh
.pixi/envs/default/bin/python docs/proposals/pbz-core-draft/examples.py --output /private/tmp/pbz-core-examples-20260930-9305
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
