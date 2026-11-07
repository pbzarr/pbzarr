# PBZ format specification

Status: draft, updated 2026-11-06. This directory is the home of the PBZ
specification. It describes format `0.5`, which imports write and the Python
`pbzarr.Dataset` handle reads. The rules can still change before the 1.0
release candidate; `0.5` is not a published, stable format version. Format
`0.4` (one group per track) is legacy and is not described here.

[Conformance examples and validation rules](CONFORMANCE.md) make the contract
reviewable. [Rust and CLI decisions](../docs/proposals/format-05/RUST_CLI.md) define the agreed implementation
scope. The [validation record](../docs/proposals/format-05/VALIDATION.md) distinguishes executed checks from
future Rust/Python conformance requirements.
The Rust `pbzarr::dataset::Dataset` reader implements milestone A of this
specification. Dataset creation
and existing 1D/2D imports implement milestone B; the validation record
distinguishes executed checks from work still pending.
Milestone B's [writing contract](../docs/proposals/format-05/WRITING.md) records what imports write.

## Model

A PBZ store is a Zarr v3 root group that opens as one ordinary Xarray Dataset.
The reference is a set of root arrays with fixed names. A track is a root array
whose first dimension is `position` or `bin_N`. Other root arrays are dimension
labels or non-genomic results, such as a sample-level mean or a scalar.

The core adds two root attributes:

| Attribute | Purpose |
| --- | --- |
| zarr_conventions | Identify the PBZ convention using its UUID. |
| pbz:version | Identify this encoding independently of package versions. |

No PBZ attributes are required on individual tracks. Native Zarr metadata owns
shape, dimension names, dtype, fill, chunks, and codecs. Readers classify arrays
by name and dimension names only. Descriptions, units, notes, and user-defined
JSON attributes are optional.

The Xarray `coordinates` attribute has no PBZ meaning. Writers MAY emit it, so
that Xarray shows the reference arrays as coordinates. Readers MUST NOT depend
on it. A store without the attribute is the same PBZ store.

Pyramids, multiscales, automatic operation capture, provenance graphs, and
transactional storage are outside this proposal. The optional `contig_md5`
array is the only checksum this draft defines.
Sample/column grouping uses ordinary Xarray coordinates and operations; its
workflows belong in API documentation and tutorials, not the core schema.

## Example store

    example.pbz/
      zarr.json
      contig_id              (contig)                       ["chr1", "chr2"]
      contig_length          (contig)                       [1500, 2200]
      sample                 (sample)                       ["sample_1", "sample_2", "sample_3"]
      threshold              (threshold)                    [5, 10]
      depth                  (position, sample)             shape [3700, 3]
      callable               (position, sample, threshold)  shape [3700, 3, 2]
      callable_count         (position, threshold)          shape [3700, 2]
      coverage               (bin_1000, sample)             shape [5, 3]
      signal                 (bin_1500, sample)             shape [3, 3]
      mean_depth             (sample)                       shape [3]
      total_callable_count   ()                             scalar

Each entry other than zarr.json is an ordinary Zarr array. The names
`contig_id` and `contig_length` and the dimension `contig` are fixed. The other
array names are examples. The genomic dimension names are the reserved names
described below.

Both bin widths are independent user choices. Neither binned track needs a
base-resolution source, an aggregation declaration, or a relation to another track.

In this example, callable indicates whether each sample meets each depth
threshold. Summing it over sample produces callable_count. These names and
calculations illustrate ordinary tracks; the schema does not prescribe them.

[example-root.zarr.json](example-root.zarr.json) contains the example root
metadata. Its `coordinates` attribute is Xarray's encoding for the reference
arrays, not a PBZ field. Python writers can put the reference arrays in
Dataset.coords and let Xarray write that attribute. The Rust writer emits the
same attribute.

## Reference

The reference is the ordered list of contig names and lengths. It lives in
direct root arrays with fixed names over the dimension `contig`.

1. `contig_id` MUST exist, with the single dimension `contig` and a string
   dtype. Names are unique, nonempty strings. Missing names are invalid.
2. `contig_length` MUST exist, with the single dimension `contig` and an
   integer dtype. It has one nonnegative length per name. Boolean and floating
   dtypes are not integer length encodings.
3. `contig_md5` is optional. If it exists, it has the single dimension `contig`
   and a string dtype.
4. Index `i` in every reference array refers to the same contig. The order is
   the concatenation order. Readers derive offsets by prefix sum over
   `contig_length`.
5. The reference arrays are coordinates by name, not tracks. The dimension
   `contig` has no array named `contig`; `contig_id` holds its labels.

This draft permits zero-length contigs and an empty reference. Lengths and the
total reference extent must fit signed 64-bit indexing. Integer storage and
exact interpreted lengths are checked separately; see the
[reference validation rules](CONFORMANCE.md#reference).

The coordinate convention is fixed by the version: zero-based, half-open
contig-local bounds. It needs no repeated metadata attribute.

Names and lengths describe geometry. They do not prove that two reference
assemblies have identical sequence bases. An assembly label or other identifying
information can be supplied as descriptive metadata.

## Dimensions and genomic geometry

Every non-scalar array has non-null, distinct native Zarr dimension_names, one
per axis. A scalar has shape [] and dimension_names either absent or []; native
Xarray omits this field for scalars. Dimension names are nonempty strings.
Variables sharing a dimension name have the same extent and coordinate identity.

Genomic dimensions are:

| Name | Extent | Interpretation |
| --- | --- | --- |
| position | sum(L_i) | Concatenate all bases of every contig in reference order. |
| bin_N | sum(ceil(L_i / N)) | Concatenate contig-local bins of N bases in reference order. |

N is any positive integer. Canonical spelling is bin_[1-9][0-9]*;
reserve the bin_ prefix for genomic dimension names. For local bin k:

    start = k * N
    end   = min((k + 1) * N, contig_length)

Bins restart at zero in every contig. Readers derive the per-contig bin offsets;
no bin-boundary or per-base coordinate arrays are required. Core stores omit
explicit coordinate variables for position and bin_N, so the implicit mapping
has a single authority.

A genomic data variable has exactly one genomic dimension, and it is first.
Any number of labeled non-genomic dimensions may follow. Non-genomic results
and scalars are valid data variables without this first-axis requirement.
Every array with a genomic dimension follows the same geometry, so it is a track.

Different bin_N dimensions need not divide one another. The spelling bin_1 is
allowed under the positive-width rule; it describes the same locations as
position but remains a different named dimension to Xarray. Writers must not
silently equate or rename those dimensions.

## Dimension labels and other root arrays

Every non-genomic dimension other than `contig` has a one-dimensional
coordinate array with the same name, matching length, and unique, nonmissing
labels. Labels can be strings or numbers; they are not restricted to sample
identifiers. The labels of `contig` are in `contig_id`.

Other root arrays without a genomic dimension, such as a
`sample_group(sample)` array, may contain repeated values. PBZ assigns no
application-specific meaning to them. A writer MAY declare them as Xarray
coordinates; PBZ readers treat them the same either way.

Tracks and coordinates share one root namespace. Measurement names must not
collide with coordinate or dimension names.

## Values and user metadata

Zarr storage fill_value is required by Zarr and describes unwritten values.
The optional standard Xarray _FillValue declaration identifies values to decode
as missing. Zero and false keep their ordinary meaning unless explicitly
declared missing. A matching measured value cannot be distinguished from a
missing sentinel with the same encoding.

Coordinate identity is checked after the declared standard decoding. For
example, decoding a label sentinel as missing must not create an unlabeled
sample. Missing-value decoding can change an integer or Boolean array's
in-memory dtype; the stored dtype and decoded dtype need not be identical.

The convention does not add a dtype, chunking, or codec registry. Implementations
must document supported standard Zarr/Xarray encodings and reject unsupported
encodings rather than silently reinterpret values. This does not promise that
the current Rust reader supports all encodings that Xarray can write.

Array attributes may contain descriptions and arbitrary JSON notes:

    {
      "description": "Number of samples meeting each depth threshold.",
      "notes": "Computed from depth for the stored samples."
    }

These attributes have no special PBZ interpretation. No automatic lineage,
replay, recomputation, or source-version guarantee is implied.

## Python workflow and retained context

The earlier [Python API proposal](../docs/proposals/format-05/PYTHON_API.md) sketched Dataset/Track
wrappers, native Xarray integration, and the write boundary.
The agreed Python API is in
[the Python API plan](../docs/plans/2026-10-05-python-api-design.md): a store
handle in 1.0, and a runtime genomic index in 1.1. Runtime indexes are not
serialized as genomic coordinate arrays.

The proposed wrappers and accessors use the same stored Dataset:

| Workflow | Information already available |
| --- | --- |
| Open and list tracks | Convention marker, the reference arrays by name, and the first dimension of each array. |
| Select a region | Ordered lengths plus position/bin_N; retained view-to-source mapping. |
| Select samples or thresholds | Ordinary dimension coordinates. |
| Compute | Native array operations; no expression schema is required. |
| Attach notes | Ordinary Dataset or DataArray attributes. |
| Save a result | Result dimensions/coordinates plus retained or explicitly supplied reference context. |

Native broadcasting can change dimension order. A PBZ writer can move the
genomic axis to first position before serialization while preserving named
dimensions and values.

A Track wrapper retains its owning Dataset's reference context. A detached
native DataArray generally does not contain the independent reference arrays or
root attributes. Genomic accessors on a detached array require explicit context
or a retained in-memory binding. Shape or dimension names alone cannot recover
that context. This requires no duplicated reference metadata in stored tracks.

In particular, arbitrary native slicing or reordering can invalidate a claimed
genomic mapping. Writers must have a trustworthy mapping before placing a
regional result; ordinary inherited attributes are not proof of that mapping.

## Writing policies supported by this schema

These are API contracts; they add no required metadata fields.

* New tracks use compatible shared dimensions and coordinate identities.
  Writers check reference order and lengths, labels, dimensions, and supplied
  mapping before changing the store.
* Creating a track fails if its name exists unless replacement is explicit.
  Replacing a whole variable and patching a region are distinct operations.
* A regional result can be written to a full-reference-shaped destination.
  The writer initializes the full shape and scatters selected values to their
  original positions. Unwritten positions use the destination's declared fill.
  This is not storage of a shortened genomic axis.
* The same explicit selection mapping can place selected columns into existing
  label positions. It must not silently union or reorder coordinates.
* Warn users about the fill value outside a stored selection. Unwritten chunks
  need not exist; touched chunks and write scheduling still have costs to profile.
* A genuinely changed shared coordinate set requires a new Dataset or an
  explicit supported mutation such as append. A full-reference sparse write
  does not change the coordinate set.
* Appending along a named non-genomic dimension requires actual new unique
  labels. Extend every root array using that dimension; write supplied values
  and use each other array's declared storage fill. Dimension coordinates
  receive the supplied new labels, not fill labels.
* The reference is fixed within a store. Generic column append cannot grow the
  reference's contig dimension; a changed reference belongs in a new Dataset.
  Unspecified categorical annotations may use their declared fill, which need
  not have a special missing interpretation.
* Arrays without the appended dimension keep their existing values and meaning.
  Derived results do not automatically recompute.

The current Rust import path populates all targets before publishing their
metadata in request order. It preserves existing tracks and removes stale
consolidated metadata so native Xarray can discover successful additions. The
[Rust writing contract](../docs/proposals/format-05/WRITING.md#publication-and-failure) records the limits:
errors propagate normally, incomplete output may remain, and callers must reopen
after an error. Cleanup and uncertain-write reporting are deferred.

This draft does not promise atomic multi-array updates, concurrent writers, or
crash recovery.

## What the JSON Schema checks

[root-attributes.schema.json](root-attributes.schema.json) validates the root
attribute document. Format 0.5 has its own convention UUID,
`d5bd7bea-0388-4e0a-9844-707c034e9add`. The 0.4 UUID `b7e3c1a2-...` identifies
the per-track group layout, and readers report it as an unsupported legacy
format.
No schema URL is invented; a versioned schema/spec URL should be added when
published. Names in zarr_conventions are descriptive, not the identity check.

A store validator must additionally find the reference arrays by name and
check the reference, shared dimensions, uniqueness, genomic extents, and axis
ordering above. JSON Schema alone cannot validate array contents or perform
the required cross-array arithmetic.

## References

The [validation record](../docs/proposals/format-05/VALIDATION.md) describes the native Xarray example checks.

* [Zarr conventions](https://github.com/zarr-conventions/zarr-conventions-spec)
* [Zarr v3 core](https://zarr-specs.readthedocs.io/en/latest/v3/core/)
* [Xarray Zarr encoding](https://docs.xarray.dev/en/stable/internals/zarr-encoding-spec.html)
* [Xarray fill handling](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.to_zarr.html#notes)
