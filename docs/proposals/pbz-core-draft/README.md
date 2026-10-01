# PBZ core convention proposal

Status: agreed direction with a concrete conformance draft, updated 2026-10-01.
This proposes a new encoding; it does not describe the implemented 0.4 format
or change the release roadmap.
The identifier "core-draft-1" is provisional, not a published format version.
The new encoding will receive a new format version. Conversion from existing
stores is outside the initial scope; this does not decide legacy read support.

[Conformance examples and validation rules](CONFORMANCE.md) make the contract
reviewable. [Rust and CLI decisions](RUST_CLI.md) define the agreed implementation
scope. The [validation record](VALIDATION.md) distinguishes executed checks from
future Rust/Python conformance requirements.
The [implementation sequence and first Rust reader plan](../../superpowers/plans/2026-09-30-pbz-core-reader.md)
turn this contract into staged work. The additive Rust `pbzarr::dataset::Dataset`
reader implements milestone A of the development convention. Dataset creation
and existing 1D/2D imports implement milestone B; the validation record
distinguishes executed checks from work still pending.
The [Rust import migration plan](../../superpowers/plans/2026-10-01-pbz-core-writer.md)
covers milestone B and its [writing contract](WRITING.md).

## Model

A PBZ store is a Zarr v3 root group that opens as one ordinary Xarray Dataset.
Root data variables are tracks. Root coordinate arrays provide reference
sequences, dimension labels, and annotations. A track may have a genomic
dimension, or be a non-genomic result such as a sample-level mean or scalar.

The core adds three root attributes:

| Attribute | Purpose |
| --- | --- |
| zarr_conventions | Identify the PBZ convention using its UUID. |
| pbz:version | Identify this encoding independently of package versions. |
| pbz:reference | Point to root arrays containing ordered sequence names and lengths. |

No PBZ attributes are required on individual tracks. Native Zarr metadata owns
shape, dimension names, dtype, fill, chunks, and codecs. Native Xarray coordinate
encoding distinguishes coordinates from data variables. Descriptions, units,
notes, and user-defined JSON attributes are optional.

Pyramids, multiscales, automatic operation capture, provenance graphs, reference
sequence checksums, and transactional storage are outside this proposal.
Sample/column grouping uses ordinary Xarray coordinates and operations; its
workflows belong in API documentation and tutorials, not the core schema.

## Example store

    example.pbz/
      zarr.json
      contig                 (contig)                       ["chr1", "chr2"]
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

Each entry other than zarr.json is an ordinary Zarr array. Array names shown
here are examples. Reference pointers allow other reference-array names.
The genomic dimension names are the reserved names described below.

Both bin widths are independent user choices. Neither binned track needs a
base-resolution source, an aggregation declaration, or a relation to another track.

In this example, callable indicates whether each sample meets each depth
threshold. Summing it over sample produces callable_count. These names and
calculations illustrate ordinary tracks; the schema does not prescribe them.

[example-root.zarr.json](example-root.zarr.json) contains the example root
metadata. Its ordinary coordinates attribute is Xarray's encoding for an
auxiliary reference coordinate, not a PBZ field. Python writers should put arrays
in Dataset.coords and let Xarray generate coordinate metadata. Rust writers must
emit an equivalent encoding.

## Reference

The reference object contains names and lengths, each a path to a direct root
array. They must point to distinct arrays.

1. The names array is a one-dimensional dimension coordinate: its sole dimension
   has the same name as the array.
2. Names are unique, nonempty strings. Missing names are invalid.
3. The lengths array is an auxiliary coordinate on that same dimension, with
   one nonnegative integer length per name. Boolean and floating dtypes are not
   integer length encodings.
4. Their order defines concatenation order. Readers derive offsets by prefix sum.
5. Reference arrays are coordinates, not measurement tracks, after normal
   Xarray decoding.

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
Auxiliary coordinates that vary over genomic positions follow the same geometry.

Different bin_N dimensions need not divide one another. The spelling bin_1 is
allowed under the positive-width rule; it describes the same locations as
position but remains a different named dimension to Xarray. Writers must not
silently equate or rename those dimensions.

## Dimension labels and auxiliary coordinates

Every non-genomic dimension has a one-dimensional coordinate array with the same
name, matching length, and unique, nonmissing labels. Labels can be strings or
numbers; they are not restricted to sample identifiers.

Other auxiliary coordinates follow ordinary Xarray encoding and may contain
repeated values. PBZ assigns no application-specific meaning to them.

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

The [Python API proposal](PYTHON_API.md) sketches the Dataset/Track wrappers,
native Xarray integration, and write boundary supported by this convention.
The later [prototype](../../../python/prototypes/dataset_track_poc/README.md)
explores a runtime genomic index instead of a public geometry argument.
Wrapper-owned windowing is the intended interface; a separate native Xarray
coarsening accessor is optional. Runtime indexes are not serialized as genomic
coordinate arrays.

The proposed wrappers and accessors use the same stored Dataset:

| Workflow | Information already available |
| --- | --- |
| Open and list tracks | Convention marker, reference pointers, native coordinate/data-variable encoding. |
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
[Rust writing contract](WRITING.md#publication-and-failure) records the limits:
errors propagate normally, incomplete output may remain, and callers must reopen
after an error. Cleanup and uncertain-write reporting are deferred.

This draft does not promise atomic multi-array updates, concurrent writers, or
crash recovery.

## What the JSON Schema checks

[root-attributes.schema.json](root-attributes.schema.json) validates the root
attribute document. It proposes retaining the existing convention UUID while
using the explicit provisional encoding version to distinguish this design.
No schema URL is invented; a versioned schema/spec URL should be added when
published. Names in zarr_conventions are descriptive, not the identity check.

A store validator must additionally resolve the arrays and check the reference,
coordinate classification, shared dimensions, uniqueness, genomic extents, and
axis ordering above. JSON Schema alone cannot validate array contents or perform
the required cross-array arithmetic.

## References

The [validation record](VALIDATION.md) describes the native Xarray example checks.

* [Zarr conventions](https://github.com/zarr-conventions/zarr-conventions-spec)
* [Zarr v3 core](https://zarr-specs.readthedocs.io/en/latest/v3/core/)
* [Xarray Zarr encoding](https://docs.xarray.dev/en/stable/internals/zarr-encoding-spec.html)
* [Xarray fill handling](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.to_zarr.html#notes)
