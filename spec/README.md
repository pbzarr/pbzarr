# PBZ format specification, version 0.5

Status: draft. Format 0.5 is the draft that format 1.0 will come from. Some
rules can still change before 1.0.

A PBZ store holds per-base and binned genomic data in Zarr v3. It opens as one
Xarray Dataset with `xarray.open_zarr`, with no PBZ code.

The words MUST, MUST NOT, SHOULD and MAY have the meanings of RFC 2119.

## Example

    example.pbz/
      zarr.json                                             root group
      contig_id       (contig)                              ["chr1", "chr2"]
      contig_length   (contig)                              [1500, 2200]
      sample          (sample)                              ["s1", "s2", "s3"]
      threshold       (threshold)                           [5, 10]
      depth           (position, sample)                    shape [3700, 3]
      callable        (position, sample, threshold)         shape [3700, 3, 2]
      coverage        (bin_1000, sample)                    shape [5, 3]
      mean_depth      (sample)                              shape [3]

Each entry other than `zarr.json` is a Zarr v3 array that is a direct child
of the root. The names `contig_id`, `contig_length` and `contig` are fixed. The
other names are examples.

## Root group

The root is a Zarr v3 group. Its attributes MUST contain:

| Attribute | Value |
| --- | --- |
| `zarr_conventions` | A list that contains exactly one entry with `"uuid": "d5bd7bea-0388-4e0a-9844-707c034e9add"`. |
| `pbz:version` | The string `"0.5"`. |

The UUID identifies PBZ. The `name` field of the entry is for display only.

The root MAY have other attributes. The Xarray `coordinates` attribute has no
PBZ meaning. Writers MAY write it. Readers MUST NOT need it.

All arrays are direct children of the root. Readers ignore child groups.

## Reference

The reference is the ordered list of contigs. It is stored in root arrays on
the dimension `contig`:

| Array | Dtype | Rule |
| --- | --- | --- |
| `contig_id` | string | MUST exist. Names are unique and not empty. |
| `contig_length` | integer | MUST exist. One length for each name, 0 or more. |
| `contig_md5` | string | MAY exist. One checksum for each name. |

Index `i` of each array refers to the same contig. The order of the arrays is
the order of the contigs on the genomic dimensions. Each length, and the sum of
all lengths, MUST fit in a signed 64-bit integer.

The reference does not change after a store is made.

## Dimensions and genomic geometry

Each array that is not a scalar MUST have `dimension_names`, with one name for
each axis. The names are not empty and are all different. All arrays that use
a dimension name MUST agree on its length.

There are two kinds of genomic dimension:

| Name | Length | Rows |
| --- | --- | --- |
| `position` | sum of `L_i` | Each base of each contig, in reference order. |
| `bin_N` | sum of `ceil(L_i / N)` | The bins of N bases of each contig, in reference order. |

`L_i` is the length of contig `i`. `N` is a positive integer with no leading
zeros (`bin_1000`, not `bin_01000`). Other names that start with `bin_` are
not allowed.

Coordinates are 0-based and half-open. Bins start again at 0 in each contig.
Bin `k` of a contig covers `[k * N, min((k + 1) * N, L_i))`, so the last bin
of a contig can be short. A reader finds the contig of a row from the prefix
sums of `contig_length`.

A track is an array with a genomic dimension. A track MUST have exactly one
genomic dimension, and it MUST be the first dimension. Any number of other
dimensions can follow.

A store MUST NOT have an array named `position` or `bin_N`. The row
coordinates come from the reference only.

## Labels and other arrays

Each dimension that is not genomic and is not `contig` MUST have a 1D array
with the same name and the same length. Its values are the labels of that
dimension. Labels are strings or numbers, unique and not missing. For example,
`sample` holds the sample names.

Other root arrays without a genomic dimension, such as `mean_depth(sample)` or
a scalar, are allowed. Their values can repeat.

All arrays share one namespace, so a track cannot have the name of a
dimension or a label array.

## Values and user metadata

Zarr owns the dtype, the shape, the chunks, the shards and the codecs of each
array. PBZ adds no rules for them. A reader that cannot decode an array MUST
report that array as unsupported. It MUST NOT read it as other data.

The Zarr `fill_value` is the value of positions with no data. The optional
array attribute `_FillValue` declares a value that means "missing". It has the
Xarray meaning:

- For a float array, `_FillValue` is the base64 text of the little-endian
  8-byte float, the form that Xarray writes. A JSON number is not allowed.
- For an integer array, `_FillValue` is a JSON integer.
- A bool array has no `_FillValue`.

Without `_FillValue`, 0 and false are ordinary values. The pbzarr writers use
NaN for floats and the largest value of the dtype for integers.

Arrays MAY have other attributes, for example `description` or `units`. They
have no PBZ meaning.

## Consolidated metadata

A writer SHOULD put consolidated metadata in the root `zarr.json` when it
finishes a write. This is the inline form of zarr-python v3: the attribute
`consolidated_metadata` holds the `zarr.json` of each child, by path. The
arrays are always the authority. A writer that changes the arrays MUST update
or remove the consolidated metadata.

## Zip stores

A store can be a single zip file. The zip file holds the complete store at its
root, so the root `zarr.json` is the entry `zarr.json`. Writers SHOULD use
Zip64, store each entry with no zip compression, and include consolidated
metadata. A zip store is read-only.

## Versions

A reader MUST check the UUID and `pbz:version` before it reads any array:

- A missing UUID, or a `pbz:version` that is missing or not a string, is an
  invalid store.
- A version that the reader does not know is an unsupported version. The
  reader MUST NOT read it as format 0.5.
- Format 0.4 used the UUID `b7e3c1a2-5f4d-4e8a-9c1b-2d3e4f5a6b7c`, with one
  group for each track. It is not described here. Readers report it as a
  legacy format.

The format version is not the version of the pbzarr packages.

## Validation

- [root-attributes.schema.json](root-attributes.schema.json) is a JSON Schema
  for the root attributes.
- [CONFORMANCE.md](CONFORMANCE.md) has numbered rules and the valid and
  invalid example stores.
- [examples.py](examples.py) makes the example stores.
- `pbz validate STORE` checks a store against these rules.
