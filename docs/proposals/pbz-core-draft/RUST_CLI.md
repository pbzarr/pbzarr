# Rust and CLI decisions for the PBZ core draft

Status: accepted design intent, 2026-09-30. This page describes the future
[core convention](README.md), not the implemented format 0.4 behavior.
It records five implementation boundaries for Rust and the CLI.
The [conformance cases](CONFORMANCE.md) supply concrete examples and validation rules.

## Format transition

The root Dataset design requires a new format version. The token
`core-draft-1` remains provisional. This proposal assigns no numeric release
version, and package versions remain separate from format versions.

Conversion from format 0.4 is outside the initial work. This decision does not
promise or prohibit legacy reading. Legacy reading needs a separate support
decision. A reader must identify the format before it interprets the store.
It must report an unsupported format explicitly.

The default entry points and release roadmap still use format 0.4. The additive
`pbzarr::dataset::Dataset` reader now implements draft-format inspection,
validation, and typed positional reads; see the [validation record](VALIDATION.md).
Writing and imports have a [milestone-B plan](../../superpowers/plans/2026-10-01-pbz-core-writer.md)
and [creation/publication contract](WRITING.md). They are not implemented yet;
CLI integration remains milestone C.

## 1. Generic Rust storage with existing array infrastructure

Rust storage supports the shapes that the core convention permits:

* Genomic variables with `position` or `bin_N` and any number of labeled
  non-genomic dimensions.
* Non-genomic variables, including vectors and multidimensional results.
* Scalar variables with no dimensions.

Rust exposes dimension names, shapes, coordinates, and variable metadata.
Selection uses named dimensions rather than a required column role.
The core convention owns reference geometry and coordinate identity.

The implementation builds on the existing `zarrs` and `ndarray` infrastructure.
Those dependencies already provide array storage and dynamically ranked arrays.
The new work adds the root variable catalog, shared coordinates, validation,
and selection rules. It also removes rank assumptions from general storage operations.

Generic storage support does not require every importer or CLI command to
support every valid shape. A narrower command capability does not make the
underlying store invalid.

## 2. Explicit import routing and shared coordinates

Initial importers continue to produce one-dimensional or two-dimensional
genomic variables. Routing explicitly maps source files or source fields to
tracks and, where requested, one non-genomic dimension.
Generic Rust storage support does not require an n-dimensional import engine.

An explicitly requested column dimension remains present for a single source.
For example, a requested `sample` dimension with one label produces
`(position, sample)` with a sample extent of one.

The importer retains `column_dim` as a convenience parameter. It names the
output dimension. The store does not require a column-role metadata field.

For a new store, the importer establishes the root reference from an explicit
reference input or validated source headers. Existing stores retain their root
reference. Import preflight validates the ordered names and lengths before mutation.

An import that uses an existing dimension must supply the same labels in the
same order. Mismatched references or labels fail before mutation. Import does
not silently reorder, union, broadcast, or extend shared coordinates.
The [writing policies](README.md#writing-policies-supported-by-this-schema)
define the separate persistence operations.

## 3. Named-dimension CLI selection with a bounded initial output model

CLI selectors identify their dimensions by name. An unnamed column selector
cannot ambiguously select among several non-genomic dimensions.
Final flag spellings and compatibility aliases remain implementation choices.

Initially, `view` and `stat` operate on a genomic variable with at most one
varying non-genomic dimension after selection. A varying dimension has more
than one selected label. Other non-genomic dimensions must select one label.
The output must retain the identities of those selected labels.

For example, a variable with dimensions `(position, sample, threshold)` needs
a threshold selection before a command can show multiple sample columns.
Selecting one sample instead permits threshold columns.
This restriction applies to command execution, not store opening or inspection.

`stat` reduces the genomic axis. It does not implicitly reduce samples,
thresholds, or other dimensions. Commands report unsupported remaining shapes
instead of flattening axes or choosing a reduction for the user.

Scalars and non-genomic variables remain valid and inspectable. The initial
genomic `view` and `stat` commands do not assign them artificial genomic bounds.

## 4. Binned selection and statistics over stored observations

Region selection on `bin_N` includes complete bins that intersect the query.
Each bin retains its actual contig-local bounds. Its final bound stops at the
contig length. Query edges do not change a bin's stored value or bounds.

For a 2200-base contig and `bin_1000`, the final bin spans `[2000, 2200)`.
A query for `[2100, 2150)` selects that bin with its full `[2000, 2200)` bounds.

Statistics reduce stored observations with equal weight. A mean over selected
bins gives each bin one weight, including a shorter terminal bin.
A histogram counts stored observations, so its counts refer to bins rather
than bases for a binned variable.

Commands do not reconstruct base values or infer an aggregation meaning.
An independently supplied binned variable need not contain means, sums, or
values derived from another track.
The [core geometry rules](README.md#dimensions-and-genomic-geometry) define the
bin layout.

## 5. Encodings, declared missingness, and partial reader support

The initial Rust value types include Boolean, signed and unsigned integers
of 8, 16, 32, and 64 bits, and `float32` and `float64`.
Coordinate support includes string and numeric labels under the core rules.
Support for 64-bit integers lets ordinary integer reduction results retain
their stored dtype without an implicit downcast.

Rust and Python must document their supported Zarr encodings. Explicit chunk,
shard, and codec choices remain authoritative. Implementations reject unsupported
choices rather than silently replacing their layouts or codecs.
The conformance cases cover the supported combinations.

The [value rules](README.md#values-and-user-metadata) distinguish storage fill
from declared missingness. Zero and false retain their ordinary meaning unless
the variable explicitly declares them missing. CLI output and reductions honor
declared `_FillValue` semantics. A storage fill alone does not exclude values
from a reduction. Any raw-value API must clearly distinguish encoded values
from values with missingness applied.
The [initial encoding profile](CONFORMANCE.md#values-and-encodings) separates
ordinary logical Booleans from integer sentinel masking. It does not claim
support for nullable Boolean encodings that native Xarray currently decodes
inconsistently.

An unsupported encoding on an unrelated variable does not block supported
variables. The catalog retains that variable's identity and the available
metadata, and reports its unsupported capability. Selecting its values produces
an explicit error. A reader must not silently omit it from the Dataset.

Unsupported reference encodings block genomic interpretation that requires
that reference. Unsupported coordinate encodings block operations that require
those coordinates. Inspection of available metadata does not establish that
the unavailable coordinate identity is valid.

## Scope

These decisions define storage and command behavior without selecting final
Rust type names, method names, or API aliases. The core proposal owns the schema.
This page adds no pyramid, provenance, or transaction contract.

The initial interoperability target is a store that native Xarray and the Rust
reader interpret consistently within their declared capabilities.
The conformance cases must distinguish an invalid store from a valid store
that a particular reader or command does not fully support.
