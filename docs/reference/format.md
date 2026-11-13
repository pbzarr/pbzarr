# Store format

A pbzarr store is one Zarr v3 group. All tracks and coordinates are arrays directly under the root. The store opens with `xarray.open_zarr`, with no pbzarr code.

```text
cohort.pbz/
├── zarr.json        root attributes and a copy of all array metadata
├── contig_id        contig names             (contig)
├── contig_length    contig lengths           (contig)
├── sample           column labels            (sample)
├── depth            a 2D track               (position, sample)
└── coverage         a binned track           (bin_1000, sample)
```

- The reference is the arrays `contig_id` and `contig_length`. Their order is the order of the contigs on the genomic dimensions.
- `position` has one row for each base of each contig, in reference order. A `bin_N` dimension has one row for each bin of N bases. Bins start again at 0 in each contig.
- Coordinates are 0-based and half-open, as in BED.
- Each other dimension, such as `sample`, has a label array with the same name.

The current format version is 0.5. It is the draft that format 1.0 will come from. pbzarr 0.8 refuses a format 0.4 store (made by pbzarr 0.7).

## Specification

The full rules are in the [format specification](https://github.com/pbzarr/pbzarr/blob/main/spec/README.md). The [conformance page](https://github.com/pbzarr/pbzarr/blob/main/spec/CONFORMANCE.md) lists the valid and invalid example stores.

To check a store against the rules:

```bash
pbz validate cohort.pbz
```
