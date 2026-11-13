# Callable mask

## Goal

Make a bool track that is True where the depth of a sample is in a usable range, here 10 to 60. Also make a track with the fraction of samples that are callable at each base. Count the callable bases for each sample and each gene.

## Before you start

- A store with a `depth` track over `(position, sample)`.
- For the counts in genes: a BED file of genes. See [Depth in regions](depth-in-regions.md) for the file in this example.

## Steps

### 1. Make the masks

```python
import pbzarr

ds = pbzarr.open("cohort.pbz")
depth = ds["depth"]

callable_ = (depth >= 10) & (depth <= 60)                 # (position, sample)
callable_fraction = callable_.mean("sample").astype("float32")  # (position,)
```

Nothing is computed yet. Both arrays are plans for the whole genome. The name `callable_` has an underscore because `callable` is a Python built-in.

### 2. Save them in the store

```python
ds = pbzarr.write(ds, {
    "callable": callable_,
    "callable_fraction": callable_fraction,
})
```

`pbzarr.write` computes the masks block by block and saves them as new tracks. It returns the store opened again, so `ds["callable"]` now reads from the store. A bool mask compresses very well: in the example store, `callable` takes less than 10 KiB on disk.

### 3. Count callable bases

For each sample, over the whole genome:

```python
ds["callable"].sum("position").compute().to_pandas()
```

```text
sample
s1    316384
s2    317913
s3    317887
s4    317987
s5    313834
s6    317998
```

For each gene and sample:

```python
import pandas as pd

genes = pd.read_csv(
    "genes.bed", sep="\t", header=None, names=["chrom", "start", "end", "name"]
)
per_gene = pbzarr.reduce_regions(ds["callable"], genes, "sum").compute()
per_gene.to_pandas().set_axis(genes["name"])
```

```text
sample     s1     s2     s3     s4     s5     s6
name
geneA    9948   9997  10000  10000   9987  10000
geneB   11955  12000  11889  11999  11980  12000
geneC   12937  12998  12999  13000  12984  13000
geneD   14919  14996  15000  15000  11208  15000
geneE    8962   8997   9000   9000   8992   9000
```

`geneD` has fewer callable bases in `s5`, because `s5` has a duplication there with depth above 60.

## Variations

**Limits from the depth of each sample.** Samples with different mean depths need different limits. Compute the mean of each sample first, then compare. xarray matches the limits to the `sample` column:

```python
mean_depth = depth.mean("position").compute()     # one value per sample
callable_relative = (depth >= 0.5 * mean_depth) & (depth <= 2 * mean_depth)
```

**Only inside some regions.** `pbzarr.interval_mask` gives a bool array over `position` that is True inside the regions:

```python
in_genes = pbzarr.interval_mask(ds, genes)
callable_in_genes = ds["callable"] & in_genes
```

**Export the mask as text.**

```bash
pbz view cohort.pbz --track callable_fraction -o callable_fraction.bedgraph.gz
```

`pbz view` writes runs of equal values as one row, so a mask gives a short file. It prints a bool track as `1` and `0`. See [Export to bedGraph](export-bedgraph.md).

## Result

The store has two new tracks:

| Track | Dimensions | Type |
|---|---|---|
| `callable` | `(position, sample)` | bool |
| `callable_fraction` | `(position,)` | float32 |
