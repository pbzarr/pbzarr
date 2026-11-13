# Compare samples

## Goal

Find regions where one sample differs from the rest of the cohort, for example a deletion or a duplication. Compare two samples base by base.

## Before you start

- A store with a `depth` track over `(position, sample)`.
- For the gene table: a BED file of genes. See [Depth in regions](depth-in-regions.md) for the file in this example.

## Steps

### 1. Correct for the depth of each sample

Samples are sequenced to different depths, so compare each sample with its own mean first:

```python
import pandas as pd
import pbzarr

ds = pbzarr.open("cohort.pbz")
depth = ds["depth"]

sample_mean = depth.mean("position").compute()
sample_mean.to_pandas().round(2)
```

```text
sample
s1    19.88
s2    24.84
s3    29.58
s4    34.77
s5    40.26
s6    29.81
```

`sample_mean` has one value for each sample. When you divide by it, xarray matches the values to the `sample` column.

### 2. Compare each sample with the cohort in each gene

Get the mean depth of each gene and sample, divide by the mean of the sample, then divide by the median of the cohort:

```python
genes = pd.read_csv(
    "genes.bed", sep="\t", header=None, names=["chrom", "start", "end", "name"]
)
gene_means = pbzarr.reduce_regions(depth, genes, "mean")
relative = gene_means / sample_mean
ratio = (relative / relative.median("sample")).compute()
ratio.to_pandas().set_axis(genes["name"]).round(2)
```

```text
sample   s1   s2    s3   s4    s5   s6
name
geneA   1.0  1.0  1.01  1.0  0.99  1.0
geneB   1.0  1.0  0.80  1.0  0.99  1.0
geneC   1.0  1.0  1.01  1.0  0.99  1.0
geneD   1.0  1.0  1.01  1.0  1.25  1.0
geneE   1.0  1.0  1.01  1.0  0.99  1.0
```

A ratio near 1 is normal. `s3` has less depth in `geneB`, and `s5` has more depth in `geneD`. The ratios are not 0.5 and 2, because the change covers only part of each gene.

### 3. Compare at each base

The same steps work at each base. The result is a whole-genome plan, so compute it for one region only, or save it with `pbzarr.write`:

```python
normalized = depth / sample_mean
ratio_per_base = normalized / normalized.median("sample")

pbzarr.reduce_regions(ratio_per_base, ["chr1:60000-65000"], "mean").compute()
```

The mean ratio of `s3` in `chr1:60000-65000` is 0.53: half the depth of the cohort.

### 4. Compare two samples

Select two columns and subtract. The result is over `position` only:

```python
diff = depth.sel(sample="s5") - depth.sel(sample="s1")
pbzarr.region(diff, "chr2:30000-30010").values
```

`depth.sel(sample=...)` selects a column by its label, which is correct on a whole-genome track. It is `position` that must not be selected directly. See [Common mistakes](../concepts/common-mistakes.md).

## Result

- `ratio`: a table of genes by samples. Values far from 1 mark a gain or a loss in one sample.
- `ratio_per_base`: the same ratio at each base, as a lazy array over `(position, sample)`. Save it with `pbzarr.write` to use it again. See [Save a derived track](derived-track.md).
