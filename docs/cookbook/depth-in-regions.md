# Depth in regions

## Goal

Get one statistic, such as the mean depth, for each region and each sample. The regions come from a BED file, for example genes or targets.

## Before you start

- A store with a `depth` track over `(position, sample)`.
- A BED file of regions. This recipe uses `genes.bed`, with a name in the fourth column:

```text title="genes.bed"
chr1	10000	20000	geneA
chr1	58000	70000	geneB
chr1	140000	155000	geneC
chr2	25000	40000	geneD
chr2	90000	99000	geneE
```

BED coordinates are 0-based and half-open, as in pbzarr.

## Steps

=== "Python"

    Read the BED file with pandas, and give it to `pbzarr.reduce_regions`. The table needs the columns `chrom`, `start`, and `end`. Other columns are ignored.

    ```python
    import pandas as pd
    import pbzarr

    ds = pbzarr.open("cohort.pbz")
    genes = pd.read_csv(
        "genes.bed", sep="\t", header=None, names=["chrom", "start", "end", "name"]
    )

    gene_means = pbzarr.reduce_regions(ds["depth"], genes, "mean").compute()
    ```

    Row `i` of the result is row `i` of the table. Regions can overlap. To show the gene names, use them as the row labels:

    ```python
    gene_means.to_pandas().set_axis(genes["name"]).round(1)
    ```

    ```text
    sample    s1    s2    s3    s4    s5    s6
    name
    geneA   20.0  25.0  30.0  35.0  40.1  30.1
    geneB   20.0  25.0  23.7  35.0  40.0  30.1
    geneC   17.4  21.7  26.0  30.3  34.7  26.0
    geneD   20.0  25.0  30.1  35.0  50.7  30.0
    geneE   20.0  25.0  30.0  35.0  40.0  30.0
    ```

    For a long table with one row for each gene and sample, add the names as a coordinate and use `to_dataframe`:

    ```python
    table = (
        gene_means.assign_coords(gene=("region", genes["name"]))
        .to_dataframe()
        .reset_index()
    )
    ```

=== "CLI"

    ```bash
    pbz stat cohort.pbz --track depth -s mean -R genes.bed
    ```

    ```text
    #chrom	start	end	s1	s2	s3	s4	s5	s6
    chr1	10000	20000	19.9962	24.9766	30.0064	34.9678	40.0672	30.0585
    chr1	58000	70000	20.0325	24.9805	23.7199	34.9829	39.9717	30.0664
    chr1	140000	155000	17.3571	21.6798	25.9696	30.2793	34.6566	26.0133
    chr2	25000	40000	20.0264	25.0117	30.0751	34.9839	50.6733	29.9928
    chr2	90000	99000	19.9609	24.971	29.9886	34.9624	40.033	30.0061
    ```

    `-s` takes `mean`, `min`, `max`, `median`, or `hist`. `-c s1,s3` selects samples. `-r chr1:58000-70000` gives one region on the command line, and you can repeat it. `-o means.tsv` writes to a file. Without `-r` or `-R`, `pbz stat` gives one row for each contig.

## More statistics in Python

The third argument is the statistic: `mean`, `sum`, `min`, `max`, `count`, `std`, `var`, `median`, `quantile`, or `summit`.

```python
p10_p90 = pbzarr.reduce_regions(ds["depth"], genes, "quantile", q=[0.1, 0.9])
spread = pbzarr.reduce_regions(ds["depth"], genes, "std", ddof=1)
```

`count` gives the number of bases that are not null. By default, all statistics skip nulls (`NaN` in a float track). Give `skipna=False` to get `NaN` for a region that has a null.

Give a Dataset to compute the same statistic for many tracks in one call:

```python
both = pbzarr.reduce_regions(ds[["depth", "signal"]], genes, "median")
```

Regions can also be a list of strings, such as `["chr1:10000-20000", "chr2:25000-40000"]`, or a list of `(contig, start, end)` tuples.

## Result

`gene_means` is an `xarray.DataArray` over `(region, sample)`. Its coordinates `region_contig`, `region_start`, and `region_end` give the region of each row:

```text
<xarray.DataArray 'depth' (region: 5, sample: 6)> Size: 240B
array([[19.9962    , 24.9766    , 30.0064    , 34.9678    , 40.0672    ,
        30.0585    ],
...
Coordinates:
  * sample         (sample) object 48B 's1' 's2' 's3' 's4' 's5' 's6'
    region_contig  (region) object 40B 'chr1' 'chr1' 'chr1' 'chr2' 'chr2'
    region_start   (region) int64 40B 10000 58000 140000 25000 90000
    region_end     (region) int64 40B 20000 70000 155000 40000 99000
  * reference      <U8 32B 'cca77687'
```

The work reads only the rows of the regions. A call with thousands of regions runs in parallel with Dask. To select samples, select them on the result. See [Common mistakes](../concepts/common-mistakes.md#select-samples-before-reduce_regions).
