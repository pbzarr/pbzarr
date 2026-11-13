# Export to bedGraph

## Goal

Write the values of a track as text, for a genome browser or another tool. Write a whole track, some regions, or some samples.

## Before you start

- A store with a track whose first dimension is `position`. `pbz view` does not read tracks on a `bin_N` dimension.

## Steps

### 1. Export a whole track

```bash
pbz view cohort.pbz --track depth -o depth.bedgraph.gz
```

The output has the columns `chrom`, `start`, and `end`, then one column for each sample. A run of bases with the same values in all columns is one row. A path that ends in `.gz` writes a BGZF file. Without `-o`, `pbz view` writes to stdout.

Use the command line for a whole track. It joins runs of equal values, and it does not hold the track in memory.

### 2. Export some regions or samples

=== "Python"

    `pbzarr.region` gives one locus. Select a sample, then convert the locus to pandas:

    ```python
    import pbzarr

    ds = pbzarr.open("cohort.pbz")
    locus = pbzarr.region(ds["depth"], "chr1:60000-60005").sel(sample="s3")
    locus.to_pandas()
    ```

    ```text
    position
    chr1:60000    12
    chr1:60001    13
    chr1:60002    18
    chr1:60003    14
    chr1:60004    16
    Name: depth, dtype: int32
    ```

    For a file with the contig and the contig-local position of each base:

    ```python
    table = locus.to_dataframe().reset_index()
    table[["region_contig", "pos", "sample", "depth"]].to_csv(
        "locus.tsv", sep="\t", index=False
    )
    ```

    ```text title="locus.tsv"
    region_contig	pos	sample	depth
    chr1	60000	s3	12
    chr1	60001	s3	13
    chr1	60002	s3	18
    chr1	60003	s3	14
    chr1	60004	s3	16
    ```

    This table has one row for each base, so use it for small regions only. Leave out `.sel(sample=...)` to get all samples.

=== "CLI"

    ```bash
    pbz view cohort.pbz --track depth -c s3 -r chr1:60000-60005
    ```

    ```text
    #chrom	start	end	s3
    chr1	60000	60001	12
    chr1	60001	60002	13
    chr1	60002	60003	18
    chr1	60003	60004	14
    chr1	60004	60005	16
    ```

    - `-r CHROM:START-END` selects one region. Repeat it for more regions. `-r chr1` selects all of `chr1`.
    - `-R regions.bed` reads the regions from a BED file.
    - `-c s1,s3` selects samples, by label.
    - `--no-header` leaves out the `#chrom` line.
    - `-p 4` prints floats with 4 significant digits. The default is 6.

    A null prints as `NaN`. A row where all columns are null is left out, so gaps in a bigWig import do not fill the file.

### 3. Export one sample for a genome browser

A bedGraph file for a browser has one value column and no header:

```bash
pbz view cohort.pbz --track depth -c s3 --no-header -o s3.bedgraph
```

To export a result that you computed in Python, save it in the store with `pbzarr.write`, then use `pbz view`. See [Save a derived track](derived-track.md).

## Result

A tab-separated text file with 0-based, half-open coordinates, as in BED and bedGraph.
