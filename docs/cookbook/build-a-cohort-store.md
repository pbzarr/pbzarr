# Build a cohort store

## Goal

Import the per-sample files of a cohort as the columns of one track, with a label for each sample. Add more samples later.

## Before you start

- One file for each sample, all made against the same reference: the same contig names, lengths, and order. This recipe uses d4 files. bigWig, BED, BAM, and CRAM files work in the same way.
- The `pbzarr` Python package or the `pbz` command. See [Install](../get-started/install.md).

## Steps

### 1. Import the files

Give the store path, the track name, and the files, each with a column label:

=== "Python"

    ```python
    import pbzarr

    pbzarr.import_d4(
        "cohort.pbz",
        "depth",
        [("s1.d4", "brain"), ("s2.d4", "liver")],
        dim="sample",
        shard_columns=1,
    )
    ```

    A source is a path, or a `(path, label)` tuple.

=== "CLI"

    ```bash
    pbz import d4 cohort.pbz s1.d4:brain s2.d4:liver --track depth --shard-columns 1
    ```

    A source is a path, or a path with `:LABEL` after it.

Without labels, each label is the file name without its extension: `s1.d4` gives `s1`. If one file has a label, all files must have labels.

A shard of one column (`shard_columns=1`, or `--shard-columns 1`) lets you add samples to the track later. Without it, a shard holds 16 samples, and `pbz append` refuses the track. See [Storage layout](../concepts/storage-layout.md).

### 2. Use a file list for many samples

For a large cohort, write the paths and labels in a tab-separated file, one sample on each line. Use it in place of the paths in step 1.

```text title="samples.tsv"
s1.d4	brain
s2.d4	liver
```

=== "Python"

    ```python
    import pbzarr

    with open("samples.tsv") as f:
        sources = [tuple(line.rstrip("\n").split("\t")) for line in f if line.strip()]

    pbzarr.import_d4("cohort.pbz", "depth", sources, dim="sample", shard_columns=1)
    ```

=== "CLI"

    ```bash
    pbz import d4 cohort.pbz --file-list samples.tsv --track depth --shard-columns 1
    ```

    A relative path in the list starts from the folder of the list file.

### 3. Add samples later

Only the command line can add samples:

```bash
pbz append d4 cohort.pbz s3.d4:heart --track depth
```

`append` adds columns to the track. Other tracks on the `sample` dimension also get the new column, with their fill value in it.

To start a cohort with one sample, give the dimension name (`dim="sample"`, or `--dim sample`). Without it, one file makes a track with no column axis, and `append` cannot grow it.

### 4. Add other tracks

An import into an existing store adds a new track. The files must match the reference of the store. A track name can be used one time only:

=== "Python"

    ```python
    pbzarr.import_bigwig(
        "cohort.pbz", "signal", [("s1.bw", "brain"), ("s2.bw", "liver")], fill=0
    )
    ```

=== "CLI"

    ```bash
    pbz import bigwig cohort.pbz s1.bw:brain s2.bw:liver --track signal --fill 0
    ```

Use a fill of 0 when a gap in a bigWig file means 0. By default a gap is missing (`NaN`). See [Fill and null](../concepts/data-model.md#fill-and-null).

A BED file has no contig lengths. To make a new store from BED files, give a `.fai` or chromosome-sizes file (`genome=`, or `--genome`). An import into an existing store takes the lengths from the store.

## Result

=== "Python"

    ```python
    ds = pbzarr.open("cohort.pbz")
    ds["sample"].values      # array(['brain', 'liver', 'heart'], dtype=object)
    ```

=== "CLI"

    ```bash
    pbz info cohort.pbz
    ```

    The output shows the reference, the `sample` dimension, and one row for each track.
