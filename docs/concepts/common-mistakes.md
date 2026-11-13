# Common mistakes

Most xarray operations on a store give correct results. A few do not know where one contig ends and the next one starts. This page lists the operations that give a wrong result or an error, and what to do in their place.

The examples use this store:

```python
import pbzarr
import xarray as xr

ds = pbzarr.open("cohort.pbz")
depth = ds["depth"]          # (position, sample)
```

## Select rows of `position` directly

**The mistake:** `depth[:1000]`, `depth.isel(position=slice(0, 1000))`, and `depth.sel(position=5)` select rows of the flat `position` axis. The rows have no contig name, and a slice can cross from one contig into the next. The pbzarr functions refuse a slice:

```python
pbzarr.reduce_regions(depth[:1000], ["chr1:0-500"], "mean")
# ValueError: 'position' has 1000 rows but the reference has 320000. This is part
# of the genome. Use the whole array, or region() for one locus.
```

`depth.sel(position=5)` gives no error: it reads row 5. Do not use `sel` on `position` of a whole-genome array.

**Do this:** use `pbzarr.region` for one locus, and `pbzarr.reduce_regions` for many intervals. Both take the whole track.

```python
locus = pbzarr.region(depth, "chr1:0-1000")
means = pbzarr.reduce_regions(depth, ["chr1:0-500", "chr1:500-1000"], "mean")
```

On the result of `pbzarr.region`, `sel` with labels is correct: `locus.sel(position="chr1:500")`.

## Reorder `position`

**The mistake:** `depth.roll(position=500)`, `depth.sortby(...)` along `position`, or `isel(position=order)` with all rows keep the full length. The pbzarr functions cannot see the change. They map the rows to the wrong bases, with no error.

**Do this:** do not reorder `position`. Reorder the result in its place, for example the rows of `pbzarr.reduce_regions`.

## Use `coarsen` or `rolling` across contigs

**The mistake:** `depth.coarsen(position=1000, boundary="trim").mean()` and `depth.rolling(position=101, center=True).mean()` work on the flat axis. The last bin of `chr1`, and the windows near its end, also use rows from the start of `chr2`. No warning appears.

**Do this:** for bins, make one interval for each bin, contig by contig, and use `pbzarr.reduce_regions`. The last bin of each contig ends at the contig end:

```python
size = 10_000
ref = pbzarr.Reference.from_xarray(depth)
bins = [
    (contig, start, min(start + size, length))
    for contig, length in ref.items()
    for start in range(0, length, size)
]
bin_means = pbzarr.reduce_regions(depth, bins, "mean")
```

For a window, select one contig with `pbzarr.region`, then use `rolling` on it:

```python
smooth_chr1 = pbzarr.region(depth, "chr1").rolling(position=101, center=True).mean()
```

## Lose the `reference` coordinate

**The mistake:** most xarray operations keep the `reference` coordinate. `quantile` removes it. `coarsen` keeps its value but removes its index. An array that you make yourself, or that you open with `xr.open_zarr`, does not have it. The pbzarr functions then refuse the array:

```python
q90 = depth.quantile(0.9, dim="sample")
pbzarr.region(q90, "chr1:0-10")
# ValueError: this object has no pbzarr reference; open it with pbzarr, or pass reference=
```

**Do this:** take the reference from the source array, and give it as `reference=`:

```python
ref = pbzarr.Reference.from_xarray(depth)
locus = pbzarr.region(q90, "chr1:0-10", reference=ref)
```

## Assign an array from another genome

**The mistake:** two stores can have the same total length with different contigs or a different contig order. Arithmetic between their arrays raises `AlignmentError`, which is correct. But `ds["x"] = other` and `ds.assign(x=other)` give no error. xarray drops the `reference` of the new array, and the array takes the reference of `ds`. `pbzarr.write` then saves the rows in the wrong place.

```python
other = pbzarr.open("other.pbz")["depth"]   # chr2 before chr1, same total length
ds["depth_other"] = other                    # no error, and the rows are wrong
```

**Do this:** use `xr.merge`, which raises `AlignmentError` when the references are different:

```python
merged = xr.merge([ds, other.to_dataset(name="depth_other")])
# AlignmentError: the objects use different references: ...
```

Or compare the two references before you assign. The order of the contigs is part of the reference:

```python
same = list(pbzarr.Reference.from_xarray(other).items()) == list(ref.items())
```

## Select samples before `reduce_regions`

**The mistake:** this is slow, but correct. `pbzarr.reduce_regions` reads the store directly when it gets a track as `pbzarr.open` gave it. After `sel(sample=...)`, `isel(sample=...)`, or `transpose`, it reads through the Dask graph, which does more work.

**Do this:** reduce the whole track first, then select samples on the small result:

```python
peaks = ["chr1:10000-20000", "chr2:25000-40000"]
s3 = pbzarr.reduce_regions(depth, peaks, "mean").sel(sample="s3")
```

For one locus, the order does not matter. `pbzarr.region` is a plain selection.

## Compute a whole-genome array

**The mistake:** `depth.compute()`, `depth.values`, or `depth.mean("sample").compute()` loads one value for each base into memory. For a human genome, one `int32` column is about 12 GB.

**Do this:** reduce before you compute, for example one value for each region or each sample. To keep a whole-genome result, save it in the store with `pbzarr.write`. See [Save a derived track](../cookbook/derived-track.md).
