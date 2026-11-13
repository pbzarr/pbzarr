# Zarr, xarray, and Dask

pbzarr uses three libraries. Zarr stores the arrays. xarray gives the arrays names and labels. Dask runs the computations in parallel. This page tells you what pbzarr uses from each library, and where to learn more.

You do not need to know these libraries well to use pbzarr. But most work on a store is ordinary xarray and Dask work, so their docs are useful.

## Zarr: the storage

[Zarr](https://zarr.dev/) is a format for large arrays. It cuts each array into chunks, compresses each chunk, and keeps each chunk as one object: a file in a directory, an entry in a zip file, or an object in cloud storage. A reader gets only the chunks that it needs.

A pbzarr store is a Zarr v3 group. Each track, coordinate, and label array is one Zarr array in that group. pbzarr adds rules for genomic data: the reference arrays, the `position` and `bin_N` dimensions, and the column labels. The [store format](../reference/format.md) page gives a map of a store.

Any Zarr v3 tool can read a store. For example, plain xarray opens it with no pbzarr code:

```python
import xarray as xr

ds = xr.open_zarr("cohort.pbz")
```

Learn more:

- [Zarr documentation](https://zarr.readthedocs.io/)
- [Zarr v3 specification](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html)
- [Storage guide of zarr-python](https://zarr.readthedocs.io/en/stable/user-guide/storage.html): local, zip, and cloud stores

## xarray: names and labels

[xarray](https://docs.xarray.dev/) adds dimension names, coordinates, and labels to NumPy-like arrays. `pbzarr.open` returns an `xarray.Dataset`. Each track is an `xarray.DataArray` in it.

With xarray, you use dimension names in place of axis numbers:

```python
import pbzarr

ds = pbzarr.open("cohort.pbz")
depth = ds["depth"]

depth.sel(sample="s3")             # one column, by label
depth.mean("sample")               # mean over samples, at each base
(depth >= 10).sum("sample")        # samples with depth 10 or more, at each base
ds["signal"].where(ds["mappable"]) # NaN where the mask is False
```

Each of these keeps every row of `position` in place, so the result is correct for the whole genome. pbzarr adds functions only where plain xarray needs to know the genome: `pbzarr.region` for one locus, `pbzarr.reduce_regions` for one value per interval, and `pbzarr.interval_mask` for a mask from intervals.

Learn more:

- [Data structures](https://docs.xarray.dev/en/stable/user-guide/data-structures.html): `Dataset`, `DataArray`, dimensions, and coordinates
- [Indexing and selecting data](https://docs.xarray.dev/en/stable/user-guide/indexing.html): `sel` and `isel`
- [Computation](https://docs.xarray.dev/en/stable/user-guide/computation.html): arithmetic, reductions, and `where`

## Dask: lazy and parallel work

[Dask](https://docs.dask.org/) splits a large array into blocks and runs the work on the blocks in parallel. The arrays from `pbzarr.open` are Dask arrays. An operation on them makes a plan, but reads no data. The work runs when you ask for values:

```python
over_10 = (depth >= 10).sum("sample")   # a plan; no data is read
result = over_10.compute()              # reads the chunks and runs the plan
```

The result of `.compute()` is in memory. Reduce the data before you compute. For example, compute one locus, or one value per region, and not a whole-genome array of many samples.

`pbzarr.open` gives each track Dask blocks of about 128 MiB, aligned to the chunks on disk. You can give other blocks with `chunks=`, as in `xarray.open_zarr`. `chunks=None` gives arrays that are not Dask arrays: values are read when you index them.

By default, Dask uses threads on one machine. For a cluster, or for a progress dashboard, install `dask.distributed` (`pip install "dask[distributed]"`) and start a client before you compute:

```python
from dask.distributed import Client

client = Client()   # local cluster; print(client) shows the dashboard address
```

To save a whole-genome result in the store, use `pbzarr.write`. It writes the result block by block, so the whole result does not need to fit in memory. See [Save a derived track](../cookbook/derived-track.md).

Learn more:

- [Dask arrays](https://docs.dask.org/en/stable/array.html)
- [Parallel computing with Dask in xarray](https://docs.xarray.dev/en/stable/user-guide/dask.html)
- [Dask distributed](https://distributed.dask.org/): clusters and the dashboard
