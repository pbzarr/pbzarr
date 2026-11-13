# pbzarr

pbzarr stores genomic measurements at base-pair resolution, such as read depth, signal and masks. It keeps them in two-dimensional arrays: one dimension for genomic position, and one for samples or other labels. You can then compute along either dimension, or both.

pbzarr uses [Zarr](https://zarr.dev/) to store compressed arrays. A pbzarr store opens as one [xarray](https://docs.xarray.dev/) Dataset, and [Dask](https://docs.dask.org/) runs the computations in parallel.

```python
import pbzarr

pbzarr.import_d4(
    "depth.pbz", "depth",
    [("brain.d4", "brain"), ("liver.d4", "liver")],
    dim="sample",
)

ds = pbzarr.open("depth.pbz")
window = pbzarr.region(ds["depth"], "chr1:100000-101000")
print(window.mean("position").compute())
```

The same store works from the command line:

```bash
pbz stat depth.pbz --track depth -s mean -r chr1:100000-101000
```

## Where to go next

- [Install](get-started/install.md) the Python package and the `pbz` command.
- [Command-line reference](reference/cli.md): every `pbz` command and flag.
- [Python API](reference/python.md): every public function.
- [Store format](reference/format.md): what is in a store, for other tools.
- [Upgrading to 0.8](UPGRADING.md): changes from pbzarr 0.7.
