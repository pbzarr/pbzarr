# pbzarr

pbzarr is a single store for genomic sequencing data that allows for joint computation and storage across data types (tracks), genomic position (`position`), and samples (column dim). For example, one store can hold coverage and DNase signal for a cohort, together with repeat annotations for stratification.

A store is a Zarr v3 group: a directory, a ZIP file, or a location in cloud object storage. It opens as one [xarray](https://docs.xarray.dev/) Dataset. See [Data model](concepts/data-model.md) for the details.

## Install

The Python package needs Python 3.12 or later:

```bash
pip install pbzarr
```

The `pbz` command-line tool installs into `~/.cargo/bin`, on Linux x86_64 or macOS on Apple Silicon:

```bash
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/pbzarr/pbzarr/releases/latest/download/pbz-installer.sh | sh
```

See [Install](get-started/install.md) for other systems and options.

## Example

=== "Python"

    ```python
    import pbzarr

    ds = pbzarr.open("cohort.pbz")
    window = pbzarr.region(ds["depth"], "chr1:100000-101000")
    print(window.mean("position").compute())
    ```

=== "CLI"

    ```bash
    pbz stat cohort.pbz --track depth -s mean -r chr1:100000-101000
    ```

## Where to go next

- [Install](get-started/install.md) the Python package and the `pbz` command.
- [Concepts](concepts/data-model.md): the data model, storage layout, and common mistakes.
- [Cookbook](cookbook/index.md): recipes for common tasks.
- [CLI Reference](reference/cli.md)
- [Python API](reference/python.md)
- [Store Format](reference/format.md)
