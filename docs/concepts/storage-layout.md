# Storage layout

This page explains how a track is cut into chunks and shards, how the chunks are compressed, and when to change the defaults. The defaults are good for most stores. Read this page before you build a large store, or if you plan to add samples later.

## Chunks and shards

Zarr cuts each array into chunks and compresses each chunk on its own. A reader decodes whole chunks only. Small chunks make small reads fast, but a genome has billions of bases, so small chunks give a very large number of chunks.

pbzarr solves this with shards. A shard is one stored object that holds many inner chunks and an index of them. A reader can get one inner chunk from a shard and does not read the rest. The file system then has one file for each shard.

The default layout of a track that an import or `pbzarr.write` makes:

| | Positions | Columns |
|---|---|---|
| inner chunk | 65,536 | 1 |
| shard | 1,024 inner chunks (67,108,864 positions) | 16 |

A shard is never larger than the track. For a short genome or a narrow track, the shard is the size of the track, rounded up to whole inner chunks.

Each inner chunk holds one column. Thus a read of one sample decodes only that sample, also when the shard holds 16 samples.

## Compression

Each inner chunk is compressed with Blosc, with zstd at level 5 and byte shuffle. Depth and signal data compress well, because neighboring bases often have the same or a near value. A region with no data costs almost nothing. When no source covers a whole chunk, the import does not write the chunk, and a read gives the fill value.

## See the layout of a store

`pbz info` prints the layout of each array:

```bash
pbz info cohort.pbz
```

```text
variables:
  name           role        dtype    dims             shape     chunk    shard     codecs             fill   null  stored   objects
  contig_id      coordinate  string   contig           2         2        -         vlen-utf8          ""     none  20 B     1
  contig_length  coordinate  int64    contig           2         2        -         bytes              0      none  16 B     1
  depth          track       int32    position,sample  320000x6  65536x1  327680x6  bytes,blosc(5)     0      none  1.1 MiB  1
  mappable       track       bool     position         320000    65536    327680    bytes,blosc(5)     0      none  319 B    1
  sample         coordinate  string   sample           6         6        -         vlen-utf8,zstd(0)  ""     none  39 B     1
  signal         track       float32  position,sample  320000x6  65536x1  327680x6  bytes,blosc(5)     "NaN"  none  6.1 MiB  1
```

`chunk` is the inner chunk and `shard` is the shard. `stored` and `objects` give the size on disk and the number of stored objects. `pbz info --json` gives the same data as JSON.

## Change the layout

The import commands and functions take three settings:

| Command line | Python | Meaning | Default |
|---|---|---|---|
| `--chunk N` | `chunk=N` | positions in one inner chunk | 65,536 |
| `--shard N` | `shard=N` | inner chunks in one shard, along positions | 1,024 |
| `--shard-columns N` | `shard_columns=N` | columns in one shard | 16 |

For example:

=== "Python"

    ```python
    import pbzarr

    pbzarr.import_d4(
        "cohort.pbz", "depth", ["s1.d4", "s2.d4", "s3.d4"], shard_columns=1
    )
    ```

=== "CLI"

    ```bash
    pbz import d4 cohort.pbz s1.d4 s2.d4 s3.d4 --track depth --shard-columns 1
    ```

When to change them:

- **You plan to add samples later.** `pbz append` can grow only a track that has one column in each shard. Import with `--shard-columns 1`. See [Build a cohort store](../cookbook/build-a-cohort-store.md).
- **You read small regions of a very wide track over a network.** Each read gets at least one inner chunk for each column. A smaller `--chunk` makes these reads smaller.
- **Your file system limits the number of files.** Larger shards give fewer files. A [zip store](#zip-stores) gives one file.

`--codecs` (or `codecs=` in Python) replaces the whole layout with Zarr v3 codec metadata. It is for users who know the Zarr v3 codecs. It cannot be used with the three settings above.

A track keeps its layout. To change the layout of a track, import it again.

## Layout of a written track

`pbzarr.write` gives a new track on `position` the same layout as an import. The write works one shard at a time. A shard of 16 `int32` columns over 67,108,864 positions is 4 GiB in memory, so each Dask worker needs this much memory. For a wide track with less memory, give the layout in `encoding=`:

```python
import pbzarr

ds = pbzarr.open("cohort.pbz")
callable_ = (ds["depth"] >= 10) & (ds["depth"] <= 60)
ds = pbzarr.write(
    ds,
    {"callable": callable_},
    encoding={"callable": {"chunks": (65_536, 1), "shards": (16_777_216, 4)}},
)
```

Here a shard holds 16,777,216 positions and 4 columns.

When `encoding=` gives `chunks`, `shards`, or `compressors` for a track, xarray uses those values in place of the default layout.

## Blocks in memory

The Dask blocks of an open store are not the chunks on disk. `pbzarr.open` makes blocks of about 128 MiB. A block holds whole inner chunks and does not cross a shard edge. Thus each Dask task reads whole chunks. Give `chunks=` to `pbzarr.open` for other blocks.

## Zip stores

A zip store is a whole store in one zip file. It is easy to copy and share, and it is one file on a cluster file system. Make it with `--zip`, or give a store path that ends in `.zip`:

```bash
pbz import d4 cohort.pbz.zip s1.d4 s2.d4 s3.d4 --track depth
```

A zip store is written one time. You cannot add tracks or samples to it. `pbz view`, `pbz stat`, `pbz info`, `pbz validate`, and `pbzarr.open` read it. `pbzarr.write` and the Python imports refuse it.
