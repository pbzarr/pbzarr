# Save a derived track

## Goal

Compute a result for the whole genome, such as the fraction of samples with depth 10 or more at each base, and save it in the store as a new track. Then read it again, export it, or use it in other work, with no new computation.

## Before you start

- A store in a directory. `pbzarr.write` cannot add to a zip store.
- Enough disk space for the new track. Most derived tracks compress well.

## Steps

### 1. Compute the result as a plan

```python
import pbzarr

ds = pbzarr.open("cohort.pbz")
over_10 = (ds["depth"] >= 10).mean("sample").astype("float32")
```

`over_10` is lazy. It covers the whole genome on `position`, and it has the `reference` coordinate of the store.

### 2. Write it

```python
ds = pbzarr.write(ds, {"over_10": over_10})
list(ds.data_vars)       # ['depth', 'mappable', 'over_10', 'signal']
```

`pbzarr.write` computes the result one shard at a time and writes each shard. The whole result is never in memory. It returns the store opened again. The Dataset that you gave it does not change, so use the returned Dataset.

The first argument can also be the store path: `pbzarr.write("cohort.pbz", {...})`.

### 3. Write many tracks in one call

```python
ds = pbzarr.write(ds, {
    "depth_mean": ds["depth"].mean("sample").astype("float32"),
    "depth_max": ds["depth"].max("sample"),
})
```

### 4. Write a value for each sample

A variable with no genomic dimension is also accepted, for example one value for each sample:

```python
ds = pbzarr.write(ds, {"mean_depth": ds["depth"].mean("position")})
```

## Rules

`pbzarr.write` refuses a variable that does not fit the store:

| Problem | Error |
|---|---|
| The name is already in the store | `ValueError: 'over_10' already exists in the store` |
| Part of `position`, for example from `pbzarr.region` | `ValueError: 'x' has 100 rows on 'position'; the reference has 320000` |
| An array from a store with another reference | `AlignmentError` |
| Sample labels that differ from the store | `ValueError` |

The write never changes or removes an existing track. To keep a new version of a result, write it under a new name.

The first dimension of a track must be `position` or `bin_N`, and it must cover the whole genome. A new dimension, for example `group`, needs a label coordinate.

## The layout of a written track

A new track on `position` gets the same chunks, shards, and compression as an import. One shard of 16 `int32` columns is 4 GiB in memory. If this is too much for your machine, give smaller shards in `encoding=`:

```python
ds = pbzarr.write(
    ds,
    {"depth_x2": ds["depth"] * 2},
    encoding={"depth_x2": {"chunks": (65_536, 1), "shards": (16_777_216, 4)}},
)
```

See [Storage layout](../concepts/storage-layout.md).

## Result

The new track is in the store, and other tools can read it:

```bash
pbz info cohort.pbz
pbz view cohort.pbz --track over_10 -r chr1:149990-152010
```
