"""Combine N single-sample .pbz stores into a cohort along the sample axis.

Deliberately loose: it moves only the `values` arrays. A production stack must
also copy each track's `offsets`/`contigs` arrays and write the `perbase:*`
attribute block LAST (with column_dim="sample"), or store discovery skips the
group as incomplete. The native stack op (deferred, shared with append-samples)
will own that bookkeeping.
"""

import xarray as xr

single = {"s1": "s1.pbz", "s2": "s2.pbz"}  # label -> path
labels = list(single)

trees = {lbl: xr.open_datatree(p, engine="zarr") for lbl, p in single.items()}
tracks = list(trees[labels[0]].children)

for tk in tracks:
    da = xr.concat(
        [trees[lbl][tk]["values"].squeeze(drop=True) for lbl in labels],
        dim="sample",
    ).transpose("position", "sample").assign_coords(sample=("sample", labels))
    (
        da.to_dataset(name="values")
        .chunk({"position": 1_000_000, "sample": len(labels)})
        .to_zarr(f"cohort.pbz/{tk}", mode="w", zarr_format=3)
    )
