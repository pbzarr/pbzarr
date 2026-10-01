"""Run with `pixi run dataset-track-poc` (add --interactive to explore).

THROWAWAY API EXPERIMENT. Uses synthetic data and temporary stores only.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import dask
import dask.array as da
from dask.callbacks import Callback
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_track_poc import Dataset, GenomicIndex, Track  # noqa: E402


def fixture():
    values = (da.arange(3700 * 3, chunks=1500).reshape((3700, 3)) % 21).astype("uint16")
    native = xr.Dataset(
        {
            "depth": (("position", "sample"), values),
            "external_signal": (("bin_1500", "sample"), np.arange(9).reshape(3, 3)),
            "sample_summary": ("sample", [1.5, 2.5, 3.5]),
            "run_number": xr.DataArray(1),
        },
        coords={
            "contig": np.asarray(["chr1", "chr2"], dtype=object),
            "contig_length": ("contig", [1500, 2200]),
            "sample": np.asarray(["sample_1", "sample_2", "sample_3"], dtype=object),
            "threshold": [5, 10],
        },
        attrs={
            "zarr_conventions": [{
                "uuid": "b7e3c1a2-5f4d-4e8a-9c1b-2d3e4f5a6b7c", "name": "pbz:",
            }],
            "pbz:version": "core-draft-1",
            "pbz:reference": {"names": "contig", "lengths": "contig_length"},
        },
    )
    return Dataset.from_xarray(native)


def show(label, value):
    print(f"\n{label}\n{value}")


def expected_error(label, operation):
    try:
        operation()
    except (ValueError, NotImplementedError, TypeError) as error:
        print(f"Rejected {label}: {error}")
    else:
        raise AssertionError(f"Expected rejection: {label}")


def run_demo():
    ds = fixture()
    show("THROWAWAY Dataset / Track API prototype", ds)

    executed = []
    with Callback(pretask=lambda key, *args: executed.append(key)):
        selected = ds.select(region="chr1:100-1000", sample=["sample_1", "sample_3"])
        counts = (selected["depth"] >= 5).sum("sample")
        by_threshold = (ds["depth"] >= ds.coords["threshold"]).sum("sample")
        bins = ds["depth"].coarsen(position=1000).mean()
        native = ds["depth"].to_xarray().sel(position="chr1:100-1000")
        native_counts = (native >= 5).sum("sample")
        returned = Track.from_xarray(native_counts)
        quantiles = Track.from_xarray(native.quantile([0.25, 0.75], dim="sample"))
    assert not executed
    assert dask.is_dask_collection(counts.to_xarray().data)
    show("Selected Dataset (base and bin axes selected separately)", selected)
    show("Wrapper callable count", counts)
    show("Native Xarray count, wrapped with no context keyword", returned)
    show("Native quantile, no Track.quantile implementation needed", quantiles)
    show("Contig-aware means", bins)
    print("Dask tasks during construction:", len(executed))

    raw = (np.arange(3700 * 3).reshape((3700, 3)) % 21).astype("uint16")
    np.testing.assert_array_equal(counts.compute().to_xarray(), (raw[100:1000, [0, 2]] >= 5).sum(1))
    assert by_threshold.dims == ("position", "threshold")
    np.testing.assert_array_equal(by_threshold.compute().to_xarray(), (raw[:, :, None] >= [5, 10]).sum(1))
    np.testing.assert_array_equal(returned.compute().to_xarray(), (raw[100:1000] >= 5).sum(1))
    expected_bins = np.stack([raw[a:b].mean(0) for a, b in
                              [(0, 1000), (1000, 1500), (1500, 2500), (2500, 3500), (3500, 3700)]])
    np.testing.assert_allclose(bins.compute().to_xarray(), expected_bins)
    assert bins.index.rows == range(5)
    assert ds["depth"].mean("position").index is None
    assert isinstance(ds["depth"].sum().compute().item(), int)

    reverse = Track.from_xarray(native_counts.isel(position=slice(None, None, -1)))
    assert reverse.index.rows == range(999, 99, -1)
    expected_error("different genomic order", lambda: reverse + returned)
    expected_error("different sample order", lambda: ds["depth"] + ds["depth"].select(
        sample=["sample_3", "sample_2", "sample_1"]))
    expected_error("lost native index", lambda: Track.from_xarray(native_counts.drop_indexes("position")))
    expected_error("regional coarsening", lambda: selected["depth"].coarsen(position=1000))
    expected_error("implicit in-memory scatter", lambda: ds.assign(regional=counts))

    # A lazy array this long proves coordinate construction does not need a
    # dense range. This does not benchmark reads, peak memory, or task counts.
    large = Track.from_array(
        da.empty((3_000_000_000,), chunks=1_000_000), dims="position",
        reference={"chr_large": 3_000_000_000},
    )
    assert isinstance(large.index, GenomicIndex)
    compact = large.index.to_pandas_index()
    assert compact.memory_usage() < 1024
    print("3-billion-row coordinate representation:", type(compact).__name__,
          f"({compact.memory_usage()} bytes reported by pandas)")

    with TemporaryDirectory(prefix="pbz-dataset-track-poc-") as tmp:
        path = Path(tmp) / "example.pbz"
        with Dataset.create(path, reference=ds.reference) as store:
            store.write({"depth": ds["depth"], "external_signal": ds["external_signal"]})
            store.write({
                "callable_count": (store["depth"] >= 5).sum("sample").with_attrs(notes="Depth >= 5"),
                "mean_depth_1000": bins,
                "count_by_threshold": by_threshold,
            })
            regional = (store["depth"].select(region="chr1:100-1000") >= 5).sum("sample")
            store.write({"regional_count": regional}, encoding={"regional_count": {"fill_value": 0}})
            expected_error("existing track name", lambda: store.write({"depth": store["depth"]}))
            show("Written store (new tracks, full reference)", store)

        with xr.open_zarr(path, consolidated=False) as plain:
            assert "position" not in plain.coords and "bin_1000" not in plain.coords
            assert "contig_length" in plain.coords
            assert "contig" not in plain.data_vars
            np.testing.assert_array_equal(plain.callable_count.compute(), (raw >= 5).sum(1))
            np.testing.assert_allclose(plain.mean_depth_1000.compute(), expected_bins)
            regional_values = plain.regional_count.compute().values
            np.testing.assert_array_equal(regional_values[100:1000], (raw[100:1000] >= 5).sum(1))
            assert not regional_values[:100].any() and not regional_values[1000:].any()
            assert plain.callable_count.attrs["notes"] == "Depth >= 5"
        with Dataset.open(path) as reopened:
            assert isinstance(reopened["depth"].index, GenomicIndex)
            computed = reopened["callable_count"].sum().compute()
            old_track = reopened["depth"]
        assert computed.item() == int((raw >= 5).sum())
        expected_error("write through a view", lambda: ds.select(region="chr1:0-10").write({"x": counts}))
        try:
            old_track.compute()
        except RuntimeError:
            print("Rejected computation through closed owner")
        else:
            raise AssertionError("Closed owner should be detected")

    print("\nPASS: lazy wrappers, native index retention, strict alignment, contig bins,")
    print("      full and regional Zarr writes, native reopening, and explicit lifetime.")
    print("Temporary stores removed. This is a prototype, not production PBZ support.")


def interactive():
    ds, current = fixture(), None
    while True:
        print("\033[2J\033[H", end="")
        print("THROWAWAY API playground — all state is synthetic and in memory")
        print(ds)
        if current is not None:
            print("\nCurrent result:", current)
        print("\n[1] select region/samples  [2] callable counts  [3] native quantile")
        print("[4] contig-aware bins      [5] compute preview [r] reset  [q] quit")
        action = input("> ").strip().lower()
        if action == "q":
            return
        try:
            if action == "1":
                current = ds["depth"].select(region="chr1:100-1000", sample=["sample_1", "sample_3"])
            elif action == "2":
                current = ((ds["depth"] if current is None else current) >= 5).sum("sample")
            elif action == "3":
                current = Track.from_xarray(ds["depth"].to_xarray().quantile([0.25, 0.75], dim="sample"))
            elif action == "4":
                current = ds["depth"].coarsen(position=1000).mean()
            elif action == "5" and current is not None:
                native = current.to_xarray()
                if current.index is not None:
                    native = native.isel({current.index.dim: slice(0, 5)})
                print(native.compute())
                input("Enter to continue")
            elif action == "r":
                ds, current = fixture(), None
        except (ValueError, TypeError, KeyError) as error:
            print(error)
            input("Enter to continue")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interactive", action="store_true")
    arguments = parser.parse_args()
    interactive() if arguments.interactive else run_demo()
