#!/usr/bin/env bash
# Build the wheel, install it (not editable) into a temporary venv that reuses
# the pixi env packages, then import a d4 fixture, open the store, and read a
# region. Run from the repo root: pixi run wheel-smoke
set -euo pipefail

cd "$(dirname "$0")/.."
root=$(pwd)

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

maturin build --release --out "$tmp/dist"
python -m venv --system-site-packages "$tmp/venv"
"$tmp/venv/bin/python" -m pip install --no-deps --no-index --quiet "$tmp"/dist/pbzarr-*.whl

cd "$tmp"
"$tmp/venv/bin/python" - "$root/fixtures/d4/banded_1k.d4" "$tmp/venv" <<'EOF'
import os
import sys

import numpy as np

import pbzarr

fixture, venv = sys.argv[1], sys.argv[2]
assert os.path.realpath(pbzarr.__file__).startswith(os.path.realpath(venv)), pbzarr.__file__
pbzarr.import_d4("smoke.pbz", "depth", fixture)
with pbzarr.open("smoke.pbz") as ds:
    view = pbzarr.region(ds["depth"], "chr1:100-110")
    np.testing.assert_array_equal(view.values, np.full(10, 11))
print(f"wheel smoke ok: pbzarr {pbzarr.__version__}")
EOF
