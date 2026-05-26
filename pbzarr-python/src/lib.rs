//! PyO3 bindings for pbzarr. Surfaces a single function (`import_d4`)
//! that wraps `pbzarr::ingest::import_d4`. Store/track creation and
//! reads are pure Python (see `python/pbzarr/`).

use pyo3::prelude::*;

#[pymodule]
fn _native(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
