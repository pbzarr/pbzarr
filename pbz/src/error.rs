//! CLI-side error helpers. We use `color-eyre::Result` end-to-end and wrap
//! library `PbzError`s with surrounding context via `wrap_err_with`.

use color_eyre::Result;

pub fn install() -> Result<()> {
    color_eyre::install()
}
