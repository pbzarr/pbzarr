//! The only legal home for coordinate math between pbzarr's internal
//! convention (0-based, half-open, `u64`) and noodles' convention (1-based,
//! inclusive, `Position`). Every other module in this crate must go through
//! these two functions rather than hand-rolling the +1/-1; `clippy.toml`
//! bans `noodles_core::Position`/`noodles_core::region::Interval` outside
//! this file to keep it that way.
//!
//! Numerically, a half-open `[start, end)` and an inclusive `[start, end]`
//! cover the same positions when the inclusive `end` is one less than the
//! half-open `end` — so converting 0-based half-open to 1-based inclusive
//! only shifts `start` (0-based -> 1-based is `+1`); `end` is unchanged,
//! because the half-open `end`'s "one past the last position" already equals
//! the 1-based index of that same last position.
//!
//! `clippy::disallowed_types` is allowed for the whole module (not just the
//! two functions below) because the `use` declarations themselves trip the
//! lint.
#![allow(clippy::disallowed_types)]

use noodles_core::Position;
use noodles_core::region::Interval;

use pbzarr::io::{ReaderError, Result};

/// Converts a 0-based half-open `[start, end)` query range to a noodles
/// `Interval` (1-based, inclusive). Returns `Ok(None)` for an empty or
/// inverted range (`end <= start`) rather than erroring, since `Position`
/// has no representation for an empty interval (its minimum is 1, not 0).
/// Errors only on genuine overflow: a position beyond what `Position`
/// represents.
pub(crate) fn noodles_query_interval(start: u64, end: u64) -> Result<Option<Interval>> {
    if end <= start {
        return Ok(None);
    }
    let q_start = Position::try_from(start as usize + 1)
        .map_err(|e| ReaderError::Other(anyhow::anyhow!("bad start {start}: {e}")))?;
    let q_end = Position::try_from(end as usize)
        .map_err(|e| ReaderError::Other(anyhow::anyhow!("bad end {end}: {e}")))?;
    Ok(Some(Interval::from(q_start..=q_end)))
}

/// Converts a noodles 1-based `Position` back to a 0-based coordinate (the
/// decode direction: `usize::from(p) - 1`).
pub(crate) fn pos_to_zero_based(p: Position) -> u64 {
    (usize::from(p) - 1) as u64
}
