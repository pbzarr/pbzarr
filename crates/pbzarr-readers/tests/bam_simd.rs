//! Pins `bump_depth` (the no-gate depth Match-block inner loop,
//! `crates/pbzarr-readers/src/bam/walk.rs`) against a scalar reference across
//! the SIMD-relevant length boundaries (0, 1, one-below/at/one-above a
//! 16-lane chunk, two chunks, and a length that isn't a multiple of 16) and
//! the extremes of `min_bq`.

use pbzarr_readers::bam::walk::bump_depth;

fn scalar_reference(depth: &mut [i32], quals: &[u8], min_bq: u8) {
    for (d, &q) in depth.iter_mut().zip(quals.iter()) {
        if q >= min_bq {
            *d += 1;
        }
    }
}

#[test]
fn bump_depth_matches_scalar_reference_at_simd_boundaries() {
    for len in [0usize, 1, 15, 16, 17, 31, 32, 33, 64] {
        // Deterministic, non-uniform quals covering the full u8 range so both
        // the passing and gated lanes of the mask are exercised.
        let quals: Vec<u8> = (0..len).map(|i| ((i * 91) % 251) as u8).collect();
        for min_bq in [0u8, 30, 255] {
            // Non-zero, non-uniform starting depth so the test would catch an
            // implementation that overwrites instead of accumulating.
            let mut got = vec![3i32; len];
            let mut expected = got.clone();

            bump_depth(&mut got, &quals, min_bq);
            scalar_reference(&mut expected, &quals, min_bq);

            assert_eq!(got, expected, "len={len} min_bq={min_bq} quals={quals:?}");
        }
    }
}
