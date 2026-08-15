mod common;

use pbzarr::io::Dtype;
use pbzarr_readers::{InferRows, Source, infer_bed_dtypes_for_sources};
use tempfile::TempDir;

use common::{htslib_available, write_bed_bgzip_tabix};

#[test]
fn schema_dtype_inference_combines_observations_from_every_source() {
    assert!(
        htslib_available(),
        "bgzip and tabix are required by the BED schema inference regressions; run them with `pixi run test`"
    );
    let dir = TempDir::new().unwrap();
    let first = write_bed_bgzip_tabix(
        dir.path(),
        "first",
        &["chrom", "start", "end", "cov", "mapq"],
        &[("chr1", 0, 8, vec!["4", "60"])],
    );
    let second = write_bed_bgzip_tabix(
        dir.path(),
        "second",
        &["chrom", "start", "end", "cov", "mapq"],
        &[("chr1", 0, 8, vec!["70000", "30"])],
    );

    let dtypes = infer_bed_dtypes_for_sources(
        &[Source::new(first), Source::new(second)],
        &[3, 4],
        InferRows::All,
    )
    .unwrap();

    assert_eq!(dtypes, vec![Dtype::U32, Dtype::U8]);
}

#[test]
fn schema_dtype_inference_rejects_boolean_words_mixed_with_numbers_across_sources() {
    assert!(
        htslib_available(),
        "bgzip and tabix are required by the BED schema inference regressions; run them with `pixi run test`"
    );
    let dir = TempDir::new().unwrap();
    let boolean = write_bed_bgzip_tabix(
        dir.path(),
        "boolean",
        &["chrom", "start", "end", "value"],
        &[("chr1", 0, 8, vec!["true"])],
    );
    let numeric = write_bed_bgzip_tabix(
        dir.path(),
        "numeric",
        &["chrom", "start", "end", "value"],
        &[("chr1", 0, 8, vec!["2"])],
    );

    let error = infer_bed_dtypes_for_sources(
        &[Source::new(boolean), Source::new(numeric)],
        &[3],
        InferRows::All,
    )
    .expect_err("shared inference must reject a BED column that mixes Boolean words and numbers")
    .to_string();

    assert!(error.contains("mixes true/false with numbers"), "{error}");
}
