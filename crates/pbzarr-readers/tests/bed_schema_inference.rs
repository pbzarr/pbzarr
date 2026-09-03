mod common;

use pbzarr::io::Dtype;
use pbzarr_readers::{InferRows, Source, infer_bed_dtypes_for_sources};
use tempfile::TempDir;

use common::write_bed_bgzip_tabix;

#[test]
fn schema_dtype_inference_combines_observations_from_every_source() {
    let dir = TempDir::new().unwrap();
    let first = write_bed_bgzip_tabix(
        dir.path(),
        "first",
        &["chrom", "start", "end", "cov", "mapq", "flag"],
        &[("chr1", 0, 8, vec!["4", "60", "true"])],
    );
    let second = write_bed_bgzip_tabix(
        dir.path(),
        "second",
        &["chrom", "start", "end", "cov", "mapq", "flag"],
        &[("chr1", 0, 8, vec!["70000", "30", "2"])],
    );
    let sources = [Source::new(first), Source::new(second)];

    let dtypes = infer_bed_dtypes_for_sources(&sources, &[3, 4], InferRows::All).unwrap();

    assert_eq!(dtypes, vec![Dtype::U32, Dtype::U8]);

    let error = infer_bed_dtypes_for_sources(&sources, &[5], InferRows::All)
        .expect_err(
            "shared inference must reject a BED column that mixes Boolean words and numbers",
        )
        .to_string();

    assert!(error.contains("mixes true/false with numbers"), "{error}");
}
