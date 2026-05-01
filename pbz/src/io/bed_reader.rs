//! BED file reader. Streams bgzipped BED into an in-memory interval map.
//!
//! `BedIntervalSource` is the canonical interval producer. `BedPerBaseReader`
//! adapts it to the per-base `ValueReader` contract used by the v0.1 ingest
//! pipeline; the v0.2 interval-layout writer will consume `BedIntervalSource`
//! directly.

use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BedFlavor {
    Mask,
    BedGraph,
}

impl BedFlavor {
    pub(crate) fn expected_cols(self) -> usize {
        match self {
            BedFlavor::Mask => 3,
            BedFlavor::BedGraph => 4,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum IntervalValue {
    Mask,
    Numeric(f64),
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct IntervalRecord {
    pub start: u64,
    pub end: u64,
    pub value: IntervalValue,
}

/// Owns the result of streaming a single BED file: a flavor and a per-contig
/// vector of sorted, non-overlapping intervals.
#[derive(Debug)]
pub(crate) struct BedIntervalSource {
    pub(crate) path: PathBuf,
    pub(crate) flavor: BedFlavor,
    pub(crate) contigs: Vec<(String, u64)>,
    pub(crate) by_contig: HashMap<String, Vec<IntervalRecord>>,
}

use crate::io::bed_error::BedError;

pub(crate) fn is_skippable_line(line: &str) -> bool {
    let t = line.trim();
    t.is_empty()
        || t.starts_with('#')
        || t.starts_with("track ")
        || t == "track"
        || t.starts_with("browser ")
        || t == "browser"
}

pub(crate) fn detect_flavor(
    line: &str,
    path: &std::path::Path,
    lineno: usize,
) -> Result<BedFlavor, BedError> {
    let trimmed = line.trim_end_matches(['\r', '\n']);
    let cols = trimmed.split('\t').count();
    match cols {
        3 => Ok(BedFlavor::Mask),
        4 => {
            // First-line bedGraph detection: parse col 4 as f64.
            let mut it = trimmed.split('\t');
            it.next();
            it.next();
            it.next();
            let v = it.next().unwrap();
            v.parse::<f64>().map_err(|_| BedError::FlavorAmbiguous {
                path: path.to_path_buf(),
                line: lineno,
                got_cols: cols,
            })?;
            Ok(BedFlavor::BedGraph)
        }
        n => Err(BedError::FlavorAmbiguous {
            path: path.to_path_buf(),
            line: lineno,
            got_cols: n,
        }),
    }
}

pub(crate) fn parse_data_line(
    line: &str,
    path: &std::path::Path,
    lineno: usize,
    flavor: BedFlavor,
) -> Result<(String, IntervalRecord), BedError> {
    let trimmed = line.trim_end_matches(['\r', '\n']);
    let mut it = trimmed.split('\t');
    let contig = it.next().ok_or(BedError::Malformed {
        path: path.to_path_buf(),
        line: lineno,
    })?;
    let start_s = it.next().ok_or(BedError::Malformed {
        path: path.to_path_buf(),
        line: lineno,
    })?;
    let end_s = it.next().ok_or(BedError::Malformed {
        path: path.to_path_buf(),
        line: lineno,
    })?;

    let start: u64 = start_s.parse().map_err(|_| BedError::Malformed {
        path: path.to_path_buf(),
        line: lineno,
    })?;
    let end: u64 = end_s.parse().map_err(|_| BedError::Malformed {
        path: path.to_path_buf(),
        line: lineno,
    })?;

    if start > end {
        return Err(BedError::ReversedInterval {
            path: path.to_path_buf(),
            line: lineno,
            start,
            end,
        });
    }
    if start == end {
        return Err(BedError::ZeroLength {
            path: path.to_path_buf(),
            line: lineno,
            start,
        });
    }

    let value = match flavor {
        BedFlavor::Mask => {
            // Reject any extra columns beyond the expected 3.
            if it.next().is_some() {
                return Err(BedError::FlavorMixed {
                    path: path.to_path_buf(),
                    line: lineno,
                    expected: 3,
                    got: trimmed.split('\t').count(),
                });
            }
            IntervalValue::Mask
        }
        BedFlavor::BedGraph => {
            let v_s = it.next().ok_or(BedError::FlavorMixed {
                path: path.to_path_buf(),
                line: lineno,
                expected: 4,
                got: 3,
            })?;
            // Reject any extra columns beyond the expected 4.
            if it.next().is_some() {
                return Err(BedError::FlavorMixed {
                    path: path.to_path_buf(),
                    line: lineno,
                    expected: 4,
                    got: trimmed.split('\t').count(),
                });
            }
            let v: f64 = v_s.parse().map_err(|_| BedError::ValueParse {
                path: path.to_path_buf(),
                line: lineno,
                value: v_s.to_string(),
            })?;
            IntervalValue::Numeric(v)
        }
    };

    Ok((contig.to_string(), IntervalRecord { start, end, value }))
}

use noodles::bgzf;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::num::NonZero;
use std::path::Path;

impl BedIntervalSource {
    pub(crate) fn open(
        path: &Path,
        contigs: Vec<(String, u64)>,
        worker_count: NonZero<usize>,
    ) -> Result<Self, BedError> {
        let file = File::open(path).map_err(|e| BedError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;

        // Validate bgzf magic by attempting to construct the reader.
        let reader = bgzf::io::MultithreadedReader::with_worker_count(worker_count, file);
        let mut buf = BufReader::new(reader);

        let contig_lengths: HashMap<String, u64> =
            contigs.iter().map(|(n, l)| (n.clone(), *l)).collect();

        let mut flavor: Option<BedFlavor> = None;
        let mut by_contig: HashMap<String, Vec<IntervalRecord>> = HashMap::new();
        let mut prev: Option<(String, u64)> = None;
        let mut max_end_per_contig: HashMap<String, u64> = HashMap::new();

        let mut line = String::new();
        let mut lineno: usize = 0;
        let mut data_seen = false;

        loop {
            line.clear();
            let n = buf.read_line(&mut line).map_err(|e| BedError::Io {
                path: path.to_path_buf(),
                source: e,
            })?;
            if n == 0 {
                break;
            }
            lineno += 1;
            if is_skippable_line(&line) {
                continue;
            }

            let f = match flavor {
                Some(f) => f,
                None => {
                    let f = detect_flavor(&line, path, lineno)?;
                    flavor = Some(f);
                    f
                }
            };
            let (contig, rec) = parse_data_line(&line, path, lineno, f)?;
            data_seen = true;

            // Contig membership
            let contig_length =
                contig_lengths
                    .get(&contig)
                    .ok_or_else(|| BedError::UnknownContig {
                        path: path.to_path_buf(),
                        line: lineno,
                        contig: contig.clone(),
                    })?;
            if rec.end > *contig_length {
                return Err(BedError::PastContigEnd {
                    path: path.to_path_buf(),
                    line: lineno,
                    contig: contig.clone(),
                    contig_length: *contig_length,
                    end: rec.end,
                });
            }

            // Sortedness within contig
            if let Some((prev_contig, prev_start)) = &prev {
                if *prev_contig == contig && rec.start < *prev_start {
                    return Err(BedError::Unsorted {
                        path: path.to_path_buf(),
                        line: lineno,
                        contig: contig.clone(),
                        this_start: rec.start,
                        prev_contig: prev_contig.clone(),
                        prev_start: *prev_start,
                    });
                }
            }
            prev = Some((contig.clone(), rec.start));

            // Overlap within contig
            let max_end = max_end_per_contig.entry(contig.clone()).or_insert(0);
            if rec.start < *max_end {
                return Err(BedError::Overlap {
                    path: path.to_path_buf(),
                    line: lineno,
                    contig: contig.clone(),
                    this_start: rec.start,
                    this_end: rec.end,
                    prev_end: *max_end,
                });
            }
            *max_end = rec.end;

            by_contig.entry(contig).or_default().push(rec);
        }

        if !data_seen {
            return Err(BedError::Empty {
                path: path.to_path_buf(),
            });
        }

        Ok(Self {
            path: path.to_path_buf(),
            flavor: flavor.expect("data_seen ⇒ flavor set"),
            contigs,
            by_contig,
        })
    }
}

use crate::io::interval_to_per_base::{
    cast_f64_to_f32, cast_f64_to_f64, cast_f64_to_int8, cast_f64_to_int16, cast_f64_to_int32,
    cast_f64_to_uint8, cast_f64_to_uint16, cast_f64_to_uint32, cast_mask_to_bool,
    expand_intervals_into,
};
use crate::io::value_reader::{ValueChunk, ValueDtype, ValueReader};
use color_eyre::eyre::eyre;

impl BedIntervalSource {
    pub(crate) fn intervals_in_region<'a>(
        &'a self,
        contig: &str,
        start: u64,
        end: u64,
    ) -> impl Iterator<Item = &'a IntervalRecord> + 'a {
        let v = self.by_contig.get(contig);
        let slice: &'a [IntervalRecord] = v.map(|x| x.as_slice()).unwrap_or(&[]);
        // Binary-search for first interval with end > start.
        let lo = slice.partition_point(|r| r.end <= start);
        slice[lo..].iter().take_while(move |r| r.start < end)
    }
}

#[derive(Debug)]
pub struct BedPerBaseReader {
    source: BedIntervalSource,
    dtype: ValueDtype,
}

impl BedPerBaseReader {
    pub fn from_source(
        source: BedIntervalSource,
        dtype: ValueDtype,
    ) -> color_eyre::Result<Self> {
        match (source.flavor, dtype) {
            (BedFlavor::Mask, ValueDtype::Bool) => {}
            (BedFlavor::Mask, _) => {
                return Err(eyre!(
                    "BED file {:?} is BED3 (mask); track dtype must be 'bool', got {dtype}",
                    source.path
                ));
            }
            (BedFlavor::BedGraph, ValueDtype::Bool) => {
                return Err(eyre!(
                    "BED file {:?} is bedGraph (numeric); track dtype must not be 'bool'",
                    source.path
                ));
            }
            (BedFlavor::BedGraph, _) => {}
        }
        Ok(Self { source, dtype })
    }

    pub fn flavor(&self) -> BedFlavor {
        self.source.flavor
    }
}

impl ValueReader for BedPerBaseReader {
    fn dtype(&self) -> ValueDtype {
        self.dtype
    }
    fn contigs(&self) -> &[(String, u64)] {
        &self.source.contigs
    }

    fn read_chunk(
        &mut self,
        contig: &str,
        start: u64,
        end: u64,
    ) -> color_eyre::Result<ValueChunk> {
        let iter = self.source.intervals_in_region(contig, start, end);
        let p = self.source.path.as_path();
        let chunk = match self.dtype {
            ValueDtype::Bool => ValueChunk::Bool(expand_intervals_into(
                iter,
                contig,
                start,
                end,
                false,
                cast_mask_to_bool,
                p,
            )?),
            ValueDtype::F32 => ValueChunk::F32(expand_intervals_into(
                iter,
                contig,
                start,
                end,
                f32::NAN,
                cast_f64_to_f32,
                p,
            )?),
            ValueDtype::F64 => ValueChunk::F64(expand_intervals_into(
                iter,
                contig,
                start,
                end,
                f64::NAN,
                cast_f64_to_f64,
                p,
            )?),
            ValueDtype::U8 => ValueChunk::U8(expand_intervals_into(
                iter,
                contig,
                start,
                end,
                0u8,
                cast_f64_to_uint8,
                p,
            )?),
            ValueDtype::U16 => ValueChunk::U16(expand_intervals_into(
                iter,
                contig,
                start,
                end,
                0u16,
                cast_f64_to_uint16,
                p,
            )?),
            ValueDtype::U32 => ValueChunk::U32(expand_intervals_into(
                iter,
                contig,
                start,
                end,
                0u32,
                cast_f64_to_uint32,
                p,
            )?),
            ValueDtype::I8 => ValueChunk::I8(expand_intervals_into(
                iter,
                contig,
                start,
                end,
                0i8,
                cast_f64_to_int8,
                p,
            )?),
            ValueDtype::I16 => ValueChunk::I16(expand_intervals_into(
                iter,
                contig,
                start,
                end,
                0i16,
                cast_f64_to_int16,
                p,
            )?),
            ValueDtype::I32 => ValueChunk::I32(expand_intervals_into(
                iter,
                contig,
                start,
                end,
                0i32,
                cast_f64_to_int32,
                p,
            )?),
        };
        Ok(chunk)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn p() -> PathBuf {
        PathBuf::from("/test.bed.gz")
    }

    #[test]
    fn parse_bed3_line() {
        let (contig, rec) = parse_data_line("chr1\t10\t20", &p(), 1, BedFlavor::Mask).unwrap();
        assert_eq!(contig, "chr1");
        assert_eq!(rec.start, 10);
        assert_eq!(rec.end, 20);
        assert_eq!(rec.value, IntervalValue::Mask);
    }

    #[test]
    fn parse_bedgraph_line() {
        let (contig, rec) =
            parse_data_line("chr1\t10\t20\t3.5", &p(), 1, BedFlavor::BedGraph).unwrap();
        assert_eq!(contig, "chr1");
        assert_eq!(rec.start, 10);
        assert_eq!(rec.end, 20);
        assert_eq!(rec.value, IntervalValue::Numeric(3.5));
    }

    #[test]
    fn parse_rejects_zero_length() {
        let err = parse_data_line("chr1\t10\t10", &p(), 7, BedFlavor::Mask).unwrap_err();
        assert!(matches!(
            err,
            crate::io::bed_error::BedError::ZeroLength { line: 7, .. }
        ));
    }

    #[test]
    fn parse_rejects_reversed() {
        let err = parse_data_line("chr1\t20\t10", &p(), 7, BedFlavor::Mask).unwrap_err();
        assert!(matches!(
            err,
            crate::io::bed_error::BedError::ReversedInterval { line: 7, .. }
        ));
    }

    #[test]
    fn parse_rejects_too_few_cols() {
        let err = parse_data_line("chr1\t10", &p(), 7, BedFlavor::Mask).unwrap_err();
        assert!(matches!(
            err,
            crate::io::bed_error::BedError::Malformed { line: 7, .. }
        ));
    }

    #[test]
    fn parse_rejects_flavor_mismatch_extra_cols() {
        let err = parse_data_line("chr1\t10\t20\t3.5", &p(), 7, BedFlavor::Mask).unwrap_err();
        assert!(matches!(
            err,
            crate::io::bed_error::BedError::FlavorMixed { line: 7, .. }
        ));
    }

    #[test]
    fn parse_rejects_unparseable_bedgraph_value() {
        let err = parse_data_line("chr1\t10\t20\tfoo", &p(), 9, BedFlavor::BedGraph).unwrap_err();
        assert!(matches!(
            err,
            crate::io::bed_error::BedError::ValueParse { line: 9, .. }
        ));
    }

    #[test]
    fn parse_tolerates_trailing_cr() {
        let (_, rec) = parse_data_line("chr1\t10\t20\r", &p(), 1, BedFlavor::Mask).unwrap();
        assert_eq!(rec.start, 10);
        assert_eq!(rec.end, 20);
    }

    #[test]
    fn detect_flavor_bed3() {
        assert_eq!(
            detect_flavor("chr1\t10\t20", &p(), 1).unwrap(),
            BedFlavor::Mask
        );
    }

    #[test]
    fn detect_flavor_bedgraph() {
        assert_eq!(
            detect_flavor("chr1\t10\t20\t1.5", &p(), 1).unwrap(),
            BedFlavor::BedGraph
        );
    }

    #[test]
    fn detect_flavor_too_few_cols_errors() {
        let err = detect_flavor("chr1\t10", &p(), 1).unwrap_err();
        assert!(matches!(
            err,
            crate::io::bed_error::BedError::FlavorAmbiguous { .. }
        ));
    }

    #[test]
    fn detect_flavor_too_many_cols_errors() {
        let err = detect_flavor("chr1\t10\t20\t1.5\textra", &p(), 1).unwrap_err();
        assert!(matches!(
            err,
            crate::io::bed_error::BedError::FlavorAmbiguous { .. }
        ));
    }

    #[test]
    fn is_skippable_recognizes_comments_and_headers() {
        assert!(is_skippable_line("# comment"));
        assert!(is_skippable_line("track name=foo"));
        assert!(is_skippable_line("browser dense"));
        assert!(is_skippable_line(""));
        assert!(is_skippable_line("   "));
        assert!(!is_skippable_line("chr1\t10\t20"));
    }

    use noodles::bgzf;
    use std::io::Write;

    /// Build a bgzipped buffer from raw BED text. Returns a tempfile path.
    fn build_bgz(text: &str) -> tempfile::NamedTempFile {
        let f = tempfile::Builder::new()
            .prefix("test-bed-")
            .suffix(".bed.gz")
            .tempfile()
            .unwrap();
        let mut writer = bgzf::io::Writer::new(std::fs::File::create(f.path()).unwrap());
        writer.write_all(text.as_bytes()).unwrap();
        writer.finish().unwrap();
        f
    }

    fn contigs_grch38_subset() -> Vec<(String, u64)> {
        vec![("chr1".into(), 1_000_000), ("chr2".into(), 500_000)]
    }

    #[test]
    fn open_loads_bed3_mask() {
        let f = build_bgz("chr1\t10\t20\nchr1\t30\t40\nchr2\t100\t200\n");
        let src = BedIntervalSource::open(
            f.path(),
            contigs_grch38_subset(),
            std::num::NonZero::new(1).unwrap(),
        )
        .unwrap();
        assert_eq!(src.flavor, BedFlavor::Mask);
        assert_eq!(src.by_contig["chr1"].len(), 2);
        assert_eq!(src.by_contig["chr2"].len(), 1);
        assert_eq!(src.by_contig["chr1"][0].start, 10);
        assert_eq!(src.by_contig["chr1"][1].start, 30);
    }

    #[test]
    fn open_loads_bedgraph() {
        let f = build_bgz("chr1\t10\t20\t1.5\nchr1\t30\t40\t2.5\n");
        let src = BedIntervalSource::open(
            f.path(),
            contigs_grch38_subset(),
            std::num::NonZero::new(1).unwrap(),
        )
        .unwrap();
        assert_eq!(src.flavor, BedFlavor::BedGraph);
        assert_eq!(src.by_contig["chr1"][0].value, IntervalValue::Numeric(1.5));
        assert_eq!(src.by_contig["chr1"][1].value, IntervalValue::Numeric(2.5));
    }

    #[test]
    fn open_skips_comments_and_headers() {
        let f = build_bgz("# header\ntrack name=foo\nchr1\t10\t20\nbrowser dense\nchr1\t30\t40\n");
        let src = BedIntervalSource::open(
            f.path(),
            contigs_grch38_subset(),
            std::num::NonZero::new(1).unwrap(),
        )
        .unwrap();
        assert_eq!(src.by_contig["chr1"].len(), 2);
    }

    #[test]
    fn open_rejects_empty() {
        let f = build_bgz("# nothing\n");
        let err = BedIntervalSource::open(
            f.path(),
            contigs_grch38_subset(),
            std::num::NonZero::new(1).unwrap(),
        )
        .unwrap_err();
        assert!(matches!(err, BedError::Empty { .. }));
    }

    #[test]
    fn open_rejects_unsorted() {
        let f = build_bgz("chr1\t100\t200\nchr1\t10\t20\n");
        let err = BedIntervalSource::open(
            f.path(),
            contigs_grch38_subset(),
            std::num::NonZero::new(1).unwrap(),
        )
        .unwrap_err();
        assert!(matches!(err, BedError::Unsorted { .. }));
    }

    #[test]
    fn open_rejects_overlap() {
        let f = build_bgz("chr1\t10\t100\nchr1\t50\t200\n");
        let err = BedIntervalSource::open(
            f.path(),
            contigs_grch38_subset(),
            std::num::NonZero::new(1).unwrap(),
        )
        .unwrap_err();
        assert!(matches!(err, BedError::Overlap { .. }));
    }

    #[test]
    fn open_rejects_unknown_contig() {
        let f = build_bgz("chrX\t10\t20\n");
        let err = BedIntervalSource::open(
            f.path(),
            contigs_grch38_subset(),
            std::num::NonZero::new(1).unwrap(),
        )
        .unwrap_err();
        assert!(matches!(err, BedError::UnknownContig { .. }));
    }

    #[test]
    fn open_rejects_past_contig_end() {
        let f = build_bgz("chr2\t499000\t600000\n");
        let err = BedIntervalSource::open(
            f.path(),
            contigs_grch38_subset(),
            std::num::NonZero::new(1).unwrap(),
        )
        .unwrap_err();
        assert!(matches!(err, BedError::PastContigEnd { .. }));
    }

    #[test]
    fn open_rejects_flavor_mixed() {
        let f = build_bgz("chr1\t10\t20\nchr1\t30\t40\t1.5\n");
        let err = BedIntervalSource::open(
            f.path(),
            contigs_grch38_subset(),
            std::num::NonZero::new(1).unwrap(),
        )
        .unwrap_err();
        assert!(matches!(err, BedError::FlavorMixed { .. }));
    }

    #[test]
    fn open_contigs_within_a_file_can_change() {
        // Sortedness is per-contig, but the file may move from chr1 to chr2.
        let f = build_bgz("chr1\t10\t20\nchr1\t30\t40\nchr2\t10\t20\n");
        let src = BedIntervalSource::open(
            f.path(),
            contigs_grch38_subset(),
            std::num::NonZero::new(1).unwrap(),
        )
        .unwrap();
        assert_eq!(src.by_contig["chr1"].len(), 2);
        assert_eq!(src.by_contig["chr2"].len(), 1);
    }

    use crate::io::value_reader::{ValueChunk, ValueDtype, ValueReader};

    #[test]
    fn intervals_in_region_binary_search() {
        let f = build_bgz("chr1\t10\t20\nchr1\t30\t40\nchr1\t50\t60\nchr1\t70\t80\n");
        let src = BedIntervalSource::open(
            f.path(),
            contigs_grch38_subset(),
            std::num::NonZero::new(1).unwrap(),
        )
        .unwrap();

        let v: Vec<_> = src.intervals_in_region("chr1", 25, 65).collect();
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].start, 30);
        assert_eq!(v[1].start, 50);
    }

    #[test]
    fn intervals_in_region_unknown_contig_empty() {
        let f = build_bgz("chr1\t10\t20\n");
        let src = BedIntervalSource::open(
            f.path(),
            contigs_grch38_subset(),
            std::num::NonZero::new(1).unwrap(),
        )
        .unwrap();
        assert_eq!(src.intervals_in_region("chrZ", 0, 100).count(), 0);
    }

    #[test]
    fn bed_per_base_reader_mask() {
        let f = build_bgz("chr1\t10\t20\nchr1\t30\t40\n");
        let src = BedIntervalSource::open(
            f.path(),
            contigs_grch38_subset(),
            std::num::NonZero::new(1).unwrap(),
        )
        .unwrap();
        let mut reader = BedPerBaseReader::from_source(src, ValueDtype::Bool).unwrap();
        let chunk = reader.read_chunk("chr1", 0, 50).unwrap();
        match chunk {
            ValueChunk::Bool(arr) => {
                assert_eq!(arr.len(), 50);
                assert!(arr[15]);
                assert!(!arr[5]);
                assert!(arr[35]);
                assert!(!arr[25]);
            }
            _ => panic!("expected Bool chunk"),
        }
    }

    #[test]
    fn bed_per_base_reader_bedgraph_default_f32() {
        let f = build_bgz("chr1\t10\t15\t1.5\nchr1\t20\t25\t2.5\n");
        let src = BedIntervalSource::open(
            f.path(),
            contigs_grch38_subset(),
            std::num::NonZero::new(1).unwrap(),
        )
        .unwrap();
        let mut reader = BedPerBaseReader::from_source(src, ValueDtype::F32).unwrap();
        let chunk = reader.read_chunk("chr1", 0, 30).unwrap();
        match chunk {
            ValueChunk::F32(arr) => {
                assert_eq!(arr.len(), 30);
                assert!(arr[12].is_finite() && (arr[12] - 1.5).abs() < 1e-6);
                assert!(arr[22].is_finite() && (arr[22] - 2.5).abs() < 1e-6);
                assert!(arr[5].is_nan());
            }
            _ => panic!("expected F32 chunk"),
        }
    }

    #[test]
    fn bed_per_base_reader_rejects_dtype_mismatch() {
        let f = build_bgz("chr1\t10\t20\n"); // Mask flavor
        let src = BedIntervalSource::open(
            f.path(),
            contigs_grch38_subset(),
            std::num::NonZero::new(1).unwrap(),
        )
        .unwrap();
        let err = BedPerBaseReader::from_source(src, ValueDtype::F32).unwrap_err();
        assert!(format!("{err}").contains("dtype"));
    }
}
