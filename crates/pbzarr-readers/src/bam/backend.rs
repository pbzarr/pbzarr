use std::fs::File;
use std::io::Read as IoRead;
use std::path::{Path, PathBuf};

use noodles_bam as bam;
use noodles_bgzf_bam as bgzf;
use noodles_core::Region as NoodlesRegion;
use noodles_sam as sam;
use rust_htslib::bam::Read as HtslibRead;
use rust_htslib::bam::record::Cigar as HtslibCigar;

use pbzarr::genome::{Contig, Genome};
use pbzarr::io::{ReaderError, Result};

use crate::coords::{noodles_query_interval, pos_to_zero_based};

use super::record::{AlignedRead, CigarKind};

/// A BAM index, read into a concrete type rather than relying on
/// `noodles_bam::io::IndexedReader`'s internal `Box<dyn BinningIndex>`.
pub enum BamIndex {
    Bai(bam::bai::Index),
    Csi(noodles_csi_bam::Index),
}


fn bam_query<'r>(
    reader: &'r mut bam::io::Reader<bgzf::io::Reader<File>>,
    header: &sam::Header,
    index: &BamIndex,
    region: &NoodlesRegion,
) -> std::io::Result<bam::io::reader::Query<'r, bgzf::io::Reader<File>>> {
    match index {
        BamIndex::Bai(idx) => reader.query(header, idx, region),
        BamIndex::Csi(idx) => reader.query(header, idx, region),
    }
}

pub enum Backend {
    Bam {
        reader: Box<bam::io::Reader<bgzf::io::Reader<File>>>,
        header: Box<sam::Header>,
        index: BamIndex,
    },
    Cram {
        reader: rust_htslib::bam::IndexedReader,
    },
}

impl Backend {
    /// Open `path` as BAM or CRAM (by content magic, falling back to the
    /// extension) and derive its `Genome`
    /// from the header's `@SQ` lines. `reference` is the FASTA a CRAM needs
    /// to decode records; required up front so a missing reference fails at
    /// open rather than on the first `fetch`.
    pub fn open(path: &Path, reference: Option<&Path>) -> Result<(Backend, Genome)> {
        if is_cram(path) {
            open_cram(path, reference)
        } else {
            open_bam(path)
        }
    }

    pub fn fetch(
        &mut self,
        contig: &str,
        start: u64,
        end: u64,
        visit: &mut dyn FnMut(&AlignedRead) -> Result<()>,
    ) -> Result<()> {
        match self {
            Backend::Bam {
                reader,
                header,
                index,
            } => fetch_bam(reader, header, index, contig, start, end, visit),
            Backend::Cram { reader } => fetch_cram(reader, contig, start, end, visit),
        }
    }

    /// Cheap existence probe: does `[start, end)` on `contig` overlap any
    /// record? Stops at the first hit instead of decoding the whole region.
    pub fn has_records(&mut self, contig: &str, start: u64, end: u64) -> Result<bool> {
        match self {
            Backend::Bam {
                reader,
                header,
                index,
            } => has_records_bam(reader, header, index, contig, start, end),
            Backend::Cram { reader } => has_records_cram(reader, contig, start, end),
        }
    }
}

const CRAM_MAGIC: [u8; 4] = *b"CRAM";
const BGZF_MAGIC: [u8; 2] = [0x1f, 0x8b];

fn is_cram(path: &Path) -> bool {
    match sniff_magic(path) {
        Some(magic) if magic[..4] == CRAM_MAGIC => true,
        Some(magic) if magic[..2] == BGZF_MAGIC => false,
        _ => is_cram_by_extension(path),
    }
}

fn sniff_magic(path: &Path) -> Option<[u8; 4]> {
    let mut file = File::open(path).ok()?;
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf).ok()?;
    Some(buf)
}

fn is_cram_by_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("cram"))
}

fn open_bam(path: &Path) -> Result<(Backend, Genome)> {
    let file = File::open(path).map_err(|source| ReaderError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut reader = bam::io::Reader::new(file);
    let header = reader.read_header().map_err(|source| ReaderError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let index = read_associated_index(path)?;

    if header.reference_sequences().is_empty() {
        return Err(ReaderError::Other(anyhow::anyhow!(
            "{} has no @SQ reference sequences",
            path.display()
        )));
    }
    let contigs = header
        .reference_sequences()
        .iter()
        .map(|(name, map)| Contig {
            name: String::from_utf8_lossy(name.as_ref()).into_owned(),
            length: map.length().get() as u64,
        })
        .collect();
    let genome = Genome::new(contigs).map_err(|e| ReaderError::Other(anyhow::anyhow!(e)))?;

    Ok((
        Backend::Bam {
            reader: Box::new(reader),
            header: Box::new(header),
            index,
        },
        genome,
    ))
}

/// Read `<path>.bai`, falling back to `<path>.csi` (mirrors
/// `noodles_bam::io::indexed_reader::Builder`'s associated-index lookup,
/// minus the `Box<dyn BinningIndex>` it stores internally).
fn read_associated_index(path: &Path) -> Result<BamIndex> {
    let bai_path = append_ext(path, "bai");
    match bam::bai::fs::read(&bai_path) {
        Ok(index) => Ok(BamIndex::Bai(index)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            let csi_path = append_ext(path, "csi");
            match noodles_csi_bam::fs::read(&csi_path) {
                Ok(index) => Ok(BamIndex::Csi(index)),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    Err(ReaderError::Other(anyhow::anyhow!(
                        "missing index for {}: expected a .bai or .csi alongside it",
                        path.display()
                    )))
                }
                Err(source) => Err(ReaderError::Io {
                    path: csi_path,
                    source,
                }),
            }
        }
        Err(source) => Err(ReaderError::Io {
            path: bai_path,
            source,
        }),
    }
}

fn append_ext(path: &Path, ext: &str) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(".");
    s.push(ext);
    PathBuf::from(s)
}

fn open_cram(path: &Path, reference: Option<&Path>) -> Result<(Backend, Genome)> {
    let Some(reference) = reference else {
        return Err(ReaderError::Other(anyhow::anyhow!(
            "CRAM {} requires an explicit reference FASTA",
            path.display()
        )));
    };

    let mut reader = rust_htslib::bam::IndexedReader::from_path(path).map_err(|source| {
        ReaderError::Other(anyhow::anyhow!(
            "opening CRAM {} (index missing or invalid?): {source}",
            path.display()
        ))
    })?;
    reader.set_reference(reference).map_err(|source| {
        ReaderError::Other(anyhow::anyhow!(
            "setting CRAM reference {} for {}: {source}",
            reference.display(),
            path.display()
        ))
    })?;

    let header = reader.header();
    if header.target_count() == 0 {
        return Err(ReaderError::Other(anyhow::anyhow!(
            "{} has no @SQ reference sequences",
            path.display()
        )));
    }
    let contigs = header
        .target_names()
        .iter()
        .enumerate()
        .map(|(tid, name)| {
            let length = header.target_len(tid as u32).unwrap_or(0);
            Contig {
                name: String::from_utf8_lossy(name).into_owned(),
                length,
            }
        })
        .collect();
    let genome = Genome::new(contigs).map_err(|e| ReaderError::Other(anyhow::anyhow!(e)))?;

    Ok((Backend::Cram { reader }, genome))
}

/// Builds the half-open `[start, end)` query region (0-based) as noodles'
/// 1-based inclusive `Region`, and runs it against `index` to get a `Query`
/// iterator. Returns `None` for an empty/inverted range instead of erroring.
fn bam_query_region<'r>(
    reader: &'r mut bam::io::Reader<bgzf::io::Reader<File>>,
    header: &sam::Header,
    index: &BamIndex,
    contig: &str,
    start: u64,
    end: u64,
) -> Result<Option<bam::io::reader::Query<'r, bgzf::io::Reader<File>>>> {
    let Some(interval) = noodles_query_interval(start, end)? else {
        return Ok(None);
    };
    let region = NoodlesRegion::new(contig, interval);

    bam_query(reader, header, index, &region)
        .map(Some)
        .map_err(|source| {
            ReaderError::Other(anyhow::anyhow!("query {contig}:{start}-{end}: {source}"))
        })
}

fn fetch_bam(
    reader: &mut bam::io::Reader<bgzf::io::Reader<File>>,
    header: &sam::Header,
    index: &BamIndex,
    contig: &str,
    start: u64,
    end: u64,
    visit: &mut dyn FnMut(&AlignedRead) -> Result<()>,
) -> Result<()> {
    let Some(mut query) = bam_query_region(reader, header, index, contig, start, end)? else {
        return Ok(());
    };

    let mut record = noodles_bam::Record::default();
    let mut read = AlignedRead::default();
    loop {
        let n = query
            .read_record(&mut record)
            .map_err(|source| ReaderError::Other(anyhow::anyhow!("decode BAM record: {source}")))?;
        if n == 0 {
            break;
        }
        decode_bam_record(&record, &mut read)?;
        visit(&read)?;
    }
    Ok(())
}

fn has_records_bam(
    reader: &mut bam::io::Reader<bgzf::io::Reader<File>>,
    header: &sam::Header,
    index: &BamIndex,
    contig: &str,
    start: u64,
    end: u64,
) -> Result<bool> {
    let Some(mut query) = bam_query_region(reader, header, index, contig, start, end)? else {
        return Ok(false);
    };

    let mut record = noodles_bam::Record::default();
    let n = query
        .read_record(&mut record)
        .map_err(|source| ReaderError::Other(anyhow::anyhow!("decode BAM record: {source}")))?;
    Ok(n != 0)
}

fn has_records_cram(
    reader: &mut rust_htslib::bam::IndexedReader,
    contig: &str,
    start: u64,
    end: u64,
) -> Result<bool> {
    // htslib's fetch() is natively 0-based half-open; no conversion needed.
    reader
        .fetch((contig, start as i64, end as i64))
        .map_err(|source| {
            ReaderError::Other(anyhow::anyhow!("fetch {contig}:{start}-{end}: {source}"))
        })?;

    let mut record = rust_htslib::bam::Record::new();
    match reader.read(&mut record) {
        None => Ok(false),
        Some(result) => {
            result.map_err(|source| {
                ReaderError::Other(anyhow::anyhow!("decode CRAM record: {source}"))
            })?;
            Ok(true)
        }
    }
}

fn decode_bam_record(record: &noodles_bam::Record, out: &mut AlignedRead) -> Result<()> {
    out.clear();

    let flags = record.flags();
    out.flags = flags.bits();
    out.paired = flags.is_segmented();

    out.mapq = record.mapping_quality().map(|m| m.get()).unwrap_or(u8::MAX);

    let start = record
        .alignment_start()
        .transpose()
        .map_err(|e| ReaderError::Other(anyhow::anyhow!("alignment_start: {e}")))?
        .map(pos_to_zero_based)
        .unwrap_or(0);
    out.start = start;

    let ref_id = record
        .reference_sequence_id()
        .transpose()
        .map_err(|e| ReaderError::Other(anyhow::anyhow!("reference_sequence_id: {e}")))?;
    let mate_ref_id = record
        .mate_reference_sequence_id()
        .transpose()
        .map_err(|e| ReaderError::Other(anyhow::anyhow!("mate_reference_sequence_id: {e}")))?;
    out.mate_same_contig = matches!((ref_id, mate_ref_id), (Some(a), Some(b)) if a == b);

    out.mate_start = record
        .mate_alignment_start()
        .transpose()
        .map_err(|e| ReaderError::Other(anyhow::anyhow!("mate_alignment_start: {e}")))?
        .map(|p| pos_to_zero_based(p) as i64)
        .unwrap_or(-1);

    if let Some(name) = record.name() {
        out.set_qname(name.as_ref());
    }

    for op in record.cigar().iter() {
        let op = op.map_err(|e| ReaderError::Other(anyhow::anyhow!("cigar op: {e}")))?;
        out.cigar
            .push((map_sam_cigar_kind(op.kind()), op.len() as u32));
    }

    out.seq.extend(record.sequence().iter());
    out.qual.extend(record.quality_scores().iter());

    Ok(())
}

fn map_sam_cigar_kind(kind: sam::alignment::record::cigar::op::Kind) -> CigarKind {
    use sam::alignment::record::cigar::op::Kind;
    match kind {
        Kind::Match | Kind::SequenceMatch | Kind::SequenceMismatch => CigarKind::Match,
        Kind::Insertion => CigarKind::Ins,
        Kind::Deletion => CigarKind::Del,
        Kind::Skip => CigarKind::RefSkip,
        Kind::SoftClip => CigarKind::SoftClip,
        Kind::HardClip | Kind::Pad => CigarKind::Other,
    }
}

fn fetch_cram(
    reader: &mut rust_htslib::bam::IndexedReader,
    contig: &str,
    start: u64,
    end: u64,
    visit: &mut dyn FnMut(&AlignedRead) -> Result<()>,
) -> Result<()> {
    // htslib's fetch() is natively 0-based half-open; no conversion needed.
    reader
        .fetch((contig, start as i64, end as i64))
        .map_err(|source| {
            ReaderError::Other(anyhow::anyhow!("fetch {contig}:{start}-{end}: {source}"))
        })?;

    let mut record = rust_htslib::bam::Record::new();
    let mut read = AlignedRead::default();
    loop {
        match reader.read(&mut record) {
            None => break,
            Some(result) => {
                result.map_err(|source| {
                    ReaderError::Other(anyhow::anyhow!("decode CRAM record: {source}"))
                })?;
                decode_cram_record(&record, &mut read);
                visit(&read)?;
            }
        }
    }
    Ok(())
}

fn decode_cram_record(record: &rust_htslib::bam::Record, out: &mut AlignedRead) {
    out.clear();

    out.flags = record.flags();
    out.paired = record.is_paired();
    out.mapq = record.mapq();
    // htslib's pos()/mpos() are natively 0-based; no conversion needed. An
    // unmapped record's pos() is -1, clamped to the 0 sentinel here.
    out.start = record.pos().max(0) as u64;

    out.mate_same_contig =
        record.is_paired() && record.tid() == record.mtid() && record.mtid() >= 0;
    out.mate_start = if record.is_paired() && record.mtid() >= 0 {
        record.mpos()
    } else {
        -1
    };

    out.set_qname(record.qname());

    for cigar in record.cigar().iter() {
        let (kind, len) = map_htslib_cigar(cigar);
        out.cigar.push((kind, len));
    }

    let seq = record.seq();
    out.seq.reserve(seq.len());
    for i in 0..seq.len() {
        out.seq.push(seq[i]);
    }

    out.qual.extend_from_slice(record.qual());
}

fn map_htslib_cigar(cigar: &HtslibCigar) -> (CigarKind, u32) {
    match *cigar {
        HtslibCigar::Match(len) | HtslibCigar::Equal(len) | HtslibCigar::Diff(len) => {
            (CigarKind::Match, len)
        }
        HtslibCigar::Ins(len) => (CigarKind::Ins, len),
        HtslibCigar::Del(len) => (CigarKind::Del, len),
        HtslibCigar::RefSkip(len) => (CigarKind::RefSkip, len),
        HtslibCigar::SoftClip(len) => (CigarKind::SoftClip, len),
        HtslibCigar::HardClip(len) | HtslibCigar::Pad(len) => (CigarKind::Other, len),
    }
}
