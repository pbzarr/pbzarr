//! `pbz stat`: summary statistics for one track.

use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use anyhow::{Context as _, Result, bail};
use pbzarr::{Genome, Region, RegionQuery};

#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct NamedRegion {
    pub chrom: String,
    pub region: Region,
}

/// Plain, gzip, or bgzip BED by the 2-byte gzip magic.
fn open_bed_reader(path: &Path) -> Result<Box<dyn BufRead>> {
    let mut file = File::open(path).with_context(|| format!("open regions {}", path.display()))?;
    let mut magic = [0u8; 2];
    let got = file
        .read(&mut magic)
        .with_context(|| format!("read {}", path.display()))?;
    file.seek(SeekFrom::Start(0))
        .with_context(|| format!("read {}", path.display()))?;
    if got == 2 && magic == [0x1f, 0x8b] {
        Ok(Box::new(BufReader::new(flate2::read::MultiGzDecoder::new(
            file,
        ))))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
}

#[allow(dead_code)]
pub(crate) fn load_bed_regions(path: &Path, genome: &Genome) -> Result<Vec<NamedRegion>> {
    let reader = open_bed_reader(path)?;
    let mut out = Vec::new();
    for (index, line) in reader.lines().enumerate() {
        let lineno = index + 1;
        let line = line.with_context(|| format!("read {} line {lineno}", path.display()))?;
        let text = line.trim_end();
        if text.is_empty()
            || text.starts_with('#')
            || text.starts_with("track")
            || text.starts_with("browser")
        {
            continue;
        }
        let mut cols = text.split('\t');
        let (Some(chrom), Some(start), Some(end)) = (cols.next(), cols.next(), cols.next()) else {
            bail!(
                "{} line {lineno}: expected at least 3 tab-separated columns",
                path.display()
            );
        };
        let start: u64 = start
            .parse()
            .with_context(|| format!("{} line {lineno}: bad start {start:?}", path.display()))?;
        let end: u64 = end
            .parse()
            .with_context(|| format!("{} line {lineno}: bad end {end:?}", path.display()))?;
        let query = RegionQuery {
            contig: chrom.to_string(),
            start: Some(start),
            end: Some(end),
        };
        let region = genome
            .resolve(&query)
            .with_context(|| format!("{} line {lineno}", path.display()))?;
        out.push(NamedRegion {
            chrom: chrom.to_string(),
            region,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pbzarr::Contig;
    use std::io::Write;

    fn genome() -> Genome {
        Genome::new(vec![
            Contig {
                name: "chr1".into(),
                length: 10,
            },
            Contig {
                name: "chr2".into(),
                length: 6,
            },
        ])
        .unwrap()
    }

    #[test]
    fn load_bed_keeps_order_and_skips_headers() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("r.bed");
        std::fs::write(
            &path,
            "# comment\ntrack name=x\nbrowser position chr1\n\nchr2\t0\t6\tname\t7\nchr1\t3\t8\nchr1\t3\t8\n",
        )
        .unwrap();
        let regions = load_bed_regions(&path, &genome()).unwrap();
        let flat: Vec<(String, u64, u64)> = regions
            .iter()
            .map(|n| (n.chrom.clone(), n.region.start, n.region.end))
            .collect();
        assert_eq!(
            flat,
            vec![
                ("chr2".into(), 0, 6),
                ("chr1".into(), 3, 8),
                ("chr1".into(), 3, 8),
            ]
        );
    }

    #[test]
    fn load_bed_clamps_end_and_reports_line_numbers() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("r.bed");
        std::fs::write(&path, "chr1\t5\t9999\n").unwrap();
        let regions = load_bed_regions(&path, &genome()).unwrap();
        assert_eq!(regions[0].region.end, 10);

        std::fs::write(&path, "chr1\t3\t8\nchrX\t0\t5\n").unwrap();
        let err = load_bed_regions(&path, &genome()).unwrap_err();
        assert!(format!("{err:#}").contains("line 2"), "{err:#}");

        std::fs::write(&path, "chr1\tabc\t8\n").unwrap();
        let err = load_bed_regions(&path, &genome()).unwrap_err();
        assert!(format!("{err:#}").contains("line 1"), "{err:#}");

        std::fs::write(&path, "chr1\t5\t5\n").unwrap();
        assert!(load_bed_regions(&path, &genome()).is_err());
    }

    #[test]
    fn load_bed_reads_gzip_input() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("r.bed.gz");
        let file = std::fs::File::create(&path).unwrap();
        let mut enc = flate2::write::GzEncoder::new(file, flate2::Compression::default());
        enc.write_all(b"chr1\t0\t5\n").unwrap();
        enc.finish().unwrap();
        let regions = load_bed_regions(&path, &genome()).unwrap();
        assert_eq!(regions[0].region.end, 5);
    }
}
