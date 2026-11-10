//! Region selection shared by `pbz view` and `pbz stat`.

use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, bail};
use pbzarr::{Reference, Region, RegionQuery};

#[derive(Debug)]
pub(crate) struct NamedRegion {
    pub chrom: String,
    pub region: Region,
}

/// `-r` regions in the order given, then the `-R` file in file order.
/// With neither, one region per contig.
pub(crate) fn select_regions(
    genome: &Reference,
    queries: &[RegionQuery],
    bed: Option<&PathBuf>,
) -> Result<Vec<NamedRegion>> {
    if queries.is_empty() && bed.is_none() {
        return Ok(genome
            .iter()
            .map(|(id, contig)| NamedRegion {
                chrom: contig.name.clone(),
                region: Region {
                    contig: id,
                    start: 0,
                    end: contig.length,
                },
            })
            .collect());
    }
    let mut out = Vec::new();
    for query in queries {
        let region = genome
            .region(query)
            .with_context(|| format!("region {query}"))?;
        out.push(NamedRegion {
            chrom: query.contig.clone(),
            region,
        });
    }
    if let Some(path) = bed {
        out.extend(load_bed_regions(path, genome)?);
    }
    Ok(out)
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

pub(crate) fn load_bed_regions(path: &Path, genome: &Reference) -> Result<Vec<NamedRegion>> {
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
            .region(&query)
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

    fn genome() -> Reference {
        Reference::new(vec![
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

    fn flat(regions: &[NamedRegion]) -> Vec<(String, u64, u64)> {
        regions
            .iter()
            .map(|n| (n.chrom.clone(), n.region.start, n.region.end))
            .collect()
    }

    #[test]
    fn region_flags_come_before_the_bed_file_and_refuse_ends_past_the_contig() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("r.bed");
        std::fs::write(&path, "chr1\t0\t2\n").unwrap();
        let queries: Vec<RegionQuery> = ["chr2:1-4", "chr1"]
            .iter()
            .map(|q| q.parse().unwrap())
            .collect();
        let regions = select_regions(&genome(), &queries, Some(&path)).unwrap();
        assert_eq!(
            flat(&regions),
            vec![
                ("chr2".into(), 1, 4),
                ("chr1".into(), 0, 10),
                ("chr1".into(), 0, 2),
            ]
        );
        let past_end = ["chr2:0-7".parse().unwrap()];
        let err = select_regions(&genome(), &past_end, None).unwrap_err();
        assert!(format!("{err:#}").contains("outside length 6"), "{err:#}");
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
        assert_eq!(
            flat(&regions),
            vec![
                ("chr2".into(), 0, 6),
                ("chr1".into(), 3, 8),
                ("chr1".into(), 3, 8),
            ]
        );
    }

    #[test]
    fn load_bed_refuses_end_past_contig_and_reports_line_numbers() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("r.bed");
        std::fs::write(&path, "chr1\t5\t11\n").unwrap();
        let err = load_bed_regions(&path, &genome()).unwrap_err();
        assert!(format!("{err:#}").contains("outside length 10"), "{err:#}");

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
        // Two finished members, as bgzip produces; a single-member decoder
        // would silently drop chr2.
        let mut bytes = Vec::new();
        for line in [&b"chr1\t0\t5\n"[..], &b"chr2\t1\t4\n"[..]] {
            let mut enc = flate2::write::GzEncoder::new(&mut bytes, flate2::Compression::default());
            enc.write_all(line).unwrap();
            enc.finish().unwrap();
        }
        std::fs::write(&path, &bytes).unwrap();
        let regions = load_bed_regions(&path, &genome()).unwrap();
        assert_eq!(
            flat(&regions),
            vec![("chr1".into(), 0, 5), ("chr2".into(), 1, 4)]
        );
    }
}
