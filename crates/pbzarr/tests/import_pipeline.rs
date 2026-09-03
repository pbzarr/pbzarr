use pbzarr::import::{Import, PipelineOptions};
use pbzarr::io::{Dtype, OutputSchema, OutputSinkMut, ValueReader};
use pbzarr::{Contig, Genome, PbzStore, Region, TrackConfig};
use tempfile::TempDir;

/// A reader that fills every position with a constant value.
struct ConstReader {
    genome: Genome,
    schema: OutputSchema,
    val: u32,
}

impl ConstReader {
    fn new(genome: Genome, val: u32) -> Self {
        Self {
            genome,
            schema: OutputSchema::single("value", Dtype::U32),
            val,
        }
    }
}

impl ValueReader for ConstReader {
    fn contigs(&self) -> &Genome {
        &self.genome
    }

    fn output_schema(&self) -> &OutputSchema {
        &self.schema
    }

    fn read_into(
        &mut self,
        _contig: &str,
        _start: u64,
        _end: u64,
        outputs: &mut [OutputSinkMut<'_>],
    ) -> pbzarr::io::Result<()> {
        outputs[0].as_u32_mut()?.fill(self.val);
        Ok(())
    }

    fn fork(&self) -> pbzarr::io::Result<Self> {
        Ok(Self::new(self.genome.clone(), self.val))
    }
}

#[test]
fn pipeline_writes_constants_into_cohort_track() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("t.pbz");
    let genome = Genome::new(vec![
        Contig {
            name: "chr1".into(),
            length: 5_000,
        },
        Contig {
            name: "chr2".into(),
            length: 1_500,
        },
    ])
    .unwrap();

    let mut store = PbzStore::create(&path).unwrap();
    store
        .create_track(
            "depth",
            genome.clone(),
            TrackConfig::new(Dtype::U32)
                .columns(vec!["A".into(), "B".into()])
                .column_dim("sample"),
        )
        .unwrap();

    let readers = vec![
        ConstReader::new(genome.clone(), 7),
        ConstReader::new(genome.clone(), 13),
    ];

    let track = store.track("depth").unwrap();
    let report = Import::from_readers(readers)
        .unwrap()
        .into_track(track)
        .readers_as_columns()
        .run()
        .unwrap();
    assert!(report.bytes_written > 0);
    assert_eq!(report.contigs_written, 2);

    for (name, len) in [("chr1", 5_000usize), ("chr2", 1_500)] {
        let region = Region {
            contig: genome.id(name).unwrap(),
            start: 0,
            end: len as u64,
        };
        let got = store
            .track("depth")
            .unwrap()
            .read_region::<u32>(&region)
            .unwrap();
        let got2 = got.into_dimensionality::<ndarray::Ix2>().unwrap();
        for i in 0..len {
            assert_eq!(got2[[i, 0]], 7, "{name} col 0 at i={i}");
            assert_eq!(got2[[i, 1]], 13, "{name} col 1 at i={i}");
        }
    }
}

#[test]
fn cohort_pipeline_tiles_columns_with_multiple_workers() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("tiled.pbz");
    let genome = Genome::new(vec![Contig {
        name: "chr1".into(),
        length: 8,
    }])
    .unwrap();
    let mut store = PbzStore::create(&path).unwrap();
    store
        .create_track(
            "depth",
            genome.clone(),
            TrackConfig::new(Dtype::U32)
                .columns(vec!["a".into(), "b".into(), "c".into(), "d".into()])
                .chunk_size(4)
                .column_chunk_size(2),
        )
        .unwrap();

    let readers = [7, 8, 9, 10]
        .into_iter()
        .map(|val| ConstReader::new(genome.clone(), val))
        .collect();
    let report = Import::from_readers(readers)
        .unwrap()
        .into_track(store.track("depth").unwrap())
        .readers_as_columns()
        .options(PipelineOptions {
            workers: 2,
            ..PipelineOptions::default()
        })
        .run()
        .unwrap();

    assert_eq!(report.tasks_completed, 4);
    let region = Region {
        contig: genome.id("chr1").unwrap(),
        start: 0,
        end: 8,
    };
    let values = store
        .track("depth")
        .unwrap()
        .read_region::<u32>(&region)
        .unwrap()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();
    for row in values.rows() {
        assert_eq!(row.as_slice().unwrap(), &[7, 8, 9, 10]);
    }
}

/// Fills each position with `contig_index * 1_000_000 + local_position`, so a
/// read-back reveals whether a span handed the reader the right contig name and
/// local range.
struct RampReader {
    genome: Genome,
    schema: OutputSchema,
}

impl RampReader {
    fn new(genome: Genome) -> Self {
        Self {
            genome,
            schema: OutputSchema::single("value", Dtype::U32),
        }
    }
}

impl ValueReader for RampReader {
    fn contigs(&self) -> &Genome {
        &self.genome
    }

    fn output_schema(&self) -> &OutputSchema {
        &self.schema
    }

    fn read_into(
        &mut self,
        contig_name: &str,
        start: u64,
        end: u64,
        outputs: &mut [OutputSinkMut<'_>],
    ) -> pbzarr::io::Result<()> {
        let cid = self
            .genome
            .contigs()
            .iter()
            .position(|c| c.name == contig_name)
            .unwrap() as u32;
        let dst = outputs[0].as_u32_mut()?;
        for (row, pos) in (start..end).enumerate() {
            dst[row] = cid * 1_000_000 + pos as u32;
        }
        Ok(())
    }

    fn fork(&self) -> pbzarr::io::Result<Self> {
        Ok(Self::new(self.genome.clone()))
    }
}

#[test]
fn pipeline_span_straddling_contig_boundary_maps_local_coords() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("r.pbz");
    let genome = Genome::new(vec![
        Contig {
            name: "chr1".into(),
            length: 2_000,
        },
        Contig {
            name: "chr2".into(),
            length: 1_500,
        },
    ])
    .unwrap();

    let mut store = PbzStore::create(&path).unwrap();
    // chunk_size 700 → chunks at 0,700,1400,2100,2800; [1400,2100) straddles
    // the chr1/chr2 boundary at flat 2000.
    store
        .create_track(
            "depth",
            genome.clone(),
            TrackConfig::new(Dtype::U32).chunk_size(700),
        )
        .unwrap();

    let readers = vec![RampReader::new(genome.clone())];
    let track = store.track("depth").unwrap();
    Import::from_readers(readers)
        .unwrap()
        .into_track(track)
        .run()
        .unwrap();

    for (name, cid, len) in [("chr1", 0u32, 2_000usize), ("chr2", 1, 1_500)] {
        let region = Region {
            contig: genome.id(name).unwrap(),
            start: 0,
            end: len as u64,
        };
        let got = store
            .track("depth")
            .unwrap()
            .read_region::<u32>(&region)
            .unwrap();
        let got1 = got.into_dimensionality::<ndarray::Ix1>().unwrap();
        for pos in 0..len {
            assert_eq!(got1[pos], cid * 1_000_000 + pos as u32, "{name} pos={pos}");
        }
    }
}
