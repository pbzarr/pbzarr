use std::ffi::OsString;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use ndarray::ArrayView1;
use noodles_bgzf as bgzf;
use noodles_core::Position;
use noodles_csi::{self as csi, binning_index::index::reference_sequence::bin::Chunk};
use noodles_tabix as tabix;
use pbzarr::import::Import;
use pbzarr::io::{Dtype, OutputSchema, OutputSinkMut, ValueReader};
use pbzarr::{Contig, Dataset, Fill, Reference, TrackConfig};

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

/// Write a BGZF-compressed BED with a `#`-prefixed header and a sibling `.tbi`.
/// `rows` are `(chrom, start, end, data_cells)` and must be coordinate-sorted.
#[allow(dead_code)]
pub fn write_bed_bgzip_tabix(
    dir: &Path,
    name: &str,
    header: &[&str],
    rows: &[(&str, u64, u64, &[&str])],
) -> PathBuf {
    let gz = dir.join(format!("{name}.bed.gz"));
    let mut writer = File::create(&gz).map(bgzf::io::Writer::new).unwrap();
    writeln!(writer, "#{}", header.join("\t")).unwrap();

    let mut indexer = tabix::index::Indexer::default();
    indexer.set_header(csi::binning_index::index::header::Builder::bed().build());

    let mut start_position = writer.virtual_position();
    for (chrom, start, end, cells) in rows {
        write!(writer, "{chrom}\t{start}\t{end}").unwrap();
        for cell in *cells {
            write!(writer, "\t{cell}").unwrap();
        }
        writeln!(writer).unwrap();

        let end_position = writer.virtual_position();
        indexer
            .add_record(
                chrom,
                Position::try_from((start + 1) as usize).unwrap(),
                Position::try_from(*end as usize).unwrap(),
                Chunk::new(start_position, end_position),
            )
            .unwrap();
        start_position = end_position;
    }
    writer.finish().unwrap();

    tabix::fs::write(dir.join(format!("{name}.bed.gz.tbi")), &indexer.build()).unwrap();
    gz
}

pub fn run_pbz(args: impl IntoIterator<Item = OsString>) -> Output {
    Command::new(env!("CARGO_BIN_EXE_pbz"))
        .args(args)
        .output()
        .unwrap()
}

#[allow(dead_code)]
pub fn stdout_of(output: &Output) -> String {
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout.clone()).unwrap()
}

#[derive(Clone)]
enum Column {
    Ints(Vec<Vec<i32>>),
    Floats(Vec<Vec<f32>>),
}

/// Serves one pre-built value per position, per contig in genome order.
#[derive(Clone)]
struct ColumnReader {
    genome: Reference,
    schema: OutputSchema,
    column: Column,
}

impl ColumnReader {
    fn new(column: Column) -> Self {
        let dtype = match &column {
            Column::Ints(_) => Dtype::I32,
            Column::Floats(_) => Dtype::F32,
        };
        Self {
            genome: genome(),
            schema: OutputSchema::single("value", dtype),
            column,
        }
    }
}

impl ValueReader for ColumnReader {
    fn contigs(&self) -> &Reference {
        &self.genome
    }
    fn output_schema(&self) -> &OutputSchema {
        &self.schema
    }
    fn fork(&self) -> pbzarr::io::Result<Self> {
        Ok(self.clone())
    }
    fn read_into(
        &mut self,
        contig: &str,
        start: u64,
        end: u64,
        outputs: &mut [OutputSinkMut<'_>],
    ) -> pbzarr::io::Result<()> {
        let index = self.genome.id(contig).expect("fixture contig").as_usize();
        let span = start as usize..end as usize;
        match &self.column {
            Column::Ints(per_contig) => outputs[0]
                .as_i32_mut()?
                .assign(&ArrayView1::from(&per_contig[index][span])),
            Column::Floats(per_contig) => outputs[0]
                .as_f32_mut()?
                .assign(&ArrayView1::from(&per_contig[index][span])),
        }
        Ok(())
    }
}

/// A store with an integer `depth` track and a two-sample float `af` track.
#[allow(dead_code)]
pub fn build_store(dir: &Path) -> PathBuf {
    let path = dir.join("store.pbz");
    let mut dataset = Dataset::create(&path, genome()).unwrap();
    let depth = Column::Ints(vec![vec![5, 5, 5, 7, 7, 0, 0, 0, 0, 0], vec![0; 6]]);
    Import::from_readers(vec![ColumnReader::new(depth)])
        .unwrap()
        .into_dataset(
            &mut dataset,
            vec![(
                "depth".into(),
                TrackConfig::new(Dtype::I32).chunk(4).unsharded(),
            )],
        )
        .run()
        .unwrap();
    let n = f32::NAN;
    let s1 = Column::Floats(vec![
        vec![n, n, 0.5, 0.5, 0.25, 0.25, n, n, n, n],
        vec![n; 6],
    ]);
    let s2 = Column::Floats(vec![vec![n, n, 1.0, 1.0, 1.0, 1.0, n, n, n, n], vec![n; 6]]);
    // One sample per inner chunk, sharded, with the NaN `_FillValue` Xarray
    // writes for float variables.
    let mut af = TrackConfig::new(Dtype::F32)
        .columns(vec!["s1".into(), "s2".into()])
        .column_dim("sample")
        .chunk(4)
        .shard(2)
        .shard_columns(1);
    af.extra.insert("_FillValue".into(), "AAAAAAAA+H8=".into());
    Import::from_readers(vec![ColumnReader::new(s1), ColumnReader::new(s2)])
        .unwrap()
        .into_dataset(&mut dataset, vec![("af".into(), af)])
        .readers_as_columns()
        .run()
        .unwrap();
    path
}

/// A store with a two-sample int32 `score` track whose fill is the null.
#[allow(dead_code)]
pub fn build_null_store(dir: &Path) -> PathBuf {
    let path = dir.join("null.pbz");
    let mut dataset = Dataset::create(&path, genome()).unwrap();
    let n = i32::MAX;
    let s1 = Column::Ints(vec![vec![n, n, 4, 4, n, n, n, n, n, n], vec![n; 6]]);
    let s2 = Column::Ints(vec![vec![n, n, 2, 2, 6, n, n, n, n, n], vec![n; 6]]);
    let score = TrackConfig::new(Dtype::I32)
        .columns(vec!["s1".into(), "s2".into()])
        .column_dim("sample")
        .chunk(4)
        .unsharded()
        .fill(Fill::Na);
    Import::from_readers(vec![ColumnReader::new(s1), ColumnReader::new(s2)])
        .unwrap()
        .into_dataset(&mut dataset, vec![("score".into(), score)])
        .readers_as_columns()
        .run()
        .unwrap();
    path
}

/// Hand-written metadata for a shape no importer produces.
#[allow(dead_code)]
pub fn add_rank_three_variable(store: &Path, name: &str) {
    let dir = store.join(name);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("zarr.json"),
        r#"{
          "zarr_format": 3,
          "node_type": "array",
          "shape": [16, 2, 2],
          "data_type": "uint8",
          "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [16, 2, 2]}},
          "chunk_key_encoding": {"name": "default"},
          "fill_value": 0,
          "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
          "attributes": {},
          "dimension_names": ["position", "sample", "threshold"]
        }"#,
    )
    .unwrap();
}
