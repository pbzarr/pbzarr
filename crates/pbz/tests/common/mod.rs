use std::ffi::OsString;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use ndarray::{Array1, Array2};
use noodles_bgzf as bgzf;
use noodles_core::Position;
use noodles_csi::{self as csi, binning_index::index::reference_sequence::bin::Chunk};
use noodles_tabix as tabix;
use pbzarr::io::Dtype;
use pbzarr::{Contig, Genome, PbzStore, TrackConfig};

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

#[allow(dead_code)]
pub fn build_store(dir: &Path) -> PathBuf {
    let path = dir.join("store.pbz");
    let mut store = PbzStore::create(&path).unwrap();
    store
        .create_track("depth", genome(), TrackConfig::new(Dtype::I32))
        .unwrap();
    store
        .create_track(
            "af",
            genome(),
            TrackConfig::new(Dtype::F32)
                .columns(vec!["s1".into(), "s2".into()])
                .column_dim("sample"),
        )
        .unwrap();
    let reference = genome();
    let depth = store.track("depth").unwrap();
    let region = reference.resolve(&"chr1".parse().unwrap()).unwrap();
    depth
        .write_region(
            &region,
            Array1::from(vec![5i32, 5, 5, 7, 7, 0, 0, 0, 0, 0]).into_dyn(),
        )
        .unwrap();
    let af = store.track("af").unwrap();
    let region = reference.resolve(&"chr1:2-6".parse().unwrap()).unwrap();
    af.write_region(
        &region,
        Array2::from_shape_vec((4, 2), vec![0.5f32, 1.0, 0.5, 1.0, 0.25, 1.0, 0.25, 1.0])
            .unwrap()
            .into_dyn(),
    )
    .unwrap();
    path
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

use ndarray::ArrayView1;
use pbzarr::dataset::{Dataset, Reference};
use pbzarr::import::Import;
use pbzarr::io::{OutputSchema, OutputSinkMut, ValueReader};

#[derive(Clone)]
enum Column {
    Ints(Vec<Vec<i32>>),
    Floats(Vec<Vec<f32>>),
}

/// Serves one pre-built value per position, per contig in genome order.
#[derive(Clone)]
struct ColumnReader {
    genome: Genome,
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
    fn contigs(&self) -> &Genome {
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

/// A format-0.5 root Dataset holding the same values as [`build_store`].
#[allow(dead_code)]
pub fn build_root_store(dir: &Path) -> PathBuf {
    let path = dir.join("root.pbz");
    let reference = Reference::new(genome().contigs().to_vec()).unwrap();
    let mut dataset = Dataset::create(&path, reference).unwrap();
    let depth = Column::Ints(vec![vec![5, 5, 5, 7, 7, 0, 0, 0, 0, 0], vec![0; 6]]);
    Import::from_readers(vec![ColumnReader::new(depth)])
        .unwrap()
        .into_dataset(
            &mut dataset,
            vec![("depth".into(), TrackConfig::new(Dtype::I32).chunk_size(4))],
        )
        .run()
        .unwrap();
    let n = f32::NAN;
    let s1 = Column::Floats(vec![
        vec![n, n, 0.5, 0.5, 0.25, 0.25, n, n, n, n],
        vec![n; 6],
    ]);
    let s2 = Column::Floats(vec![vec![n, n, 1.0, 1.0, 1.0, 1.0, n, n, n, n], vec![n; 6]]);
    // One sample per inner chunk, sharded.
    let af = TrackConfig::new(Dtype::F32)
        .columns(vec!["s1".into(), "s2".into()])
        .column_dim("sample")
        .chunk_size(4)
        .column_chunk_size(1)
        .shard_size(8)
        .shard_column_size(1);
    Import::from_readers(vec![ColumnReader::new(s1), ColumnReader::new(s2)])
        .unwrap()
        .into_dataset(&mut dataset, vec![("af".into(), af)])
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
