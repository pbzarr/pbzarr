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
