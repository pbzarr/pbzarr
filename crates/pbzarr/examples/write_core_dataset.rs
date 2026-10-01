//! Small native-Xarray conformance fixture for the existing Rust import API.
use pbzarr::dataset::{Dataset, Reference};
use pbzarr::import::Import;
use pbzarr::io::{Dtype, OutputSchema, OutputSinkMut, ValueReader};
use pbzarr::{Contig, ExplicitArraySpec, Genome, TrackConfig};

#[derive(Clone, Copy)]
enum Value {
    Count(u32),
    Mask(bool),
}
#[derive(Clone)]
struct ConstantReader {
    genome: Genome,
    schema: OutputSchema,
    value: Value,
}
impl ConstantReader {
    fn new(value: Value) -> Self {
        Self {
            genome: Genome::new(reference().contigs().to_vec()).unwrap(),
            schema: OutputSchema::single(
                "value",
                match value {
                    Value::Count(_) => Dtype::U32,
                    Value::Mask(_) => Dtype::Bool,
                },
            ),
            value,
        }
    }
}
impl ValueReader for ConstantReader {
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
        _: &str,
        _: u64,
        _: u64,
        outputs: &mut [OutputSinkMut<'_>],
    ) -> pbzarr::io::Result<()> {
        match self.value {
            Value::Count(value) => outputs[0].as_u32_mut()?.fill(value),
            Value::Mask(value) => outputs[0].as_bool_mut()?.fill(value),
        }
        Ok(())
    }
}
fn reference() -> Reference {
    Reference::new(vec![
        Contig {
            name: "chr1".into(),
            length: 5,
        },
        Contig {
            name: "chr2".into(),
            length: 7,
        },
    ])
    .unwrap()
}
fn import_columns(ds: &mut Dataset, name: &str, values: [u32; 2]) -> anyhow::Result<()> {
    let readers = values
        .into_iter()
        .map(|n| ConstantReader::new(Value::Count(n)))
        .collect();
    let config = TrackConfig::new(Dtype::U32)
        .columns(vec!["S1".into(), "S2".into()])
        .column_dim("sample")
        .chunk_size(4)
        .column_chunk_size(1)
        .shard_size(8)
        .shard_column_size(2);
    Import::from_readers(readers)?
        .into_dataset(ds, vec![(name.into(), config)])
        .readers_as_columns()
        .run()?;
    Ok(())
}
fn main() -> anyhow::Result<()> {
    let mut args = std::env::args_os().skip(1);
    let path = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("usage: write_core_dataset STORE [add]"))?;
    let mode = args.next();
    anyhow::ensure!(args.next().is_none(), "too many arguments");
    if let Some(mode) = mode {
        anyhow::ensure!(mode == "add", "expected add");
        return import_columns(&mut Dataset::open(path)?, "added", [12, 20]);
    }
    let mut ds = Dataset::create(path, reference())?;
    import_columns(&mut ds, "depth", [7, 3])?;
    let mut zero = TrackConfig::new(Dtype::U32)
        .chunk_size(4)
        .codecs(ExplicitArraySpec::parse(
            r#"[{"name":"bytes","configuration":{"endian":"little"}},
            {"name":"zstd","configuration":{"level":1,"checksum":false}}]"#,
        )?);
    zero.extra
        .insert("notes".into(), "Zero remains an ordinary value.".into());
    Import::from_readers(vec![ConstantReader::new(Value::Count(0))])?
        .into_dataset(&mut ds, vec![("zero".into(), zero)])
        .run()?;
    Import::from_readers(vec![ConstantReader::new(Value::Mask(false))])?
        .into_dataset(
            &mut ds,
            vec![("mask".into(), TrackConfig::new(Dtype::Bool).chunk_size(4))],
        )
        .run()?;
    Ok(())
}
