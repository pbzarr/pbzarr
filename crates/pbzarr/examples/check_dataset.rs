//! Conformance probe for small draft fixtures; not the production PBZ CLI.
use pbzarr::{
    PbzError,
    dataset::{Dataset, GenomicAxis, ValidationStatus},
};
use serde_json::{Value, json};
use std::{collections::BTreeMap, ops::Range};
use zarrs::array::ElementOwned;

fn raw<T: ElementOwned + serde::Serialize>(
    ds: &Dataset,
    name: &str,
    ranges: &[Range<u64>],
) -> anyhow::Result<Value> {
    let array = ds.read_raw::<T>(name, ranges)?;
    if array.ndim() == 0 {
        Ok(serde_json::to_value(&array[ndarray::IxDyn(&[])])?)
    } else {
        Ok(serde_json::to_value(array.iter().collect::<Vec<_>>())?)
    }
}
fn check(path: &std::path::Path) -> anyhow::Result<Value> {
    let ds = match Dataset::open(path) {
        Ok(ds) => ds,
        Err(PbzError::UnsupportedFormat(version)) => {
            return Ok(json!({"status":"unsupported_version","version":version}));
        }
        Err(PbzError::Metadata(message)) => {
            let rule = message
                .split(':')
                .next()
                .filter(|r| r.len() == 2)
                .unwrap_or("R1");
            return Ok(
                json!({"status":"invalid","issues":[{"array":null,"rule":rule,"message":message}]}),
            );
        }
        Err(e) => return Err(e.into()),
    };
    let validation = ds.validate()?;
    let mut result =
        json!({"status":validation.status,"issues":validation.issues,"variables":ds.variables()});
    if validation.status != ValidationStatus::Valid {
        return Ok(result);
    }
    let mut output = BTreeMap::new();
    let mut bins = BTreeMap::new();
    for (name, info) in ds.variables() {
        let count = info.shape.iter().try_fold(1_u64, |n, &s| n.checked_mul(s));
        anyhow::ensure!(
            count.is_some_and(|n| n <= 1_000_000),
            "probe only reads small fixtures (array {name})"
        );
        let ranges: Vec<Range<u64>> = info.shape.iter().map(|&n| 0..n).collect();
        let dtype = info.metadata["data_type"].as_str().unwrap_or("");
        let value = if dtype == "bool" || info.metadata["attributes"]["dtype"] == "bool" {
            let a = ds.read_bool(name, &ranges)?;
            if a.ndim() == 0 {
                json!(a[ndarray::IxDyn(&[])])
            } else {
                json!(a.iter().collect::<Vec<_>>())
            }
        } else {
            match dtype {
                "uint8" => raw::<u8>(&ds, name, &ranges)?,
                "uint16" => raw::<u16>(&ds, name, &ranges)?,
                "uint32" => raw::<u32>(&ds, name, &ranges)?,
                "uint64" => raw::<u64>(&ds, name, &ranges)?,
                "int8" => raw::<i8>(&ds, name, &ranges)?,
                "int16" => raw::<i16>(&ds, name, &ranges)?,
                "int32" => raw::<i32>(&ds, name, &ranges)?,
                "int64" => raw::<i64>(&ds, name, &ranges)?,
                "float32" => raw::<f32>(&ds, name, &ranges)?,
                "float64" => raw::<f64>(&ds, name, &ranges)?,
                "string" => raw::<String>(&ds, name, &ranges)?,
                other => anyhow::bail!("unsupported probe dtype {other}"),
            }
        };
        output.insert(name, value);
        for dim in &info.dims {
            if let Some(axis @ GenomicAxis::Bin(_)) = GenomicAxis::from_dimension(dim)? {
                let reference = ds.reference()?;
                let intervals: Vec<Value> = (0..reference.extent(&axis)?)
                    .map(|row| {
                        let i = reference
                            .interval(&axis, row)
                            .expect("in-bounds fixture row");
                        json!({"contig":i.contig,"start":i.start,"stop":i.stop})
                    })
                    .collect();
                bins.insert(dim, intervals);
            }
        }
    }
    result["values"] = json!(output);
    result["bins"] = json!(bins);
    result["reference"] = json!(
        ds.reference()?
            .contigs()
            .iter()
            .map(|c| json!({"name":c.name,"length":c.length}))
            .collect::<Vec<_>>()
    );
    if ds
        .variables()
        .get("depth")
        .is_some_and(|v| v.shape == [12, 3])
    {
        result["depth_subset"] = raw::<u16>(&ds, "depth", &[2..4, 1..3])?;
    }
    Ok(result)
}
fn main() -> anyhow::Result<()> {
    let path = std::env::args_os()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("usage: check_dataset STORE"))?;
    println!(
        "{}",
        serde_json::to_string(&check(std::path::Path::new(&path))?)?
    );
    Ok(())
}
