//! `pbz info`: a summary of the store layout.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};
use pbzarr::{Dataset, FORMAT_VERSION, Role, VariableInfo, is_zip_store};
use serde_json::{Value, json};

const CONTIGS_SHOWN: usize = 10;

pub(crate) fn run_info(path: &Path, as_json: bool) -> Result<()> {
    let store = crate::view::open_store(path)?;
    let report = report(path, &store)?;
    let mut out = std::io::stdout().lock();
    if as_json {
        serde_json::to_writer_pretty(&mut out, &report)?;
        writeln!(out)?;
    } else {
        write_text(&mut out, &report)?;
    }
    out.flush()?;
    Ok(())
}

fn report(path: &Path, store: &Dataset) -> Result<Value> {
    let reference = store
        .reference()
        .context("read the store reference (run pbz validate for details)")?;
    let contigs: Vec<Value> = reference
        .contigs()
        .iter()
        .map(|c| json!({"name": c.name, "length": c.length}))
        .collect();
    let mut dimensions = BTreeMap::new();
    let mut variables = serde_json::Map::new();
    for (name, info) in store.variables() {
        for (dim, &size) in info.dims.iter().zip(&info.shape) {
            dimensions.entry(dim.clone()).or_insert(size);
        }
        let stored = store.stored_size(name)?;
        let (chunks, shards) = grid(info);
        variables.insert(
            name.clone(),
            json!({
                "role": role(info),
                "dtype": dtype(info),
                "dims": info.dims,
                "shape": info.shape,
                "chunks": chunks,
                "shards": shards,
                "codecs": codecs(info),
                "fill_value": info.metadata["fill_value"],
                "stored_bytes": stored.map(|s| s.bytes),
                "objects": stored.map(|s| s.objects),
            }),
        );
    }
    Ok(json!({
        "path": path.display().to_string(),
        "kind": if is_zip_store(path) { "zip" } else { "directory" },
        "format_version": FORMAT_VERSION,
        "contigs": contigs,
        "genome_length": reference.contigs().iter().map(|c| c.length).sum::<u64>(),
        "dimensions": dimensions,
        "variables": variables,
    }))
}

fn role(info: &VariableInfo) -> &'static str {
    if info.is_track() {
        "track"
    } else if info.role == Role::Coordinate {
        "coordinate"
    } else {
        "other"
    }
}

fn dtype(info: &VariableInfo) -> &str {
    if info.metadata["attributes"]["dtype"] == "bool" {
        return "bool";
    }
    info.metadata["data_type"].as_str().unwrap_or("?")
}

fn sharding(info: &VariableInfo) -> Option<&Value> {
    info.metadata["codecs"]
        .as_array()?
        .iter()
        .find(|c| c["name"] == "sharding_indexed")
        .map(|c| &c["configuration"])
}

/// The inner chunk shape and, for a sharded array, the shard shape.
fn grid(info: &VariableInfo) -> (Value, Value) {
    let outer = info.metadata["chunk_grid"]["configuration"]["chunk_shape"].clone();
    match sharding(info) {
        Some(config) => (config["chunk_shape"].clone(), outer),
        None => (outer, Value::Null),
    }
}

/// Codec names of the chunk codec chain, with the compression level when the
/// codec has one.
fn codecs(info: &VariableInfo) -> Vec<String> {
    let chain = sharding(info)
        .map(|config| &config["codecs"])
        .unwrap_or(&info.metadata["codecs"]);
    chain
        .as_array()
        .map(|codecs| {
            codecs
                .iter()
                .map(|c| {
                    let name = c["name"].as_str().unwrap_or("?");
                    match c["configuration"]
                        .get("level")
                        .or(c["configuration"].get("clevel"))
                    {
                        Some(level) => format!("{name}({level})"),
                        None => name.to_owned(),
                    }
                })
                .collect()
        })
        .unwrap_or_default()
}

fn shape_text(value: &Value) -> String {
    match value.as_array() {
        Some(items) if !items.is_empty() => items
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join("x"),
        _ => "-".into(),
    }
}

fn bytes_text(value: &Value) -> String {
    let Some(bytes) = value.as_u64() else {
        return "-".into();
    };
    let units = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut size = bytes as f64;
    let mut unit = 0;
    while size >= 1024.0 && unit + 1 < units.len() {
        size /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{size:.1} {}", units[unit])
    }
}

fn write_table(out: &mut impl Write, rows: &[Vec<String>]) -> std::io::Result<()> {
    let columns = rows.first().map_or(0, Vec::len);
    let widths: Vec<usize> = (0..columns)
        .map(|i| rows.iter().map(|r| r[i].len()).max().unwrap_or(0))
        .collect();
    for row in rows {
        let line: Vec<String> = row
            .iter()
            .zip(&widths)
            .map(|(cell, width)| format!("{cell:<width$}"))
            .collect();
        writeln!(out, "  {}", line.join("  ").trim_end())?;
    }
    Ok(())
}

fn write_text(out: &mut impl Write, report: &Value) -> std::io::Result<()> {
    writeln!(
        out,
        "store: {} ({} store, format {})",
        report["path"].as_str().unwrap_or(""),
        report["kind"].as_str().unwrap_or(""),
        report["format_version"].as_str().unwrap_or("")
    )?;
    let contigs = report["contigs"].as_array().cloned().unwrap_or_default();
    writeln!(
        out,
        "\nreference: {} contig{}, {} bases",
        contigs.len(),
        if contigs.len() == 1 { "" } else { "s" },
        report["genome_length"]
    )?;
    let mut rows: Vec<Vec<String>> = contigs
        .iter()
        .take(CONTIGS_SHOWN)
        .map(|c| {
            vec![
                c["name"].as_str().unwrap_or("").to_owned(),
                c["length"].to_string(),
            ]
        })
        .collect();
    if contigs.len() > CONTIGS_SHOWN {
        rows.push(vec![
            format!("({} more)", contigs.len() - CONTIGS_SHOWN),
            String::new(),
        ]);
    }
    write_table(out, &rows)?;

    writeln!(out, "\ndimensions:")?;
    let rows: Vec<Vec<String>> = report["dimensions"]
        .as_object()
        .into_iter()
        .flatten()
        .map(|(dim, size)| vec![dim.clone(), size.to_string()])
        .collect();
    write_table(out, &rows)?;

    writeln!(out, "\nvariables:")?;
    let mut rows = vec![
        [
            "name", "role", "dtype", "dims", "shape", "chunk", "shard", "codecs", "fill", "stored",
            "objects",
        ]
        .map(String::from)
        .to_vec(),
    ];
    for (name, v) in report["variables"].as_object().into_iter().flatten() {
        let dims: Vec<&str> = v["dims"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect();
        let codecs: Vec<&str> = v["codecs"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect();
        rows.push(vec![
            name.clone(),
            v["role"].as_str().unwrap_or("").to_owned(),
            v["dtype"].as_str().unwrap_or("").to_owned(),
            if dims.is_empty() {
                "-".into()
            } else {
                dims.join(",")
            },
            shape_text(&v["shape"]),
            shape_text(&v["chunks"]),
            shape_text(&v["shards"]),
            if codecs.is_empty() {
                "-".into()
            } else {
                codecs.join(",")
            },
            v["fill_value"].to_string(),
            bytes_text(&v["stored_bytes"]),
            v["objects"].as_u64().map_or("-".into(), |n| n.to_string()),
        ]);
    }
    write_table(out, &rows)
}
