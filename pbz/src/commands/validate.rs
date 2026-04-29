use color_eyre::Result;
use pbzarr::{PERBASE_ZARR_VERSION, PbzError, PbzStore};
use serde::Serialize;
use std::fs;
use std::process;

use crate::cli::ValidateArgs;

// Stable JSON-schema codes. Centralized so adding a code is mechanical
// and a typo at a use site breaks compilation.
const CODE_MISSING_ROOT_ATTR: &str = "missing_root_attr";
const CODE_OPEN_FAILED: &str = "open_failed";
const CODE_TRACK_OPEN_FAILED: &str = "track_open_failed";
const CODE_TRACKS_LISTING_FAILED: &str = "tracks_listing_failed";
const CODE_UNKNOWN_TRACK_GROUP: &str = "unknown_track_group";
const CODE_TRACKS_DIR_UNREADABLE: &str = "tracks_dir_unreadable";
const CODE_TRACK_ZARR_JSON_MALFORMED: &str = "track_zarr_json_malformed";
const CODE_TRACK_ZARR_JSON_UNREADABLE: &str = "track_zarr_json_unreadable";

#[derive(Serialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum Level {
    Info,
    Warn,
    Error,
}

#[derive(Serialize)]
struct Finding {
    level: Level,
    code: &'static str,
    message: String,
    path: String,
}

#[derive(Serialize)]
struct Report {
    findings: Vec<Finding>,
    summary: Summary,
}

#[derive(Serialize)]
struct Summary {
    error: usize,
    warn: usize,
    info: usize,
}

pub fn run(args: ValidateArgs) -> Result<()> {
    let mut findings: Vec<Finding> = Vec::new();

    let open_result = PbzStore::open(&args.store);
    if let Err(e) = &open_result {
        // PbzError::Metadata is constructed only for the two root-attribute
        // checks in PbzStore::open, so matching the variant is precise and
        // stable across error-message rewordings.
        let code = match e {
            PbzError::Metadata(_) => CODE_MISSING_ROOT_ATTR,
            _ => CODE_OPEN_FAILED,
        };
        findings.push(Finding {
            level: Level::Error,
            code,
            message: e.to_string(),
            path: args.store.display().to_string(),
        });
    }

    if let Ok(store) = open_result {
        match store.tracks() {
            Ok(tracks) => {
                for tname in tracks {
                    let opened = store.track(&tname);
                    if let Err(e) = &opened {
                        findings.push(Finding {
                            level: Level::Error,
                            code: CODE_TRACK_OPEN_FAILED,
                            message: e.to_string(),
                            path: format!("/tracks/{tname}"),
                        });
                    }
                }
            }
            Err(e) => findings.push(Finding {
                level: Level::Error,
                code: CODE_TRACKS_LISTING_FAILED,
                message: e.to_string(),
                path: "/tracks".into(),
            }),
        }

        let tracks_root = args.store.join("tracks");
        if tracks_root.is_dir() {
            match fs::read_dir(&tracks_root) {
                Ok(entries) => {
                    for entry in entries.flatten() {
                        let p = entry.path();
                        if !p.is_dir() {
                            continue;
                        }
                        let name = p.file_name().unwrap().to_string_lossy().to_string();
                        let zarr_json = p.join("zarr.json");
                        if !zarr_json.exists() {
                            continue;
                        }
                        let raw_str = match fs::read_to_string(&zarr_json) {
                            Ok(s) => s,
                            Err(e) => {
                                findings.push(Finding {
                                    level: Level::Error,
                                    code: CODE_TRACK_ZARR_JSON_UNREADABLE,
                                    message: format!("could not read {}: {e}", zarr_json.display()),
                                    path: format!("/tracks/{name}"),
                                });
                                continue;
                            }
                        };
                        let raw: serde_json::Value = match serde_json::from_str(&raw_str) {
                            Ok(v) => v,
                            Err(e) => {
                                findings.push(Finding {
                                    level: Level::Error,
                                    code: CODE_TRACK_ZARR_JSON_MALFORMED,
                                    message: format!(
                                        "malformed JSON in {}: {e}",
                                        zarr_json.display()
                                    ),
                                    path: format!("/tracks/{name}"),
                                });
                                continue;
                            }
                        };
                        let has_attr = raw
                            .get("attributes")
                            .and_then(|a| a.get("perbase_zarr_track"))
                            .is_some();
                        if !has_attr {
                            findings.push(Finding {
                                level: Level::Warn,
                                code: CODE_UNKNOWN_TRACK_GROUP,
                                message: format!(
                                    "group '/tracks/{name}' lacks perbase_zarr_track attribute"
                                ),
                                path: format!("/tracks/{name}"),
                            });
                        }
                    }
                }
                Err(e) => {
                    findings.push(Finding {
                        level: Level::Error,
                        code: CODE_TRACKS_DIR_UNREADABLE,
                        message: format!("could not read tracks directory: {e}"),
                        path: "/tracks".into(),
                    });
                }
            }
        }
    }

    let summary = Summary {
        error: findings.iter().filter(|f| f.level == Level::Error).count(),
        warn: findings.iter().filter(|f| f.level == Level::Warn).count(),
        info: findings.iter().filter(|f| f.level == Level::Info).count(),
    };

    if args.json {
        let report = Report { findings, summary };
        println!("{}", serde_json::to_string_pretty(&report)?);
        if report.summary.error > 0 {
            process::exit(2);
        }
    } else {
        print_human(&findings, &summary, &args.store, PERBASE_ZARR_VERSION);
        if summary.error > 0 {
            process::exit(2);
        }
    }
    Ok(())
}

fn print_human(findings: &[Finding], summary: &Summary, path: &std::path::Path, version: &str) {
    eprintln!("validate: {} (spec target {version})", path.display());
    if findings.is_empty() {
        eprintln!("OK — no findings");
        return;
    }
    for f in findings {
        eprintln!("[{:?}] {} {} ({})", f.level, f.code, f.message, f.path);
    }
    eprintln!(
        "\nsummary: {} error, {} warn, {} info",
        summary.error, summary.warn, summary.info
    );
}
