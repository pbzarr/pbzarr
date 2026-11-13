//! `pbz validate`: check a store against the PBZ format rules.

use std::io::Write;
use std::path::Path;

use anyhow::Result;
use pbzarr::{Dataset, PbzError, ValidationIssue, ValidationStatus};
use serde_json::json;

/// Check the store, print the result, and return true when it is valid.
pub(crate) fn run_validate(path: &Path, as_json: bool) -> Result<bool> {
    crate::view::require_store(path)?;
    let (status, issues) = check(path)?;
    let mut out = std::io::stdout().lock();
    if as_json {
        let issues: Vec<_> = issues
            .iter()
            .map(|i| json!({"array": i.array, "rule": i.rule, "message": i.message}))
            .collect();
        writeln!(out, "{}", json!({"status": status, "issues": issues}))?;
    } else {
        let store = path.display();
        match status {
            "valid" => writeln!(out, "{store}: valid")?,
            "incomplete" => writeln!(out, "{store}: not all variables could be checked")?,
            "unsupported_version" => writeln!(out, "{store}: format version not supported")?,
            _ => writeln!(out, "{store}: not valid")?,
        }
        for issue in &issues {
            match &issue.array {
                Some(array) => writeln!(out, "  {array}: {}", issue.message)?,
                None => writeln!(out, "  {}", issue.message)?,
            }
        }
    }
    out.flush()?;
    Ok(status == "valid")
}

fn store_issue(rule: &str, message: String) -> Vec<ValidationIssue> {
    vec![ValidationIssue {
        array: None,
        rule: rule.into(),
        message,
    }]
}

fn check(path: &Path) -> Result<(&'static str, Vec<ValidationIssue>)> {
    let store = match Dataset::open_read_only(path) {
        Ok(store) => store,
        Err(e @ (PbzError::UnsupportedVersion(_) | PbzError::LegacyFormat)) => {
            return Ok(("unsupported_version", store_issue("R1", e.to_string())));
        }
        Err(PbzError::InvalidStore(message) | PbzError::Metadata(message)) => {
            return Ok(("invalid", store_issue("R1", message)));
        }
        Err(e) => return Err(e.into()),
    };
    let report = store.validate()?;
    let status = match report.status {
        ValidationStatus::Valid => "valid",
        ValidationStatus::Invalid => "invalid",
        ValidationStatus::Incomplete => "incomplete",
    };
    Ok((status, report.issues))
}
