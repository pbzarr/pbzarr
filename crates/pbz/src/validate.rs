//! `pbz validate`: check a store against the PBZ format rules.

use std::io::Write;
use std::path::Path;

use anyhow::{Result, bail};
use pbzarr::{Dataset, PbzError, ValidationIssue, ValidationStatus};
use serde_json::json;

/// Check the store, print the result, and return true when it is valid.
pub(crate) fn run_validate(path: &Path, as_json: bool) -> Result<bool> {
    if !path.exists() {
        bail!("no store at {}", path.display());
    }
    let (status, issues) = check(path)?;
    let mut out = std::io::stdout().lock();
    if as_json {
        let issues: Vec<_> = issues
            .iter()
            .map(|i| json!({"array": i.array, "rule": i.rule, "message": plain(&i.message)}))
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
                Some(array) => writeln!(out, "  {array}: {}", plain(&issue.message))?,
                None => writeln!(out, "  {}", plain(&issue.message))?,
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
        Err(PbzError::InvalidStore(message)) => {
            return Ok(("invalid", store_issue("R1", message)));
        }
        Err(PbzError::Metadata(message)) => {
            let rule = rule_code(&message).unwrap_or("R1").to_owned();
            return Ok(("invalid", store_issue(&rule, message)));
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

/// The rule code that starts a library message, as `R4` in `R4: duplicate
/// contig name`.
fn rule_code(message: &str) -> Option<&str> {
    let (head, _) = message.split_once(": ")?;
    let mut chars = head.chars();
    (chars.next()?.is_ascii_uppercase()
        && head.len() <= 4
        && chars.all(|c| c.is_ascii_uppercase() || c.is_ascii_digit()))
    .then_some(head)
}

fn plain(message: &str) -> &str {
    match rule_code(message) {
        Some(code) => &message[code.len() + 2..],
        None => message,
    }
}

#[cfg(test)]
mod tests {
    use super::plain;

    #[test]
    fn messages_drop_rule_codes() {
        assert_eq!(plain("R4: duplicate contig name"), "duplicate contig name");
        assert_eq!(plain("chr1: bad"), "chr1: bad");
    }
}
