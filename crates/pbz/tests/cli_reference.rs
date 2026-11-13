//! `docs/reference/cli.md` is the `--help` text of every `pbz` command. The
//! test fails when the page is stale. Run it with `UPDATE_CLI_DOCS=1` to
//! write the page again.

use std::path::Path;
use std::process::Command;

fn help(path: &[&str]) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_pbz"))
        .args(path)
        .arg("--help")
        .env_remove("RUST_LOG")
        .output()
        .expect("run pbz");
    assert!(out.status.success(), "pbz {} --help failed", path.join(" "));
    String::from_utf8(out.stdout)
        .expect("utf-8 help")
        .trim_end()
        .to_owned()
}

fn subcommands(help: &str) -> Vec<String> {
    help.lines()
        .skip_while(|l| l.trim() != "Commands:")
        .skip(1)
        .take_while(|l| !l.trim().is_empty())
        .filter_map(|l| l.split_whitespace().next())
        .filter(|name| *name != "help")
        .map(str::to_owned)
        .collect()
}

fn render(path: &mut Vec<String>, page: &mut String) {
    let args: Vec<&str> = path.iter().map(String::as_str).collect();
    let text = help(&args);
    let depth = path.len().min(2) + 2;
    let title = if path.is_empty() {
        "pbz".to_owned()
    } else {
        format!("pbz {}", path.join(" "))
    };
    page.push_str(&format!(
        "\n{} `{title}`\n\n```text\n{text}\n```\n",
        "#".repeat(depth)
    ));
    for sub in subcommands(&text) {
        path.push(sub);
        render(path, page);
        path.pop();
    }
}

#[test]
fn cli_reference_is_current() {
    let mut page = String::from(
        "# Command-line reference\n\nThis page is the `--help` text of each `pbz` command. \
         It is generated: run `pixi run docs-cli` after you change the CLI.\n",
    );
    render(&mut Vec::new(), &mut page);
    let file = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/reference/cli.md");
    if std::env::var_os("UPDATE_CLI_DOCS").is_some() {
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(&file, &page).unwrap();
        return;
    }
    let stored = std::fs::read_to_string(&file).unwrap_or_default();
    assert!(
        stored == page,
        "docs/reference/cli.md is stale. Run `pixi run docs-cli` to update it"
    );
}
