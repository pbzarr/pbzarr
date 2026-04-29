use indicatif::{ProgressBar, ProgressStyle};
use std::io::IsTerminal;

pub fn maybe_bar(total: u64, no_progress: bool) -> Option<ProgressBar> {
    if no_progress || !std::io::stderr().is_terminal() {
        return None;
    }
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} chunks ({eta})",
        )
        .unwrap()
        .progress_chars("=>-"),
    );
    Some(pb)
}
