//! Import progress reporting for the Python bindings.
//!
//! Bridges the pipeline's `ProgressSink` to an animated indicatif bar when
//! stderr is a terminal, and to periodic plain-line prints otherwise so that
//! batch-job logs and Jupyter (where stderr is not a TTY and control codes
//! would be noise) still get readable progress.

use std::io::IsTerminal;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use indicatif::{ProgressBar, ProgressStyle};

use pbzarr::import::ProgressSink;

/// Format a byte count with a binary unit (e.g. "12.3 GiB").
fn human_bytes(n: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = n as f64;
    let mut i = 0;
    while v >= 1024.0 && i < UNITS.len() - 1 {
        v /= 1024.0;
        i += 1;
    }
    format!("{v:.1} {}", UNITS[i])
}

/// Build a progress sink sized to `total` bytes and labeled `label`. Returns an
/// indicatif bar on a TTY, otherwise a plain periodic printer.
pub fn make_sink(label: &str, total: u64) -> Arc<dyn ProgressSink> {
    if std::io::stderr().is_terminal() {
        let pb = ProgressBar::new(total);
        let style = ProgressStyle::with_template(
            "{msg} [{elapsed_precise}] {wide_bar} {bytes}/{total_bytes} {percent}% (eta {eta})",
        )
        .unwrap_or_else(|_| ProgressStyle::default_bar());
        pb.set_style(style);
        pb.set_message(label.to_owned());
        Arc::new(BarSink { pb })
    } else {
        Arc::new(PlainSink::new(label, total))
    }
}

struct BarSink {
    pb: ProgressBar,
}

impl ProgressSink for BarSink {
    fn tick(&self, bytes: u64) {
        self.pb.inc(bytes);
    }
    fn done(&self) {
        self.pb.finish();
    }
}

struct PlainSink {
    label: String,
    total: u64,
    done: AtomicU64,
    last: Mutex<Instant>,
    interval: Duration,
}

impl PlainSink {
    fn new(label: &str, total: u64) -> Self {
        let interval = Duration::from_secs(2);
        // Bias the clock back one interval so the first tick prints right away,
        // giving batch logs an early sign of life.
        let last = Instant::now()
            .checked_sub(interval)
            .unwrap_or_else(Instant::now);
        Self {
            label: label.to_owned(),
            total,
            done: AtomicU64::new(0),
            last: Mutex::new(last),
            interval,
        }
    }

    fn line(&self, done: u64) {
        let pct = if self.total > 0 {
            done as f64 / self.total as f64 * 100.0
        } else {
            100.0
        };
        eprintln!(
            "{}: {pct:.0}% ({} / {})",
            self.label,
            human_bytes(done),
            human_bytes(self.total),
        );
    }
}

impl ProgressSink for PlainSink {
    fn tick(&self, bytes: u64) {
        let done = self.done.fetch_add(bytes, Ordering::Relaxed) + bytes;
        // try_lock so workers never block on the printer; whichever thread wins
        // the lock prints, the rest move on.
        if let Ok(mut last) = self.last.try_lock()
            && last.elapsed() >= self.interval
        {
            *last = Instant::now();
            self.line(done);
        }
    }
    fn done(&self) {
        self.line(self.done.load(Ordering::Relaxed));
    }
}
