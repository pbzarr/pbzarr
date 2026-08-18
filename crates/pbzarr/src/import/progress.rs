//! Ready-made import progress reporting, shared by the CLI and the Python
//! bindings.
//!
//! Bridges the pipeline's `ProgressSink` to an animated indicatif bar when
//! stderr is a terminal, and to periodic plain lines otherwise so that
//! batch-job logs and Jupyter (where stderr is not a TTY and control codes
//! would be noise) still get readable progress.

use std::io::IsTerminal;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

use super::ProgressSink;

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

/// The draw target every bar from [`make_sink`] joins. Callers that also log
/// to stderr hand this to `indicatif_log_bridge::LogWrapper` so log lines are
/// printed with the bars suspended instead of through them.
pub fn multi() -> &'static MultiProgress {
    static MULTI: OnceLock<MultiProgress> = OnceLock::new();
    MULTI.get_or_init(MultiProgress::new)
}

/// Build a progress sink labeled `label`. Returns an indicatif bar on a TTY,
/// otherwise a plain periodic printer. The pipeline sizes the sink itself via
/// `ProgressSink::set_total`, so callers pass no byte total.
pub fn make_sink(label: &str) -> Arc<dyn ProgressSink> {
    if std::io::stderr().is_terminal() {
        let pb = ProgressBar::new(0);
        let style = ProgressStyle::with_template(
            "{msg} [{elapsed_precise}] {wide_bar} {bytes}/{total_bytes} {percent}% (eta {eta})",
        )
        .unwrap_or_else(|_| ProgressStyle::default_bar());
        pb.set_style(style);
        pb.set_message(label.to_owned());
        Arc::new(BarSink {
            pb: multi().add(pb),
        })
    } else {
        Arc::new(PlainSink::new(label))
    }
}

struct BarSink {
    pb: ProgressBar,
}

impl ProgressSink for BarSink {
    fn set_total(&self, bytes: u64) {
        self.pb.set_length(bytes);
    }
    fn tick(&self, bytes: u64) {
        self.pb.inc(bytes);
    }
    fn done(&self) {
        self.pb.finish();
    }
}

struct PlainSink {
    label: String,
    total: OnceLock<u64>,
    done: AtomicU64,
    last: Mutex<Instant>,
    interval: Duration,
}

impl PlainSink {
    fn new(label: &str) -> Self {
        let interval = Duration::from_secs(2);
        // Bias the clock back one interval so the first tick prints right away,
        // giving batch logs an early sign of life.
        let last = Instant::now()
            .checked_sub(interval)
            .unwrap_or_else(Instant::now);
        Self {
            label: label.to_owned(),
            total: OnceLock::new(),
            done: AtomicU64::new(0),
            last: Mutex::new(last),
            interval,
        }
    }

    fn line(&self, done: u64) {
        let total = self.total.get().copied().unwrap_or(0);
        let pct = if total > 0 {
            done as f64 / total as f64 * 100.0
        } else {
            100.0
        };
        eprintln!(
            "{}: {pct:.0}% ({} / {})",
            self.label,
            human_bytes(done),
            human_bytes(total),
        );
    }
}

impl ProgressSink for PlainSink {
    fn set_total(&self, bytes: u64) {
        let _ = self.total.set(bytes);
    }
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
