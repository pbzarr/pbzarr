pub mod cat;
pub mod completions;
pub mod drop_cmd;
pub mod export;
pub mod export_engine;
pub mod import;
pub mod info;
pub mod input_resolve;
pub mod list;
pub mod rename;
pub mod set;
pub mod validate;

use tracing_subscriber::EnvFilter;

pub fn init_tracing(verbosity: u8) {
    let filter = if verbosity > 0 {
        let level = match verbosity {
            1 => "info",
            2 => "debug",
            _ => "trace",
        };
        EnvFilter::new(level)
    } else {
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"))
    };
    if let Err(e) = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .try_init()
    {
        // Don't panic on re-init in tests, but warn in main runs.
        eprintln!("warning: tracing init failed: {e}");
    }
}
