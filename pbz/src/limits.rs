use color_eyre::Result;
use color_eyre::eyre::eyre;
use rlimit::{Resource, getrlimit};

pub fn check_fd_budget(num_inputs: usize, writer_pool: usize) -> Result<()> {
    let (soft, _hard) = getrlimit(Resource::NOFILE)?;
    check_fd_budget_with(soft as usize, num_inputs, writer_pool)
}

pub(crate) fn check_fd_budget_with(
    soft: usize,
    num_inputs: usize,
    writer_pool: usize,
) -> Result<()> {
    let needed = 4 * num_inputs + 2 * writer_pool + 32;
    if soft >= needed {
        return Ok(());
    }
    Err(eyre!(
        "file descriptor limit too low: soft limit is {soft}, need at least {needed} \
         for {num_inputs} inputs and {writer_pool} writer threads. \
         Bump it with `ulimit -n 65536` (current shell) or `launchctl limit maxfiles ...` (macOS persistent)."
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ok_when_limit_high_enough() {
        let result = check_fd_budget_with(4096, 10, 8);
        assert!(result.is_ok());
    }

    #[test]
    fn errors_when_too_low() {
        let result = check_fd_budget_with(64, 100, 8);
        let e = result.unwrap_err();
        let s = e.to_string();
        assert!(s.contains("ulimit"));
        assert!(s.contains("64"));
    }
}
