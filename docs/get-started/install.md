# Install

## Python package

The Python package needs Python 3.12 or later.

```bash
pip install pbzarr
```

The package has wheels for Linux x86_64 and macOS (Intel and Apple Silicon). The Linux wheel needs glibc 2.28 or later (RHEL and Rocky 8 or later, Ubuntu 20.04 or later, Debian 10 or later).

Check the install:

```bash
python -c "import pbzarr; print(pbzarr.__version__)"
```

## Command-line tool

The `pbz` command is a single binary. To install it, run:

```bash
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/pbzarr/pbzarr/releases/latest/download/pbz-installer.sh | sh
```

The script downloads `pbz` from the latest release, checks its checksum, and puts it in `$CARGO_HOME/bin`. If `CARGO_HOME` is not set, it uses `~/.cargo/bin`, the same directory as `cargo install`. If that directory is not on your `PATH`, the script adds it to your shell profile. You can change two settings:

| Variable | Effect | Default |
|---|---|---|
| `PBZ_INSTALL_DIR` | The directory for `pbz` | `$CARGO_HOME/bin` or `~/.cargo/bin` |
| `PBZ_NO_MODIFY_PATH` | Set to `1` to keep the script from changing your shell profile | not set |

```bash
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/pbzarr/pbzarr/releases/latest/download/pbz-installer.sh | PBZ_INSTALL_DIR=/opt/bin PBZ_NO_MODIFY_PATH=1 sh
```

Each release has its own script, which installs that release. To install a specific version, put the version in the URL:

```bash
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/pbzarr/pbzarr/releases/download/v0.9.0/pbz-installer.sh | sh
```

You can also download the binary yourself from the [releases page](https://github.com/pbzarr/pbzarr/releases):

| System | File |
|---|---|
| Linux x86_64 (glibc 2.28 or later) | `pbz-x86_64-unknown-linux-gnu.tar.gz` |
| macOS on Apple Silicon | `pbz-aarch64-apple-darwin.tar.gz` |

Each file has a `.sha256` file next to it. Unpack the file and put `pbz` on your `PATH`:

```bash
tar -xzf pbz-x86_64-unknown-linux-gnu.tar.gz
mv pbz-x86_64-unknown-linux-gnu/pbz ~/.local/bin/
pbz --version
```

## Build from source

On other systems, build `pbz` with a Rust toolchain:

```bash
cargo install --locked --git https://github.com/pbzarr/pbzarr pbz
```

## Rust crate

Add the core library to a Rust project:

```bash
cargo add pbzarr
```

See [Rust](../reference/rust.md) for the limits of the crates.io build.
