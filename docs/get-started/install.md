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
curl -LsSf https://raw.githubusercontent.com/pbzarr/pbzarr/main/install.sh | sh
```

The script downloads `pbz` from the latest release and puts it in `~/.local/bin`. If that directory is not on your `PATH`, the script tells you. You can change two settings:

| Variable | Effect | Default |
|---|---|---|
| `PBZ_VERSION` | The release to install, for example `0.8.0` | the latest release |
| `PBZ_INSTALL_DIR` | The directory for `pbz` | `~/.local/bin` |

```bash
curl -LsSf https://raw.githubusercontent.com/pbzarr/pbzarr/main/install.sh | PBZ_VERSION=0.8.0 sh
```

You can also download the binary yourself from the [releases page](https://github.com/pbzarr/pbzarr/releases):

| System | File |
|---|---|
| Linux x86_64 (glibc 2.28 or later) | `pbz-x86_64-unknown-linux-gnu.tar.gz` |
| macOS on Apple Silicon | `pbz-aarch64-apple-darwin.tar.gz` |

Unpack the file and put `pbz` on your `PATH`:

```bash
tar -xzf pbz-x86_64-unknown-linux-gnu.tar.gz
mv pbz ~/.local/bin/
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
