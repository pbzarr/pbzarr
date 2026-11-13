#!/bin/sh
# Install the pbz binary from a GitHub release.
#   curl -LsSf https://raw.githubusercontent.com/pbzarr/pbzarr/main/install.sh | sh
# PBZ_VERSION=0.8.0 picks a release (default: latest).
# PBZ_INSTALL_DIR picks the directory (default: ~/.local/bin).
set -eu

repo=pbzarr/pbzarr
dir=${PBZ_INSTALL_DIR:-$HOME/.local/bin}

say() { printf 'pbz-install: %s\n' "$*" >&2; }
die() { say "error: $*"; exit 1; }

case "$(uname -s)/$(uname -m)" in
  Linux/x86_64 | Linux/amd64) target=x86_64-unknown-linux-gnu ;;
  Darwin/arm64 | Darwin/aarch64) target=aarch64-apple-darwin ;;
  *) die "no prebuilt pbz for $(uname -s) $(uname -m). Build from source: cargo install --locked --git https://github.com/$repo pbz" ;;
esac

if [ "$target" = x86_64-unknown-linux-gnu ] && command -v ldd > /dev/null 2>&1; then
  if ldd --version 2>&1 | grep -qi musl; then
    die "pbz needs glibc; this system uses musl. Build from source: cargo install --locked --git https://github.com/$repo pbz"
  fi
  glibc=$(ldd --version 2>&1 | head -n 1 | grep -Eo '[0-9]+\.[0-9]+$' || true)
  if [ -n "$glibc" ]; then
    major=${glibc%.*}
    minor=${glibc#*.}
    if [ "$major" -lt 2 ] || { [ "$major" -eq 2 ] && [ "$minor" -lt 28 ]; }; then
      die "pbz needs glibc 2.28 or later; this system has $glibc. Build from source: cargo install --locked --git https://github.com/$repo pbz"
    fi
  fi
fi

if [ -n "${PBZ_VERSION:-}" ]; then
  base="https://github.com/$repo/releases/download/v${PBZ_VERSION#v}"
else
  base="https://github.com/$repo/releases/latest/download"
fi
asset="pbz-$target.tar.gz"

if command -v curl > /dev/null 2>&1; then
  fetch() { curl --proto '=https' --tlsv1.2 -fsSL "$1" -o "$2"; }
elif command -v wget > /dev/null 2>&1; then
  fetch() { wget -q "$1" -O "$2"; }
else
  die "need curl or wget"
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

say "downloading $base/$asset"
fetch "$base/$asset" "$tmp/$asset" || die "download failed: $base/$asset"

if fetch "$base/$asset.sha256" "$tmp/$asset.sha256" 2> /dev/null; then
  want=$(cut -d ' ' -f 1 < "$tmp/$asset.sha256")
  if command -v sha256sum > /dev/null 2>&1; then
    got=$(sha256sum "$tmp/$asset" | cut -d ' ' -f 1)
  else
    got=$(shasum -a 256 "$tmp/$asset" | cut -d ' ' -f 1)
  fi
  [ "$want" = "$got" ] || die "checksum mismatch for $asset"
  say "checksum ok"
fi

tar -xzf "$tmp/$asset" -C "$tmp"
mkdir -p "$dir"
mv "$tmp/pbz" "$dir/pbz"
chmod 755 "$dir/pbz"
version=$("$dir/pbz" --version) || die "$dir/pbz was installed but does not run on this system"
say "installed $version to $dir/pbz"

case ":$PATH:" in
  *":$dir:"*) ;;
  *) say "$dir is not on your PATH. Add this line to your shell profile: export PATH=\"$dir:\$PATH\"" ;;
esac
