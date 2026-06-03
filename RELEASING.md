# Releasing

Releases are driven by [release-please](https://github.com/googleapis/release-please)
from [Conventional Commit](https://www.conventionalcommits.org/) messages on `main`.
One version covers the whole repo: the `pbzarr` crate (crates.io) and the `pbzarr`
Python wheel (PyPI) ship in lockstep.

## How a release happens

1. **Merge work to `main`** with Conventional Commit messages (`feat:`, `fix:`,
   `refactor:`, ...). release-please reads these to compute the next version and
   build the changelog.
2. **release-please opens (or updates) a release PR** titled `chore(main): release <version>`.
   It bumps the version in `crates/pbzarr/Cargo.toml`, and via `extra-files` in
   `release-please-config.json` also bumps `pyproject.toml` and
   `crates/pbzarr-python/Cargo.toml`, and writes `CHANGELOG.md`.
3. **Merge the release PR.** That creates the git tag and GitHub release
   (`release_created == true`), which triggers the publish jobs in
   `.github/workflows/release.yml`:
   - **build-wheels** — maturin builds abi3 wheels for linux x86_64, macOS x86_64,
     macOS aarch64.
   - **publish-pypi** — uploads the wheels via PyPI Trusted Publishing
     (OIDC, environment `pypi`; no token stored).
   - **publish-crate** — `cargo publish -p pbzarr` via crates.io Trusted
     Publishing (OIDC; no token stored).

## Crate and wheel stay linked

There is one version for the whole repo. release-please tracks a single package
(`crates/pbzarr`) and, via `extra-files`, bumps `pyproject.toml` and
`crates/pbzarr-python/Cargo.toml` to the same number. So the crate and the wheel
always carry the same version and are released from the same tag.

Merging the release PR fires one GitHub release, and that single event triggers
both `publish-pypi` (through `build-wheels`) and `publish-crate`. They run as
**independent parallel jobs**, not one atomic step: both publish the same version,
but if one fails the other can still succeed. Re-run a failed job from the Actions
tab; it republishes the same pinned version (a target that already exists on
PyPI/crates.io will reject the re-push, which is the expected safety net). You
cannot release just the wheel or just the crate, they always move together.

## Version pinning

The next version is pinned in `release-please-config.json` via `release-as`. It is
set to `0.2.0` for the first release out of this repo (0.1.0 of both the crate and
the wheel were published before the monorepo consolidation). Remove `release-as`
once you want release-please to infer the bump from commit history again.

## Crate publishing

`publish-crate` uses crates.io Trusted Publishing (OIDC via
`rust-lang/crates-io-auth-action@v1`), so no API token is stored in the repo. The
original blocker (the core crate's `d4` git dependency) is resolved: d4 import
moved to the unpublished `pbzarr-readers` crate, so `pbzarr` itself is git-dep-free
and `cargo publish -p pbzarr` succeeds.

`pbzarr-readers` and `pbzarr-python` stay unpublished (`publish = false`):
`pbzarr-readers` carries the `d4` git dependency, and `pbzarr-python` ships only as
the wheel. Rust d4 import becomes publishable on crates.io only once `d4` itself is
on crates.io (or vendored) and `pbzarr-readers` flips to published.

## First-time setup checklist

- **PyPI:** add a Trusted Publisher for repo `pbzarr/pbzarr`, workflow `release.yml`,
  environment `pypi`. Retire the old `pbzarr-py` repo's release workflow so two
  repos don't both publish the `pbzarr` project.
- **GitHub:** the `Release` workflow needs `contents: write` + `pull-requests: write`
  (already set); the PyPI job needs `id-token: write` (already set).
- **crates.io:** add a Trusted Publisher for the `pbzarr` crate: publisher type
  GitHub, repository `pbzarr/pbzarr`, workflow filename `release.yml` (leave
  environment blank, the job sets none). The crate `pbzarr@0.1.0` must already be
  owned by you (it is), since trusted publishing attaches to an existing crate.

## Manual fallback

- Wheel: `pixi run build-wheel` produces a wheel in `target/wheels/`; upload with
  `twine`.
- Crate: `cargo publish -p pbzarr`.
