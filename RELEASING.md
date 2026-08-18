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
   It bumps the one version line in the root `Cargo.toml` `[workspace.package]`
   table (annotated `# x-release-please-version`) and writes `CHANGELOG.md`. Every
   crate inherits that version via `version.workspace = true`, and the wheel reads
   it through `dynamic = ["version"]` in `pyproject.toml`, so nothing else needs a
   per-file bump.
3. **Merge the release PR.** That creates the git tag and GitHub release
   (`releases_created == 'true'`), which triggers the publish jobs in
   `.github/workflows/release.yml`:
   - **build-wheels** — maturin builds abi3 wheels for linux x86_64 and a macOS
     universal2 wheel (Intel + Apple Silicon) on the macos-14 Apple Silicon runner.
   - **publish-pypi** — uploads the wheels via PyPI Trusted Publishing
     (OIDC, environment `pypi`; no token stored).
   - **publish-crate** — `cargo publish -p pbzarr` via crates.io Trusted
     Publishing (OIDC; no token stored).

## Crate and wheel stay linked

There is one version for the whole repo, and it lives in exactly one place:
`[workspace.package].version` in the root `Cargo.toml`. release-please tracks a
single root package (`"."`, `release-type: simple`) and bumps that annotated line
through its generic updater. Every member crate inherits it with
`version.workspace = true`, and the wheel derives it via `dynamic = ["version"]`
(maturin reads the binding crate's inherited version). The crate and the wheel
cannot drift, and both release from the same tag. (`simple` also keeps a
release-please-managed `version.txt` at the repo root; it mirrors the same number
and is not read by cargo or maturin.)

Merging the release PR fires one GitHub release, and that single event triggers
both `publish-pypi` (through `build-wheels`) and `publish-crate`. They run as
**independent parallel jobs**, not one atomic step: both publish the same version,
but if one fails the other can still succeed. Re-run a failed job from the Actions
tab; it republishes the same pinned version (a target that already exists on
PyPI/crates.io will reject the re-push, which is the expected safety net). You
cannot release just the wheel or just the crate, they always move together.

If the wheels fail to build for a version whose crate already published (so the
tag and GitHub release exist but PyPI is missing the wheel), re-run the `Release`
workflow via **Run workflow** (`workflow_dispatch`) with the `ref` input set to the
tag (e.g. `v0.2.0`). That rebuilds the wheels off the tag and runs `publish-pypi`
through trusted publishing; `publish-crate` stays gated on a fresh release and does
not re-fire, so the crate is untouched.

## Version pinning

release-please infers the next version from conventional commit history;
`.release-please-manifest.json` tracks the current released version. (The
one-off `release-as` pin used for the first release out of this repo has been
removed.)

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
