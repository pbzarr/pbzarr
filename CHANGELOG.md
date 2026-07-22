# Changelog

## [0.4.0](https://github.com/pbzarr/pbzarr/compare/v0.3.1...v0.4.0) (2026-07-21)


### ⚠ BREAKING CHANGES

* 0.3 stores are unreadable; flat storage-agnostic layout per track group (values + offsets/contigs ragged index), per-track genome, no migration.

### Features

* flat layout ([#10](https://github.com/pbzarr/pbzarr/issues/10)) ([54b7461](https://github.com/pbzarr/pbzarr/commit/54b74618ccc92bdb89f856c098198bdaff6430ac))

## [0.3.1](https://github.com/pbzarr/pbzarr/compare/v0.3.0...v0.3.1) (2026-06-18)


### Features

* progress bar for d4/bigWig import via progress=True ([eadac00](https://github.com/pbzarr/pbzarr/commit/eadac00937d412c4ed3a0df333f589017c716173))


### Miscellaneous Chores

* release 0.3.1 ([d3e7428](https://github.com/pbzarr/pbzarr/commit/d3e74283c74d70057eb637de19bcd1ec6ca0ab44))

## [0.3.0](https://github.com/pbzarr/pbzarr/compare/v0.2.1...v0.3.0) (2026-06-18)


### Features

* from_d4/from_bigwig constructors that build a store from the source file ([33794d7](https://github.com/pbzarr/pbzarr/commit/33794d75750aec4c7f98e9d1c3f9ff72d9be287d))

## [0.2.1](https://github.com/pbzarr/pbzarr/compare/v0.2.0...v0.2.1) (2026-06-18)


### Features

* pbzstore class api ([#7](https://github.com/pbzarr/pbzarr/issues/7)) ([a6b31db](https://github.com/pbzarr/pbzarr/commit/a6b31db89b75d5c69fddfcdfb301ca61ceeb3a99))
* **python:** write_track + create_track(overwrite=True) ([843e444](https://github.com/pbzarr/pbzarr/commit/843e44469f7dfd118a8b395cd470b640cffc7c94))


### Miscellaneous Chores

* release 0.2.1 ([4e2b997](https://github.com/pbzarr/pbzarr/commit/4e2b99753cec245163ccb880daf5d36f2c78ce50))

## [0.2.0](https://github.com/pbzarr/pbzarr/compare/v0.1.0...v0.2.0) (2026-06-03)


### Miscellaneous Chores

* release 0.2.0 ([f15a342](https://github.com/pbzarr/pbzarr/commit/f15a3428b8aed69514767b38fda33e311099b589))


### Build System

* unify version across crate and wheel via workspace inheritance ([e350b0a](https://github.com/pbzarr/pbzarr/commit/e350b0a819c67b5dcaf9a5b98a2333971acb1562))
