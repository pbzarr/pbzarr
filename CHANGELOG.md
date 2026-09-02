# Changelog

## [0.6.0](https://github.com/pbzarr/pbzarr/compare/v0.5.1...v0.6.0) (2026-09-02)


### ⚠ BREAKING CHANGES

* rebuild the import pipeline on a routing engine with opt-in scales

### Features

* add pcodec ([a6f5977](https://github.com/pbzarr/pbzarr/commit/a6f5977b4fffdd16062d52aa89da05ba9f0ce340))
* add precision option to pbz view ([34f2279](https://github.com/pbzarr/pbzarr/commit/34f2279a29db3f7923c83e9d83c4b6b7bb3464e4))
* add summit reducer ([8fe9962](https://github.com/pbzarr/pbzarr/commit/8fe9962495e7aac5b97b7ef026a2570da4ce333a))
* add threads option to pbz view ([9f58bd1](https://github.com/pbzarr/pbzarr/commit/9f58bd1624a2075ba96cc6d53c95fdf73521b182))
* batch stack of single-sample stores into a cohort store ([#17](https://github.com/pbzarr/pbzarr/issues/17)) ([02010de](https://github.com/pbzarr/pbzarr/commit/02010de5ed5e031481acc7753bac7024fe985653))
* cli ([#21](https://github.com/pbzarr/pbzarr/issues/21)) ([4dcf5c8](https://github.com/pbzarr/pbzarr/commit/4dcf5c8672a573c0d9e6b215e1d5a12d4b48fa37))
* decode spans decoupled from chunk size ([694ad84](https://github.com/pbzarr/pbzarr/commit/694ad84a71e2131dfe25b2e88c1af75ce5229e72))
* engine timing summary and configurable in-flight span gate ([232ad7c](https://github.com/pbzarr/pbzarr/commit/232ad7c5ffbd2e95f8b8f1191231ce090bc1420c))
* implement pbz view ([dc1b358](https://github.com/pbzarr/pbzarr/commit/dc1b3588d2160fefe167e4ff6e09375cca23d231))
* import codec overrides via --codecs ([993239a](https://github.com/pbzarr/pbzarr/commit/993239a673c835ccc4b43e4152c8cd9ef0a06ea7))
* **pbz:** add scale subcommand for multiscale pyramid generation ([52ba1e9](https://github.com/pbzarr/pbzarr/commit/52ba1e938693a6a06d2995a0fae09253dfff4f71))
* **pbzarr:** multiscale scale engine with mean levels, publication, and base-write seal ([21db425](https://github.com/pbzarr/pbzarr/commit/21db4257a1fc8734e0fb3fe54a714ff688d70d60))
* **pbzarr:** write zarr-python-flavored consolidated metadata at the store root ([cf3e38f](https://github.com/pbzarr/pbzarr/commit/cf3e38f096c22cdb38d386749dbca84db0ee3e94))
* rebuild the import pipeline on a routing engine with opt-in scales ([7728873](https://github.com/pbzarr/pbzarr/commit/7728873d4e2eac2a5d391e32ae3599fb3bd1e422))


### Bug Fixes

* accept numpy StringDType contigs in region queries ([ae000b8](https://github.com/pbzarr/pbzarr/commit/ae000b8a39446625a2730d2f9c4560a7b6878734))
* batch shard appends per span ([51c0838](https://github.com/pbzarr/pbzarr/commit/51c0838afd8c162a6f4024027b974f004df8ac88))
* decode without holding buffer-slot locks ([a92d63b](https://github.com/pbzarr/pbzarr/commit/a92d63b6d88c64aed713cf61ea5e651e98833ff6))
* logging and progress ([717b3d5](https://github.com/pbzarr/pbzarr/commit/717b3d56ab31f28c5549dfb113f5c4a85804d5f2))
* open collections containing published scales pyramids ([0e13618](https://github.com/pbzarr/pbzarr/commit/0e136188e97d637d94c16e1e1f6026154a42f0ad))
* **pbzarr:** bound scale slab memory with column-axis blocks on wide cohorts ([b6773a1](https://github.com/pbzarr/pbzarr/commit/b6773a1c082f5e653f028f851149caa63db3d928))
* **pbzarr:** refresh consolidated metadata on track completion ([b7bea3c](https://github.com/pbzarr/pbzarr/commit/b7bea3cdb034cf45754bf08f7a4170c58f9de68d))
* pool reader handles under an fd budget ([0ada5ba](https://github.com/pbzarr/pbzarr/commit/0ada5bac56198a0184a19c8d43d7452e0433b660))
* **scale:** refresh consolidated metadata when refusing an already-published rescale ([f52a16e](https://github.com/pbzarr/pbzarr/commit/f52a16ee0d499cb1622e58df5e1eaea208b03916))


### Performance Improvements

* decode BGZF blocks with libdeflate in the noodles readers ([b0f5ffa](https://github.com/pbzarr/pbzarr/commit/b0f5ffa7e4fe79d0dcf714caaa76cc017d2b0b87))
* multithread bgzf compression in pbz view ([9f67d9a](https://github.com/pbzarr/pbzarr/commit/9f67d9a73edc77bd9adb1cbd50924ed4383e171b))
* **scale:** cascade child factors from parent accumulators ([4145fe1](https://github.com/pbzarr/pbzarr/commit/4145fe13d446ef504dac2ac917969ab124015a68))
* **scale:** parallel work units with a single writer thread ([8672642](https://github.com/pbzarr/pbzarr/commit/8672642eb4a558435b839f4273acd5dc0487599c))
* **scale:** single-pass multi-factor accumulation with Tier A/B decision ([f07f53d](https://github.com/pbzarr/pbzarr/commit/f07f53d274594bdd8a47451281f510f4cf52462b))
* size the rayon encode pool to the import worker count ([9766865](https://github.com/pbzarr/pbzarr/commit/976686577efdd6482c067bddb953eeb3006c2bb4))
* skip SEQ and QUAL decode in depth mode without a base-quality gate ([39b8603](https://github.com/pbzarr/pbzarr/commit/39b860303c9eb92adb47c43b14beff574867cdfb))

## [0.5.1](https://github.com/pbzarr/pbzarr/compare/v0.5.0...v0.5.1) (2026-07-22)


### Bug Fixes

* add PbzStore.import_bed_multi for single-pass multi-column BED import ([#15](https://github.com/pbzarr/pbzarr/issues/15)) ([f96eef0](https://github.com/pbzarr/pbzarr/commit/f96eef01b5fd62071ed02a7c86ceb769b77e8677))

## [0.5.0](https://github.com/pbzarr/pbzarr/compare/v0.4.0...v0.5.0) (2026-07-22)


### Features

* python flat api ([#13](https://github.com/pbzarr/pbzarr/issues/13)) ([bd5881d](https://github.com/pbzarr/pbzarr/commit/bd5881d9ea24317478c0d2001093ee47f50cdb81))
* single-pass multi-column BED import to per-column tracks ([#12](https://github.com/pbzarr/pbzarr/issues/12)) ([ae90098](https://github.com/pbzarr/pbzarr/commit/ae900984e0cc6f1a25237dc2d534cb04ae656027))

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
