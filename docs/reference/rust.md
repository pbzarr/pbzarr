# Rust

The `pbzarr` crate is the core library. Its API reference is on [docs.rs](https://docs.rs/pbzarr).

```rust
use pbzarr::Dataset;

let ds = Dataset::open_read_only("cohort.pbz")?;
for contig in ds.reference()?.contigs() {
    println!("{} {}", contig.name, contig.length);
}
```

## Limits of the crates.io build

- The importers (d4, bigWig, BED, BAM and CRAM) are in the `pbzarr-readers` crate, which is not on crates.io yet. Use the `pbz` command or the Python package to import files.
- This repository builds against a patched `zarrs`. A build from crates.io uses upstream `zarrs`, where a partial read of an array that uses the `transpose` codec fails. The default layout does not use `transpose`.
