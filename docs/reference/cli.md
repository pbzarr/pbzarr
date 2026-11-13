# Command-line reference

This page is the `--help` text of each `pbz` command. It is generated: run `pixi run docs-cli` after you change the CLI.

## `pbz`

```text
Per-base Zarr tools

Usage: pbz [OPTIONS] <COMMAND>

Commands:
  view      Export one track as bedGraph-like text
  stat      Compute summary statistics for one track
  import    Import data into a PBZ store
  append    Append columns to a track of a format-0.5 PBZ store
  info      Print the reference, dimensions and per-variable layout of a store
  validate  Check a store against the PBZ format rules. Exit code 1 when the store is not valid
  help      Print this message or the help of the given subcommand(s)

Options:
  -v, --verbose...  Increase logging verbosity: -v info, -vv debug, -vvv trace. Warnings and errors always print. `RUST_LOG` overrides this when set
  -q, --quiet       Hide the summary line and the progress display of import and append
  -h, --help        Print help
  -V, --version     Print version
```

### `pbz view`

```text
Export one track as bedGraph-like text

Usage: pbz view [OPTIONS] <STORE>

Arguments:
  <STORE>  PBZ store to export

Options:
      --track <TRACK>    Track to export. Optional when the store holds exactly one track
  -v, --verbose...       Increase logging verbosity: -v info, -vv debug, -vvv trace. Warnings and errors always print. `RUST_LOG` overrides this when set
  -q, --quiet            Hide the summary line and the progress display of import and append
  -r, --region <REGION>  Region CHROM:START-END, 0-based and half-open as in BED (chr1:0-100 is the first 100 bases). CHROM alone selects the whole contig, also when the name contains ':'. An end past the contig is an error. Repeat for more regions. Without -r or -R, the command reads the whole genome
  -R, --regions <BED>    BED file (plain, gzip, or bgzip) with regions to read after the -r regions. The first three tab-separated columns give CHROM, START, END, 0-based and half-open. Lines that start with #, track, or browser are skipped
  -c, --columns <LABEL>  Column labels of a rank-2 track, comma-separated or repeated. Default: all columns in stored order
  -o, --output <PATH>    Output path. A `.gz` suffix writes BGZF; default is stdout
      --no-header        Do not write the `#chrom start end ...` header line
  -t, --threads <N>      Worker threads for chunk decode and BGZF compression. Default: all cores
  -p, --precision <N>    Significant digits for float values, `%g` style (UCSC bigWig convention) [default: 6]
  -h, --help             Print help

Examples:
  pbz view cohort.pbz --track depth
  pbz view cohort.pbz --track depth -r chr1:0-1000000 -o depth.bedgraph.gz
  pbz view cohort.pbz --track af -c s1,s2 -R targets.bed

Each region is written in order. Rows do not extend across region edges.

The track must have position as its first dimension and at most one other dimension. Tracks on a bin_N dimension are not supported. A null marks a missing value: NaN in a float track, and the _FillValue in an integer track. A float _FillValue must be NaN. view prints a null as NaN and leaves out rows where all columns are null. stat skips nulls.
```

### `pbz stat`

```text
Compute summary statistics for one track

Usage: pbz stat [OPTIONS] <STORE>

Arguments:
  <STORE>  PBZ store to read

Options:
      --track <TRACK>    Track to compute over. Optional when the store holds exactly one track
  -v, --verbose...       Increase logging verbosity: -v info, -vv debug, -vvv trace. Warnings and errors always print. `RUST_LOG` overrides this when set
  -q, --quiet            Hide the summary line and the progress display of import and append
  -r, --region <REGION>  Region CHROM:START-END, 0-based and half-open as in BED (chr1:0-100 is the first 100 bases). CHROM alone selects the whole contig, also when the name contains ':'. An end past the contig is an error. Repeat for more regions. Without -r or -R, the command reads the whole genome
  -R, --regions <BED>    BED file (plain, gzip, or bgzip) with regions to read after the -r regions. The first three tab-separated columns give CHROM, START, END, 0-based and half-open. Lines that start with #, track, or browser are skipped
  -s, --stat <STAT>      Statistic to compute [default: mean] [possible values: mean, min, max, median, hist]
  -c, --columns <LABEL>  Column labels of a rank-2 track, comma-separated or repeated. Default: all columns in stored order
  -o, --output <PATH>    Output path. A `.gz` suffix writes BGZF; default is stdout
      --no-header        Do not write the header line
  -t, --threads <N>      Worker threads for chunk decode and stat batches. Default: all cores
  -p, --precision <N>    Significant digits for float values, `%g` style [default: 6]
  -h, --help             Print help

Examples:
  pbz stat cohort.pbz --track depth
  pbz stat cohort.pbz --track depth -s median -R targets.bed
  pbz stat cohort.pbz --track depth -r chr1:0-1000 -r chr2:500-900
  pbz stat cohort.pbz --track af -s hist -c s1,s2

Without a region, stat writes one row per contig.

The track must have position as its first dimension and at most one other dimension. Tracks on a bin_N dimension are not supported. A null marks a missing value: NaN in a float track, and the _FillValue in an integer track. A float _FillValue must be NaN. view prints a null as NaN and leaves out rows where all columns are null. stat skips nulls.
```

### `pbz import`

```text
Import data into a PBZ store

Usage: pbz import [OPTIONS] <COMMAND>

Commands:
  bed     Import from tabix-indexed BED files
  bam     Import per-base depth or composition counts from BAM/CRAM files
  d4      Import per-base depth from d4 files
  bigwig  Import per-base float32 signal from bigWig files
  help    Print this message or the help of the given subcommand(s)

Options:
  -v, --verbose...  Increase logging verbosity: -v info, -vv debug, -vvv trace. Warnings and errors always print. `RUST_LOG` overrides this when set
  -q, --quiet       Hide the summary line and the progress display of import and append
  -h, --help        Print help
```

#### `pbz import bed`

```text
Import from tabix-indexed BED files

Usage: pbz import bed [OPTIONS] [STORE] [PATH[:LABEL]]...

Arguments:
  [STORE]            PBZ store to write. Import creates it when the path does not exist, and adds the track(s) to it when it exists
  [PATH[:LABEL]]...  Input files, optionally suffixed with `:LABEL`. The default label is the file name without its format extension (`s1.bed.gz` gives `s1`)

Options:
  -t, --threads <THREADS>    Worker threads for decode and write [default: 4]
  -v, --verbose...           Increase logging verbosity: -v info, -vv debug, -vvv trace. Warnings and errors always print. `RUST_LOG` overrides this when set
      --decode-chunks <N>    Position chunks each reader decodes per read pass. Default: auto (up to 4M positions, fewer when that leaves workers short of pieces)
  -q, --quiet                Hide the summary line and the progress display of import and append
      --no-progress          Hide the progress display
      --chunk <N>            Positions per inner chunk. A 2D track has one column per inner chunk. Default: 65,536
      --shard <N>            Inner chunks per shard along positions, so a shard holds chunk x N positions. A shard is capped at the genome length, rounded up to whole chunks. Default: 1,024
      --shard-columns <N>    Columns per shard of a 2D track. A shard is capped at the track width. Default: 16. Give 1 to make a track that pbz append can grow
      --dim <NAME>           Name of the second dimension of a 2D track. A single source with --dim still makes a 2D track. Default: `sample` for several d4, bigWig or BAM sources. BED import has no default: it needs --dim whenever the output is 2D (several sources, or several BED columns in one track)
      --codecs <JSON|@FILE>  Zarr v3 codec metadata for the created track arrays: a `codecs` JSON array, or a `{"chunk_grid": ..., "codecs": [...]}` fragment. Pass `@path` to read the JSON from a file. Replaces the default Blosc(zstd-5, byte shuffle) pipeline and the whole chunk and shard layout; express sharding as a `sharding_indexed` codec and the outer grid as `chunk_grid`. Without `chunk_grid`, the outer grid is 65,536 positions by 16 columns, capped at the track width
      --zip                  Write a new single-file zip store. A STORE path that ends in `.zip` also does this. The path must not exist
  -f, --file-list <PATH>     Tab-delimited source list with PATH or PATH<TAB>LABEL records. Relative paths start from the directory of the list file
      --genome <GENOME>      FAI or chromosome-sizes file with the contig lengths. It is needed only when the store does not exist; an existing store gives its own
      --field <FIELDS>       BED header column to import (requires a `#`-prefixed header). Repeating selects several BED columns; omit to import all BED value columns
      --track <TRACK>        Output track name. Required for BED3, headerless BED4, and when several BED columns from one source form one wide track
      --dtype <DTYPE>        Value dtype for headerless BED4 input (inferred when omitted) [possible values: bool, uint8, uint16, uint32, uint64, int8, int16, int32, int64, float32, float64]
      --schema <PATH>        TSV schema mapping BED columns to target tracks. Unique target names create one track per BED column; repeated one-target rows create a single BED-column-axis track. A BED column is a header name or 1-based column index. A row can also give the dtype and the fill of its track
      --emit-schema <BED>    Print a schema TSV for this BED file (BED column, target track, inferred dtype, fill) to stdout and exit without importing. Edit it and feed it back via --schema
      --fill <VALUE|na>      Value of bases that no interval covers, for BED value columns: a number, true or false for a bool column, or na. na is the null of the dtype: NaN for a float, the type maximum for an integer. Default: na, and false for a bool column. BED3 masks always hold false
      --infer-rows <N|all>   Records sampled for dtype inference (default 1000), or `all`. A sample infers conservative classes (bool/int32/float32); `all` (or a file smaller than the sample) scans exhaustively and min-widths (uint8, int16, ...)
  -h, --help                 Print help

Examples:
  pbz import bed cohort.pbz s1.bed.gz s2.bed.gz --genome genome.fai --track depth --dim sample
  pbz import bed mask.pbz regions.bed.gz --genome genome.fai --track callable
  pbz import bed --emit-schema wide.bed.gz > schema.tsv
  pbz import bed stats.pbz wide.bed.gz --genome genome.fai --dim metric --schema schema.tsv

Tracks without --schema (S sources, V selected BED value columns):
  BED3            one bool track, true where an interval covers the base. Needs --track,
                  and --dim for several sources.
  S = 1, V = 1    one track (position). Needs --track when the file has no header.
  S = 1, V > 1    one track (position, V). The header names are the column labels.
                  Needs --track and --dim.
  S > 1, V = 1    one track (position, S). The source labels are the column labels.
                  Needs --dim, and --track when the files have no header.
  S > 1, V > 1    one track (position, S) for each value column, with the name of
                  that column. Needs --dim.
A file without a `#chrom start end ...` header can have one value column only.
--dim with one source and one value column also makes a (position, S) track.

Bases that no interval covers are missing by default: they hold the null of the
dtype (NaN for a float, the type maximum for an integer), which the track declares
as its _FillValue. With this fill, inference keeps the type maximum free, and a
value equal to it is an error. A bool column and a BED3 mask hold false there.
Give --fill 0 when the gaps mean 0, as in a bedGraph that leaves out zero regions.

Schema rows are BED_COLUMN[<TAB>TRACK[<TAB>DTYPE[<TAB>FILL]]]. An empty or absent
TRACK takes the header name, an empty or absent DTYPE is inferred, and an empty
or absent FILL takes --fill, else the default.
```

#### `pbz import bam`

```text
Import per-base depth or composition counts from BAM/CRAM files

Usage: pbz import bam [OPTIONS] --track <TRACK> <STORE> [PATH[:LABEL]]...

Arguments:
  <STORE>            PBZ store to write. Import creates it from the source headers when the path does not exist, and adds the track(s) to it when it exists
  [PATH[:LABEL]]...  Input files, optionally suffixed with `:LABEL`. The default label is the file name without its format extension (`s1.bed.gz` gives `s1`)

Options:
  -t, --threads <THREADS>      Worker threads for decode and write [default: 4]
  -v, --verbose...             Increase logging verbosity: -v info, -vv debug, -vvv trace. Warnings and errors always print. `RUST_LOG` overrides this when set
      --decode-chunks <N>      Position chunks each reader decodes per read pass. Default: auto (up to 4M positions, fewer when that leaves workers short of pieces)
  -q, --quiet                  Hide the summary line and the progress display of import and append
      --no-progress            Hide the progress display
      --chunk <N>              Positions per inner chunk. A 2D track has one column per inner chunk. Default: 65,536
      --shard <N>              Inner chunks per shard along positions, so a shard holds chunk x N positions. A shard is capped at the genome length, rounded up to whole chunks. Default: 1,024
      --shard-columns <N>      Columns per shard of a 2D track. A shard is capped at the track width. Default: 16. Give 1 to make a track that pbz append can grow
      --dim <NAME>             Name of the second dimension of a 2D track. A single source with --dim still makes a 2D track. Default: `sample` for several d4, bigWig or BAM sources. BED import has no default: it needs --dim whenever the output is 2D (several sources, or several BED columns in one track)
      --codecs <JSON|@FILE>    Zarr v3 codec metadata for the created track arrays: a `codecs` JSON array, or a `{"chunk_grid": ..., "codecs": [...]}` fragment. Pass `@path` to read the JSON from a file. Replaces the default Blosc(zstd-5, byte shuffle) pipeline and the whole chunk and shard layout; express sharding as a `sharding_indexed` codec and the outer grid as `chunk_grid`. Without `chunk_grid`, the outer grid is 65,536 positions by 16 columns, capped at the track width
      --zip                    Write a new single-file zip store. A STORE path that ends in `.zip` also does this. The path must not exist
  -f, --file-list <PATH>       Tab-delimited source list with PATH or PATH<TAB>LABEL records. Relative paths start from the directory of the list file
      --track <TRACK>          Output track name. In composition mode the depth track keeps this name and the per-event tracks are named `{track}_{field}`
      --mode <MODE>            Depth counts coverage only; composition also counts per-base events, emitting `{track}_{field}` for a, c, g, t, n, ins, del, ref_skip [default: depth] [possible values: depth, composition]
      --fasta <FASTA>          FASTA for CRAM sources (ignored for BAM)
      --min-mapq <MIN_MAPQ>    Minimum mapping quality for a read to count [default: 0]
      --min-bq <MIN_BQ>        Minimum per-base quality for a base to count [default: 0]
      --exclude-flags <FLAGS>  SAM flag bits that exclude a read: a decimal or 0x hex number, or comma-separated samtools flag names (PAIRED, PROPER_PAIR, UNMAP, MUNMAP, REVERSE, MREVERSE, READ1, READ2, SECONDARY, QCFAIL, DUP, SUPPLEMENTARY) [default: UNMAP,SECONDARY,QCFAIL,DUP]
      --overlap <OVERLAP>      Mate-overlap dedup mode. `proper` (default) matches mosdepth: only PROPER_PAIR-flagged overlapping mates are collapsed to one count. `all` matches riker/samtools-mpileup-style unconditional dedup of any overlapping mate pair. `none` disables dedup, double-counting every overlapping pair's shared span [default: proper] [possible values: proper, all, none]
      --count-deletions        Count CIGAR D (deletion)-spanned positions as covered depth. Off by default, matching the samtools/mosdepth/Picard/riker convention of excluding deletions from depth
  -h, --help                   Print help

Examples:
  pbz import bam depth.pbz s1.bam s2.bam --track depth
  pbz import bam comp.pbz sample.cram --track depth --mode composition --fasta ref.fa
```

#### `pbz import d4`

```text
Import per-base depth from d4 files

Usage: pbz import d4 [OPTIONS] --track <TRACK> <STORE> [PATH[:LABEL]]...

Arguments:
  <STORE>            PBZ store to write. Import creates it from the source headers when the path does not exist, and adds the track(s) to it when it exists
  [PATH[:LABEL]]...  Input files, optionally suffixed with `:LABEL`. The default label is the file name without its format extension (`s1.bed.gz` gives `s1`)

Options:
  -t, --threads <THREADS>    Worker threads for decode and write [default: 4]
  -v, --verbose...           Increase logging verbosity: -v info, -vv debug, -vvv trace. Warnings and errors always print. `RUST_LOG` overrides this when set
      --decode-chunks <N>    Position chunks each reader decodes per read pass. Default: auto (up to 4M positions, fewer when that leaves workers short of pieces)
  -q, --quiet                Hide the summary line and the progress display of import and append
      --no-progress          Hide the progress display
      --chunk <N>            Positions per inner chunk. A 2D track has one column per inner chunk. Default: 65,536
      --shard <N>            Inner chunks per shard along positions, so a shard holds chunk x N positions. A shard is capped at the genome length, rounded up to whole chunks. Default: 1,024
      --shard-columns <N>    Columns per shard of a 2D track. A shard is capped at the track width. Default: 16. Give 1 to make a track that pbz append can grow
      --dim <NAME>           Name of the second dimension of a 2D track. A single source with --dim still makes a 2D track. Default: `sample` for several d4, bigWig or BAM sources. BED import has no default: it needs --dim whenever the output is 2D (several sources, or several BED columns in one track)
      --codecs <JSON|@FILE>  Zarr v3 codec metadata for the created track arrays: a `codecs` JSON array, or a `{"chunk_grid": ..., "codecs": [...]}` fragment. Pass `@path` to read the JSON from a file. Replaces the default Blosc(zstd-5, byte shuffle) pipeline and the whole chunk and shard layout; express sharding as a `sharding_indexed` codec and the outer grid as `chunk_grid`. Without `chunk_grid`, the outer grid is 65,536 positions by 16 columns, capped at the track width
      --zip                  Write a new single-file zip store. A STORE path that ends in `.zip` also does this. The path must not exist
  -f, --file-list <PATH>     Tab-delimited source list with PATH or PATH<TAB>LABEL records. Relative paths start from the directory of the list file
      --track <TRACK>        Output track name
  -h, --help                 Print help

Examples:
  pbz import d4 cohort.pbz s1.d4 s2.d4:custom_label --track depth
```

#### `pbz import bigwig`

```text
Import per-base float32 signal from bigWig files

Usage: pbz import bigwig [OPTIONS] --track <TRACK> <STORE> [PATH[:LABEL]]...

Arguments:
  <STORE>            PBZ store to write. Import creates it from the source headers when the path does not exist, and adds the track(s) to it when it exists
  [PATH[:LABEL]]...  Input files, optionally suffixed with `:LABEL`. The default label is the file name without its format extension (`s1.bed.gz` gives `s1`)

Options:
  -t, --threads <THREADS>    Worker threads for decode and write [default: 4]
  -v, --verbose...           Increase logging verbosity: -v info, -vv debug, -vvv trace. Warnings and errors always print. `RUST_LOG` overrides this when set
      --decode-chunks <N>    Position chunks each reader decodes per read pass. Default: auto (up to 4M positions, fewer when that leaves workers short of pieces)
  -q, --quiet                Hide the summary line and the progress display of import and append
      --no-progress          Hide the progress display
      --chunk <N>            Positions per inner chunk. A 2D track has one column per inner chunk. Default: 65,536
      --shard <N>            Inner chunks per shard along positions, so a shard holds chunk x N positions. A shard is capped at the genome length, rounded up to whole chunks. Default: 1,024
      --shard-columns <N>    Columns per shard of a 2D track. A shard is capped at the track width. Default: 16. Give 1 to make a track that pbz append can grow
      --dim <NAME>           Name of the second dimension of a 2D track. A single source with --dim still makes a 2D track. Default: `sample` for several d4, bigWig or BAM sources. BED import has no default: it needs --dim whenever the output is 2D (several sources, or several BED columns in one track)
      --codecs <JSON|@FILE>  Zarr v3 codec metadata for the created track arrays: a `codecs` JSON array, or a `{"chunk_grid": ..., "codecs": [...]}` fragment. Pass `@path` to read the JSON from a file. Replaces the default Blosc(zstd-5, byte shuffle) pipeline and the whole chunk and shard layout; express sharding as a `sharding_indexed` codec and the outer grid as `chunk_grid`. Without `chunk_grid`, the outer grid is 65,536 positions by 16 columns, capped at the track width
      --zip                  Write a new single-file zip store. A STORE path that ends in `.zip` also does this. The path must not exist
  -f, --file-list <PATH>     Tab-delimited source list with PATH or PATH<TAB>LABEL records. Relative paths start from the directory of the list file
      --track <TRACK>        Output track name
      --fill <VALUE|na>      Value of bases that no bigWig interval covers: a number, or na. na marks a missing value and reads as NaN. Default: na
  -h, --help                 Print help

Examples:
  pbz import bigwig signal.pbz sample.bw --track signal
  pbz import bigwig cohort.pbz brain.bw:brain liver.bw:liver --track signal --dim sample
  pbz import bigwig signal.pbz sample.bw --track signal --fill 0

A bigWig does not say what a base outside its intervals means. By default such a base is missing: it reads as NaN, and the track declares NaN as its _FillValue. Give --fill 0 when the gaps mean 0.
```

### `pbz append`

```text
Append columns to a track of a format-0.5 PBZ store

Usage: pbz append [OPTIONS] <COMMAND>

Commands:
  bed     Append columns decoded from tabix-indexed BED files
  bam     Append columns decoded from BAM/CRAM files
  d4      Append columns decoded from d4 files
  bigwig  Append columns decoded from bigWig files
  help    Print this message or the help of the given subcommand(s)

Options:
  -v, --verbose...  Increase logging verbosity: -v info, -vv debug, -vvv trace. Warnings and errors always print. `RUST_LOG` overrides this when set
  -q, --quiet       Hide the summary line and the progress display of import and append
  -h, --help        Print help

The store fixes chunks, shards, codecs and the column dimension, so append takes no layout flags. Every root variable over the appended dimension grows. The tracks that the append writes get values in the new columns: --track, and with bam --mode composition also the {track}_{field} tracks. In the other variables, the new columns read as the fill value. A variable without that dimension, such as a reduction across columns, keeps its shape and becomes stale. pbz does not detect or mark it. Append cannot yet grow a track whose shard holds more than one column, which is the import default (--shard-columns 16). To append later, import with --shard-columns 1.
```

#### `pbz append bed`

```text
Append columns decoded from tabix-indexed BED files

Usage: pbz append bed [OPTIONS] --track <TRACK> <STORE> [PATH[:LABEL]]...

Arguments:
  <STORE>            PBZ store to grow. It must exist
  [PATH[:LABEL]]...  Input files, optionally suffixed with `:LABEL`. The default label is the file name without its format extension (`s1.bed.gz` gives `s1`)

Options:
  -t, --threads <THREADS>  Worker threads for decode and write [default: 4]
  -v, --verbose...         Increase logging verbosity: -v info, -vv debug, -vvv trace. Warnings and errors always print. `RUST_LOG` overrides this when set
      --decode-chunks <N>  Position chunks each reader decodes per read pass. Default: auto (up to 4M positions, fewer when that leaves workers short of pieces)
  -q, --quiet              Hide the summary line and the progress display of import and append
      --no-progress        Hide the progress display
  -f, --file-list <PATH>   Tab-delimited source list with PATH or PATH<TAB>LABEL records. Relative paths start from the directory of the list file
      --track <TRACK>      Track to grow. It must already have a column dimension
      --field <FIELDS>     BED header column that fills --track (requires a `#`-prefixed header). Omit when the header has one BED value column
      --fill <VALUE|na>    Value of bases that no interval covers: a number, true or false for a bool track, or na. Default: the fill of the stored track. A different fill is refused
  -h, --help               Print help

Examples:
  pbz append bed cohort.pbz s3.bed.gz s4.bed.gz --track depth
```

#### `pbz append bam`

```text
Append columns decoded from BAM/CRAM files

Usage: pbz append bam [OPTIONS] --track <TRACK> <STORE> [PATH[:LABEL]]...

Arguments:
  <STORE>            PBZ store to grow. It must exist
  [PATH[:LABEL]]...  Input files, optionally suffixed with `:LABEL`. The default label is the file name without its format extension (`s1.bed.gz` gives `s1`)

Options:
  -t, --threads <THREADS>      Worker threads for decode and write [default: 4]
  -v, --verbose...             Increase logging verbosity: -v info, -vv debug, -vvv trace. Warnings and errors always print. `RUST_LOG` overrides this when set
      --decode-chunks <N>      Position chunks each reader decodes per read pass. Default: auto (up to 4M positions, fewer when that leaves workers short of pieces)
  -q, --quiet                  Hide the summary line and the progress display of import and append
      --no-progress            Hide the progress display
  -f, --file-list <PATH>       Tab-delimited source list with PATH or PATH<TAB>LABEL records. Relative paths start from the directory of the list file
      --track <TRACK>          Track to grow. It must already have a column dimension
      --mode <MODE>            Depth counts coverage only; composition also counts per-base events, emitting `{track}_{field}` for a, c, g, t, n, ins, del, ref_skip [default: depth] [possible values: depth, composition]
      --fasta <FASTA>          FASTA for CRAM sources (ignored for BAM)
      --min-mapq <MIN_MAPQ>    Minimum mapping quality for a read to count [default: 0]
      --min-bq <MIN_BQ>        Minimum per-base quality for a base to count [default: 0]
      --exclude-flags <FLAGS>  SAM flag bits that exclude a read: a decimal or 0x hex number, or comma-separated samtools flag names (PAIRED, PROPER_PAIR, UNMAP, MUNMAP, REVERSE, MREVERSE, READ1, READ2, SECONDARY, QCFAIL, DUP, SUPPLEMENTARY) [default: UNMAP,SECONDARY,QCFAIL,DUP]
      --overlap <OVERLAP>      Mate-overlap dedup mode. `proper` (default) matches mosdepth: only PROPER_PAIR-flagged overlapping mates are collapsed to one count. `all` matches riker/samtools-mpileup-style unconditional dedup of any overlapping mate pair. `none` disables dedup, double-counting every overlapping pair's shared span [default: proper] [possible values: proper, all, none]
      --count-deletions        Count CIGAR D (deletion)-spanned positions as covered depth. Off by default, matching the samtools/mosdepth/Picard/riker convention of excluding deletions from depth
  -h, --help                   Print help

Examples:
  pbz append bam depth.pbz s3.bam --track depth
  pbz append bam comp.pbz s3.cram --track depth --mode composition --fasta ref.fa
```

#### `pbz append d4`

```text
Append columns decoded from d4 files

Usage: pbz append d4 [OPTIONS] --track <TRACK> <STORE> [PATH[:LABEL]]...

Arguments:
  <STORE>            PBZ store to grow. It must exist
  [PATH[:LABEL]]...  Input files, optionally suffixed with `:LABEL`. The default label is the file name without its format extension (`s1.bed.gz` gives `s1`)

Options:
  -t, --threads <THREADS>  Worker threads for decode and write [default: 4]
  -v, --verbose...         Increase logging verbosity: -v info, -vv debug, -vvv trace. Warnings and errors always print. `RUST_LOG` overrides this when set
      --decode-chunks <N>  Position chunks each reader decodes per read pass. Default: auto (up to 4M positions, fewer when that leaves workers short of pieces)
  -q, --quiet              Hide the summary line and the progress display of import and append
      --no-progress        Hide the progress display
  -f, --file-list <PATH>   Tab-delimited source list with PATH or PATH<TAB>LABEL records. Relative paths start from the directory of the list file
      --track <TRACK>      Track to grow. It must already have a column dimension
  -h, --help               Print help

Examples:
  pbz append d4 cohort.pbz s3.d4 s4.d4:custom_label --track depth
```

#### `pbz append bigwig`

```text
Append columns decoded from bigWig files

Usage: pbz append bigwig [OPTIONS] --track <TRACK> <STORE> [PATH[:LABEL]]...

Arguments:
  <STORE>            PBZ store to grow. It must exist
  [PATH[:LABEL]]...  Input files, optionally suffixed with `:LABEL`. The default label is the file name without its format extension (`s1.bed.gz` gives `s1`)

Options:
  -t, --threads <THREADS>  Worker threads for decode and write [default: 4]
  -v, --verbose...         Increase logging verbosity: -v info, -vv debug, -vvv trace. Warnings and errors always print. `RUST_LOG` overrides this when set
      --decode-chunks <N>  Position chunks each reader decodes per read pass. Default: auto (up to 4M positions, fewer when that leaves workers short of pieces)
  -q, --quiet              Hide the summary line and the progress display of import and append
      --no-progress        Hide the progress display
  -f, --file-list <PATH>   Tab-delimited source list with PATH or PATH<TAB>LABEL records. Relative paths start from the directory of the list file
      --track <TRACK>      Track to grow. It must already have a column dimension
      --fill <VALUE|na>    Value of bases that no bigWig interval covers: a number, or na. Default: the fill of the stored track. A different fill is refused
  -h, --help               Print help

Examples:
  pbz append bigwig cohort.pbz kidney.bw:kidney --track signal
```

### `pbz info`

```text
Print the reference, dimensions and per-variable layout of a store

Usage: pbz info [OPTIONS] <STORE>

Arguments:
  <STORE>  PBZ store to describe (directory or zip)

Options:
      --json        Print one JSON object instead of text
  -v, --verbose...  Increase logging verbosity: -v info, -vv debug, -vvv trace. Warnings and errors always print. `RUST_LOG` overrides this when set
  -q, --quiet       Hide the summary line and the progress display of import and append
  -h, --help        Print help

Examples:
  pbz info cohort.pbz
  pbz info cohort.pbz.zip --json

The chunk column gives the inner chunk shape. The shard column gives the shard shape, or - when the array is not sharded. stored and objects count the chunk or shard objects, without the zarr.json metadata.
```

### `pbz validate`

```text
Check a store against the PBZ format rules. Exit code 1 when the store is not valid

Usage: pbz validate [OPTIONS] <STORE>

Arguments:
  <STORE>  PBZ store to check (directory or zip)

Options:
      --json        Print one JSON object with the status and the problems. Each problem also gives the code of the format rule it breaks
  -v, --verbose...  Increase logging verbosity: -v info, -vv debug, -vvv trace. Warnings and errors always print. `RUST_LOG` overrides this when set
  -q, --quiet       Hide the summary line and the progress display of import and append
  -h, --help        Print help

Examples:
  pbz validate cohort.pbz
  pbz validate cohort.pbz.zip --json

The check reads metadata, the reference arrays and the dimension labels. It does not read track values.
```
