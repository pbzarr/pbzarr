# BAM/CRAM fixtures

Committed binaries built from the SAM/FASTA sources in `src/`. Regenerate everything (including the d4 files and `fixtures/SHA256SUMS`) with `pixi run -- bash scripts/regen_fixtures.sh` from the repo root.

Built with samtools 1.24 (htslib 1.24) and d4tools 0.3.11.

Base qualities default to Phred 30 (`?`) unless a record notes otherwise.

## Main set: `reads.bam` / `reads.cram` (`ref1` 100bp, `ref2` 60bp)

`ref1` is 100bp and `ref2` is 60bp, both a repeating `ACGT` motif so the reference content is deterministic but not aligned with any read's own bases. Every read sits on `ref1`; `ref2` stays empty.

Records on `ref1` (0-based half-open), post default filter (`exclude_flags = 1796` drops r4's DUP flag):

| read | flag | mapq | pos | CIGAR | footprint |
|------|------|------|-----|-------|-----------|
| r1a | 99 | 60 | 10 | 20M | [10,30), mate at 25 (paired w/ r1b) |
| r1b | 147 | 60 | 25 | 20M | [25,45), mate at 10 (paired w/ r1a) |
| r2 | 0 | 60 | 40 | 10M5D10M | M[40,50) D[50,55) M[55,65) |
| r3 | 0 | 60 | 70 | 5M10N5M | M[70,75) N[75,85) M[85,90) |
| r4 | 1024 | 60 | 10 | 20M | excluded (DUP bit set) |
| r5 | 0 | 5 | 10 | 20M | [10,30) |
| r6 | 0 | 60 | 50 | 5M1I5M | M[50,55) Ins@54 M[55,60) |
| r7 | 0 | 60 | 12 | 10M | [12,22), BQ5 (`&`) at read offset 3 (ref pos 15) |
| r8 | 0 | 60 | 90 | 8M, QUAL=* | [90,98), every base passes any min_bq |

r1a/r1b overlap on [25,30); r1a starts first (coordinate-sorted fetch order), so it parks and claims [10,30), and r1b's walk skips [25,30) under the default `overlap = OverlapMode::ProperOnly` filter (r1a/r1b are flags 99/147, i.e. proper-pair, so `ProperOnly` behaves identically to `All` for this fixture).

r8's `QUAL = *` is legal and means "quality unavailable", so every base clears any `min_bq`.

### Expected depth on `ref1`, default filter

Per-record contributions (each adds 1 over the span):

- [10,30) r1a (parks, walks in full)
- [30,45) r1b, [25,30) already claimed by r1a
- [40,50) r2 M
- r2 D [50,55) contributes nothing to depth by default (`count_deletions = false`)
- [55,65) r2 M
- [70,75) r3 M
- [85,90) r3 M (N[75,85) contributes nothing)
- [10,30) r5 (`min_mapq` default 0, so included)
- [50,55) r6 M
- [55,60) r6 M (Ins@55 contributes nothing to depth)
- [12,22) r7
- [90,98) r8 (QUAL = *, so every base clears the BQ gate)

## `dref.bam` (`dref` 60bp)

One overlapping pair where the first mate carries a D run inside the overlap and the second mate matches across it:

| read | flag | pos | CIGAR | footprint |
|------|------|-----|-------|-----------|
| r8a | 99 | 10 | 5M5D10M | M[10,15) D[15,20) M[20,30), mate at 15 |
| r8b | 147 | 15 | 20M | M[15,35), mate at 10 |

Kept out of the main set because the main set's `ref2` is asserted empty and its `ref1` depth vector is pinned by the derivation above.

## `pref.bam` (`pref` 60bp)

One overlapping pair whose PROPER_PAIR bit (0x2) is unset, i.e. a discordant pair: same flags as the main set's r1a/r1b (99/147) with bit 0x2 cleared (97/145).

| read | flag | pos | CIGAR | footprint |
|------|------|-----|-------|-----------|
| r9a | 97 | 10 | 20M | [10,30), mate at 25, NOT proper-pair |
| r9b | 145 | 25 | 20M | [25,45), mate at 10, NOT proper-pair |

Overlap span [25,30). Used to pin `OverlapMode`: `ProperOnly` (mosdepth match) must NOT dedup this pair (depth 2 across the overlap), `All` (riker match) must dedup it (depth 1), `None` must not dedup it either (depth 2).

## `mref.bam` (`mref` 120bp)

Three proper pairs whose overlap spans disagree on CIGAR op class between the mates, the exact shape that a single op-class-agnostic claim bitmap gets wrong:

| read | flag | pos | CIGAR | note |
|------|------|-----|-------|------|
| m1a | 99 | 10 | 5M5D10M | D[15,20) against m1b's M |
| m1b | 147 | 15 | 20M | |
| m2a | 99 | 40 | 5M5N10M | N[45,50) against m2b's M |
| m2b | 147 | 45 | 20M | |
| m3a | 99 | 70 | 5M2I5M | insertion anchor against m3b's M |
| m3b | 147 | 74 | 10M | |

## `iref.bam` (`iref` 40bp)

Pins insertion anchoring:

| read | pos | CIGAR | note |
|------|-----|-------|------|
| r10 | 10 | 5M1I | trailing insertion, ref span [10,15) |
| r11 | 20 | 1I5M | leading insertion, ref span [20,25) |

The trailing insertion must anchor inside the read's own reference span so that a chunked import (whose per-chunk query only returns reads overlapping that chunk) still sees it.

## `nref.bam` / `nref.cram` (`nref` 60bp)

Two overlapping proper "pairs" whose records are all unnamed (`QNAME = *`):

| flag | pos | CIGAR | count | note |
|------|-----|-------|-------|------|
| 99 | 10 | 20M | x2 | mate at 25 |
| 147 | 25 | 20M | x2 | mate at 10 |

Unnamed records can't be paired with anything, so all four must count independently, on both backends, which spell an absent name differently on the wire.
