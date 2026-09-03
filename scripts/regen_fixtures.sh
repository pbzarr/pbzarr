#!/usr/bin/env bash
# Regenerate every committed binary under fixtures/ from the SAM/FASTA
# sources in fixtures/bam/src and the inline d4 formulas below.
# Run from the repo root: pixi run -- bash scripts/regen_fixtures.sh
set -euo pipefail

cd "$(dirname "$0")/.."

src=fixtures/bam/src
bam=fixtures/bam
d4=fixtures/d4
mkdir -p "$bam" "$d4"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# BAM families
for stem in reads dref pref mref iref nref; do
  samtools sort -o "$bam/$stem.bam" "$src/$stem.sam"
  samtools index "$bam/$stem.bam"
done

cp "$src/ref.fa" "$bam/ref.fa"
samtools faidx "$bam/ref.fa"
samtools view -C -T "$bam/ref.fa" -o "$bam/reads.cram" "$bam/reads.bam"
samtools index "$bam/reads.cram"

cp "$src/nref.fa" "$bam/nref.fa"
samtools faidx "$bam/nref.fa"
samtools view -C -T "$bam/nref.fa" -o "$bam/nref.cram" "$bam/nref.bam"
samtools index "$bam/nref.cram"

# banded_1k: chr1 1000bp, bands of 10 with value (i % 50) + 1
printf 'chr1\t1000\n' > "$tmp/banded_1k.sizes"
for ((i = 0; i < 100; i++)); do
  printf 'chr1\t%d\t%d\t%d\n' $((i * 10)) $(((i + 1) * 10)) $((i % 50 + 1))
done > "$tmp/banded_1k.bedgraph"
d4tools create --genome "$tmp/banded_1k.sizes" "$tmp/banded_1k.bedgraph" "$d4/banded_1k.d4"

# cohort_s{0,1,2}: chr1 10000bp, value base + (i % 50) + 1 with base 0/100/200
printf 'chr1\t10000\n' > "$tmp/cohort.sizes"
for s in 0 1 2; do
  base=$((s * 100))
  for ((i = 0; i < 1000; i++)); do
    printf 'chr1\t%d\t%d\t%d\n' $((i * 10)) $(((i + 1) * 10)) $((base + i % 50 + 1))
  done > "$tmp/cohort_s$s.bedgraph"
  d4tools create --genome "$tmp/cohort.sizes" "$tmp/cohort_s$s.bedgraph" "$d4/cohort_s$s.d4"
done

# multi_contig: header order chr2 then chr1
printf 'chr2\t500\nchr1\t300\n' > "$tmp/multi_contig.sizes"
printf 'chr2\t0\t500\t100\nchr1\t0\t300\t7\n' > "$tmp/multi_contig.bedgraph"
d4tools create --genome "$tmp/multi_contig.sizes" "$tmp/multi_contig.bedgraph" "$d4/multi_contig.d4"

# two_contig_const7: constant 7 across both contigs
printf 'chr1\t1000\nchr2\t500\n' > "$tmp/two_contig_const7.sizes"
printf 'chr1\t0\t1000\t7\nchr2\t0\t500\t7\n' > "$tmp/two_contig_const7.bedgraph"
d4tools create --genome "$tmp/two_contig_const7.sizes" "$tmp/two_contig_const7.bedgraph" "$d4/two_contig_const7.d4"

# fixed_point: fix-point encoding (denominator), rejected on import
printf 'chr1\t10\n' > "$tmp/fixed_point.sizes"
printf 'chr1\t0\t10\t2\n' > "$tmp/fixed_point.bedgraph"
d4tools create --genome "$tmp/fixed_point.sizes" --denominator 100 "$tmp/fixed_point.bedgraph" "$d4/fixed_point.d4"

# checksums
if command -v sha256sum >/dev/null 2>&1; then
  sha=(sha256sum)
else
  sha=(shasum -a 256)
fi
(
  cd fixtures
  "${sha[@]}" \
    bam/ref.fa bam/ref.fa.fai \
    bam/reads.bam bam/reads.bam.bai bam/reads.cram bam/reads.cram.crai \
    bam/dref.bam bam/dref.bam.bai \
    bam/pref.bam bam/pref.bam.bai \
    bam/mref.bam bam/mref.bam.bai \
    bam/iref.bam bam/iref.bam.bai \
    bam/nref.fa bam/nref.fa.fai \
    bam/nref.bam bam/nref.bam.bai bam/nref.cram bam/nref.cram.crai \
    d4/banded_1k.d4 d4/cohort_s0.d4 d4/cohort_s1.d4 d4/cohort_s2.d4 \
    d4/multi_contig.d4 d4/two_contig_const7.d4 d4/fixed_point.d4
) > fixtures/SHA256SUMS
