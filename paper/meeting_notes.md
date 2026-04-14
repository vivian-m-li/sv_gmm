# Meeting Notes

### 9/24/25

Using stix to pull long reads for a region

1. query stix for long reads, similar to how we're doing it for high-cov short reads with multiple shards (split by samples)

- long read index is found in: `/scratch/Shares/layer/stix/indices/LR/GRCh38_STIX_indexes`
- example long reads query: stix -i 03.1.giggle_idx_00 -d 0.3.1.meta.ped.00.db -s 150 -t DEL -l 4:9473951-9473951 -r 4:9474534-9474534
- for short reads, slop needs to be higher than insert size to reduce false negatives, and for long reads slop should be shorter (~150)
- to give Vivian access to the stix directory: copy the directory to the stix/indices directory

2. the patched version of stix will return: file_id, fild_name, chrm, left_start, left_end, chrm, right_start, right_end, paired/split read

- our hypothesis is that we think the deletion region is right_start - left_end
- to verify our hypothesis, look at the cigar string of the primary/supplementary alignments
  - the primary alignment will have soft- or hard-clipped reads in the same place where the supplementary read should be (and vice versa)
  - the deleted region will be p1 + xS - p2, where
    - p1 is the region of the primary read
    - p2 is the region of the supplementary read
    - xS is the number of bps that are soft-clipped on the primary read (should align with the xM region in the supplementary read)
  - check if the primary/supplementary reads are forward/reverse strand
  - use samtools view to get the cigar string for the region

### 9/18/25

- Issues with processing long reads
  - Looked through code (cram file -> bam file for an SV region -> cigar string processing code)
  - Code was not correctly incrementing the reference position (to get the position of the long-read deletion)
  - Will soft/hard clipped split reads be an issue? Look into the prevalence of these reads, possibly ignore them in processing
  - Issue is: how can we know if a deletion in the cigar string corresponds to the SV that I'm interested in? What is the threshold?
- Discussed using the STIX long read index
  - Unsure if the current version of stix will return the intervals; may need to use giggle or [excord-lr](https://github.com/zhengxinchang/excord-lr)
- Replicating gnomad's merging (per Harrison's email)
  - Look into [ClusterBatch](https://broadinstitute.github.io/gatk-sv/docs/modules/cb/) in the GATK repo - what kind of data does it input? (will possibly need to generate a vcf file with the synthetic data, using pysam)
- From GATK's [SVCluster](https://gatk.broadinstitute.org/hc/en-us/articles/27007962371099-SVCluster-BETA), SVs are clustered based on:
  - Matching SV type. DEL and DUP are considered matching SV types if --enable-cnv is used and merged into a multi-allelic CNV type.
  - Matching breakend strands (BND and INV only)
  - Interval reciprocal overlap (inapplicable for BNDs).
  - Distance between corresponding event breakends (breakend window).
  - Sample reciprocal overlap, based on carrier status determined by available genotypes (GT fields). If no GT fields are called for a given variant, the tool attempts to find carriers based on copy number (CN field) and sample ploidy (as determined by the ECN FORMAT field).
