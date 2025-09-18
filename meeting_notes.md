# Meeting Notes

### 9/18/25

- Issues with processing long reads
  - Looked through code (cram file -> bam file for an SV region -> cigar string processing code)
  - Code was not correctly incrementing the reference position (to get the position of the long-read deletion)
  - Will soft/hard clipped split reads be an issue? Look into the prevalence of these reads, possibly ignore them in processing
  - Issue is: how can we know if a deletion in the cigar string corresponds to the SV that I'm interested in? What is the threshold?
- Discussed using the STIX long read index (in /scratch/Shares/layer/stix/indices/LR/GRCh38_STIX_indexes/)
  - Unsure if the current version of stix will return the intervals; may need to use giggle or [excord-lr](https://github.com/zhengxinchang/excord-lr)
- Replicating gnomad's merging (per Harrison's email)
  - Look into [ClusterBatch](https://broadinstitute.github.io/gatk-sv/docs/modules/cb/) in the GATK repo - what kind of data does it input? (will possibly need to generate a vcf file with the synthetic data, using pysam)
