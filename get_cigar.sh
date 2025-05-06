#!/bin/bash
cram_file=$1
region=$2
output_file=$3
reference_genome=$4
SAMTOOLS_BUILD="/Users/vili4418/sv/samtools-1.21/samtools"
REFERENCE_GENOME="Users/vili4418/sv/sv_gmm/long_reads/hg38.fa"

$SAMTOOLS_BUILD view -b -h -T $reference_genome $cram_file $region > $output_file
$SAMTOOLS_BUILD index $output_file
