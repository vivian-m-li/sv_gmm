#!/bin/bash
input_file=$1
output_file=$2
ploidy_table=$3
reference_file=$4

GATK_BUILD="/Users/vili4418/sv/gatk-4.6.2.0/gatk"

# use gatk to cluster the SVs 
gatk SVCluster \
  -V $input_file \
  -O $output_file \
  --ploidy-table $ploidy_table \
  -R $reference_file
