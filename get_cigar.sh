#!/bin/bash
cram_file=$1
region=$2
output_file=$3
use_current_dir=$4
SAMTOOLS_BUILD="/Users/vili4418/sv/samtools-1.21/samtools"

$SAMTOOLS_BUILD view $cram_file $region > $output_file
