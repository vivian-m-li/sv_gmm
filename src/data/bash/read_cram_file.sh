#!/bin/bash
cram_file=$1
region=$2
output_file=$3
samtools_build=$4
reference_genome=$5

$samtools_build view -b -h -T $reference_genome $cram_file $region > $output_file
$samtools_build index $output_file
