# !/bin/bash

del_file=$1
gene_file=$2
output_file=$3

$HOME/sv/bedtools/bin/bedtools intersect -a "$del_file" -b "$gene_file" -wa -wb > "$output_file"

