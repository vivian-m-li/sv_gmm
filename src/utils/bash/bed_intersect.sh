# !/bin/bash

file_a=$1
file_b=$2
output_file=$3

$HOME/sv/bedtools/bin/bedtools intersect -a "$file_a" -b "$file_b" -wa -wb > "$output_file"

