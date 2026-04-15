# !/bin/bash

bedtools_bin=$1
file_a=$2
file_b=$3
output_file=$4

"$bedtools_bin" intersect -a "$file_a" -b "$file_b" -wa -wb > "$output_file"

