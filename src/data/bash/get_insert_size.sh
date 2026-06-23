#!/bin/bash
sample_id=$1
input_file=$2
temp_output_dir=$3
output_dir=$4
samtools_exec=$5

# updated method from lumpy repository
"$samtools_exec" view "$input_file" | tail -n +100000 | python src/data/pairend_distro.py -r 101 -X 4 -N 100000 -o "${temp_output_dir}/${sample_id}.txt"

scp "${temp_output_dir}/${sample_id}.txt" "$output_dir" 

# use samtools to read the alignment file and calculate mean and standard deviation of insert sizes
# "$samtools_exec" view -q 5 -f 2 "$input_file" | tail -n +100000 | head -100000 | \
# awk 'function abs(x) { return x < 0 ? -x : x } 
#   { 
#       insert_size = abs($9); 
#       sum += insert_size; 
#       sum_sq += insert_size^2; 
#       count++; 
#   } 
#   END { 
#       mean = sum / count; 
#       stddev = sqrt((sum_sq / count) - (mean^2)); 
#       print mean, stddev; 
#   }' > "${sample_id}.txt"

# "$samtools_exec" view -q 5 -f 2 "$input_file" | tail -n +100000 | head -100000 | awk 'function abs(x) { return x < 0 ? -x : x } { print abs($9) }' > "${sample_id}.txt"