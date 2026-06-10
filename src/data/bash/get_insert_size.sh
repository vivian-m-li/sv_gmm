#!/bin/bash
sample_id=$1
input_file=$2
temp_output_dir=$3
output_dir=$4
samtools_exec=$5

CURRENT_DIR=$(pwd)
cd "$temp_output_dir"

# use samtools to read the alignment file and calculate mean and standard deviation of insert sizes
"$samtools_exec" view -q 5 -f 2 "$input_file" | tail -n +100000 | head -5000000 | \
awk 'function abs(x) { return x < 0 ? -x : x } 
  { 
      insert_size = abs($9); 
      sum += insert_size; 
      sum_sq += insert_size^2; 
      count++; 
  } 
  END { 
      mean = sum / count; 
      stddev = sqrt((sum_sq / count) - (mean^2)); 
      print mean, stddev; 
  }' > "${sample_id}.txt"

scp "${sample_id}.txt" "$output_dir" 

cd "$CURRENT_DIR"