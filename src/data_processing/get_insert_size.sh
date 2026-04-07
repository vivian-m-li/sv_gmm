#!/bin/bash
sample_id=$1
input_file=$2

# Removed incorrect reassignment of $HOME
CURRENT_DIR=$(pwd)
cd /scratch/Users/vili4418/insert_size_files

# use samtools to read the alignment file
"$HOME/sv/samtools-1.21/samtools" view -q 5 -f 2 "$input_file" | tail -n +100000 | head -5000000 | awk 'function abs(x) { return x < 0 ? -x : x } { sum+=abs($9); count++ } END {print sum/count}' > "${sample_id}.txt"

scp "${sample_id}.txt" "${HOME}/sv/sv_gmm/1kgp/insert_size_files/" 

cd "$CURRENT_DIR"