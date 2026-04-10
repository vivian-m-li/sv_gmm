#!/bin/bash
l=$1
r=$2
stix_path=$3
index_path=$4
database_path=$5
num_shards=$6
output_file=$7
stix_bin=$8

CURRENT_DIR=$(pwd)
cd "$stix_path"

if [ "$num_shards" -eq 1 ]; then
  "$stix_bin" -i "$index_path" -d "$database_path" -s 500 -t DEL -l "$l" -r "$r" -g > "${CURRENT_DIR}/${output_file}.txt"
else
  for (( idx=0; idx<num_shards; idx++ ))
  do
    "$stix_bin" -i "${index_path}_${idx}" -d "${database_path}_${idx}.ped.db" -s 500 -t DEL -l "$l" -r "$r" -g > "${CURRENT_DIR}/${output_file}_${idx}.txt"
  done
fi

cd "$CURRENT_DIR"