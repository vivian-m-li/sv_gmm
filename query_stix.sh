#!/bin/bash
l=$1
r=$2
index_path=$3
database_path=$4
num_shards=$5
output_file=$6

if [ "$num_shards" -eq 1 ]; then
  stix -i "alt_sort_b" -d "1kg.ped.db" -s 500 -t DEL -l "$l" -r "$r" -g > "${output_file}_0.txt"
else
  for (( idx=0; idx<num_shards; idx++ ))
  do
    stix -i "${index_path}_${idx}" -d "${database_path}_${idx}.ped.db" -s 500 -t DEL -l "$l" -r "$r" -g > "${output_file}_${idx}.txt"
  done
fi
