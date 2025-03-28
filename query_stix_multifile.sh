#!/bin/bash
l=$1
r=$2
output_file=$3
use_current_dir=$4
STIX_DB="/scratch/Shares/layer/stix/indices/1kg_high_coverage_vivian"
STIX_BUILD="/Users/vili4418/sv/stix_g/bin/stix"

CURRENT_DIR=$(pwd)
cd "$STIX_DB"

if [ "$use_current_dir" = "True" ]; then
  output_file="$CURRENT_DIR/${output_file}"
fi

for idx in {0..7}
do
  $STIX_BUILD -i "shard_$idx" -d "shard_$idx.ped.db" -s 500 -t DEL -l "$l" -r "$r" -g > "${output_file}_${idx}.txt"
done

cd "$CURRENT_DIR"