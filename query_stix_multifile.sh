#!/bin/bash
l=$1
r=$2
output_file=$3
STIX_DB="/scratch/Shares/layer/stix/indices/1kg_high_coverage"
STIX_BUILD="/Users/vili4418/sv/stix_g/bin/stix"

CURRENT_DIR=$(pwd)
cd "$STIX_DB"

for idx in {0..9}
do
  $STIX_BUILD -i "giggle_index_$idx" -d "1kg.$idx.ped.db" -s 500 -t DEL -l "$l" -r "$r" -g > $"$CURRENT_DIR/${output_file}_${idx}.txt"
done

cd "$CURRENT_DIR"