que#!/bin/bash
l=$1
r=$2
output_file=$3
use_current_dir=$4
STIX_DB="/scratch/Shares/layer/stix/indices/LR_vivian"
STIX_BUILD="/Users/vili4418/sv/stix_g/bin/stix"

CURRENT_DIR=$(pwd)
cd "$STIX_DB"

if [ "$use_current_dir" = "True" ]; then
  output_file="$CURRENT_DIR/${output_file}"
fi

# example query: /Users/vili4418/sv/stix_g/bin/stix -i 03.1.giggle_idx_00 -d 03.1.stix_idx_00.db -s 150 -t DEL -l 4:9473951-9473951 -r 4:9474534-9474534 -g

for i in {0..22}
do
  idx=$(printf "%02d" "$i")
  $STIX_BUILD -i "03.1.giggle_idx_$idx" -d "03.1.stix_idx_$idx.db" -s 150 -t DEL -l "$l" -r "$r" -g > "${output_file}_${i}.txt"
done

cd "$CURRENT_DIR"