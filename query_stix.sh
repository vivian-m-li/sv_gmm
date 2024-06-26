#!/bin/bash
l=$1
r=$2
output_file=$3
STIX_DB="/scratch/Shares/layer/stix/indices/1kg_hg37_low_cov"
STIX_BUILD="/Users/vili4418/sv/stix_g/bin/stix"

CURRENT_DIR=$(pwd)
cd "$STIX_DB"

$STIX_BUILD -i "alt_sort_b" -d "1kg.ped.db" -s 500 -t DEL -l "$l" -r "$r" -g > $"$CURRENT_DIR/$output_file"

cd "$CURRENT_DIR"