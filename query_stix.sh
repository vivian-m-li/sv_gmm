#!/bin/bash

l=$1
r=$2
STIX_DB="/scratch/Shares/layer/stix/indices/1kg_hg37_low_cov"
STIX_BUILD="/Users/vili4418/sv/stix_g/bin/stix"

STIX_BUILD -i "$SV_DB/alt_sort_b" -d "$SV_DB/1kg.ped.db" -s 500 -t DEL -l "$l" -r "$r" -g >> stix_output.txt
